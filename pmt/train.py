"""Single-device SSL training with the reference optimizer, cadence and loss RNG state."""

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pmt.config import Config
from pmt.data import Splits, infer_data_config, load_dataset, ssl_loader
from pmt.probe import BUDGETS

logger = logging.getLogger(__name__)


def mixed_dtype(precision: str | None, device: torch.device) -> torch.dtype | None:
    """Resolve explicit precision or the reference's automatic device-supported dtype."""
    if precision in {'bf16', 'fp16'}:
        return {'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    if precision not in {None, 'none'}:
        raise ValueError(f'Unknown training precision {precision!r}')
    if device.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_optimizer(model: nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer, tuple[list, list]]:
    """Build Adam groups and the distinct reference gate/base clipping lists."""
    from pmt.model import parameter_groups

    base, gate_params, clip_gates = parameter_groups(model)
    clipped = {id(parameter) for parameter in clip_gates}
    groups = [dict(params=base, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)]
    if gate_params:
        groups.append(dict(params=gate_params, lr=cfg.optim.lr * cfg.optim.gate_lr, weight_decay=0.0))
    params = [parameter for group in groups for parameter in group['params']]
    # The reference enables fused Adam whenever every parameter is a CUDA float tensor.
    fused = bool(params) and all(p.is_floating_point() and p.device.type == 'cuda' for p in params)
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), **({'fused': True} if fused else {}))
    return optimizer, (clip_gates, [p for p in model.parameters() if id(p) not in clipped])


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, epochs: int) -> Any:
    """Preserve epoch-end cosine stepping and fractional warmup bootstrap."""
    raw = cfg.optim.warmup
    warmup = min(max(0, round(raw * epochs) if 0 < raw < 1 else int(raw)), epochs - 1)
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.optim.min_lr)

    def schedule(base_lr: float) -> Any:
        floor = min(1.0, max(0.0, cfg.optim.min_lr / base_lr))
        remaining = max(1, epochs - warmup)

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            t = min(max(1, epoch - warmup + 1), remaining)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / remaining))

        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(group['lr']) for group in optimizer.param_groups])


def probe_epochs(cfg: Config, epochs: int) -> set[int]:
    """Resolve explicit probes or the start/interval schedule plus the final epoch."""
    if cfg.train.probe_epochs:
        planned = {int(e) for e in cfg.train.probe_epochs if 0 <= int(e) <= epochs}
    else:
        if cfg.train.probe_interval <= 0:
            raise ValueError('Probe interval must be positive')
        planned = set(range(cfg.train.probe_start, epochs, cfg.train.probe_interval)) - {0}
    return planned | {epochs}


def train(
    cfg: Config,
    run_dir: str | Path,
    *,
    model: nn.Module | None = None,
    loss_fn: Any = None,
    probe: Any = None,
    splits: Splits | None = None,
) -> dict:
    """Train SSL, retain sampler state, and write the encoder and per-probe metrics."""
    if cfg.train.epochs <= 0 or (cfg.train.steps is not None and cfg.train.steps <= 0):
        raise ValueError('Training epochs and step budgets must be positive')
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    dtype = mixed_dtype(cfg.train.precision, device)
    scaler = torch.amp.GradScaler(
        'cuda',
        init_scale=128,
        growth_factor=2.0,
        growth_interval=2000,
        backoff_factor=0.5,
        enabled=device.type == 'cuda' and dtype == torch.float16,
    )
    scaler._max_scale = 1e8  # Reference attribute, unused by Torch 2.10.
    scaled = scaler.is_enabled()
    if device.type == 'cuda':
        # Timed kernel selection makes same-seed runs differ; fixed selection repeats them.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.get_device_capability(device)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    splits = load_dataset(cfg) if splits is None else splits
    cfg = infer_data_config(cfg, splits)
    if model is None:
        from pmt.model import SSLModel

        model = SSLModel(cfg)
    model.to(device)
    if loss_fn is None:
        from pmt.losses import SSLLoss

        loss_fn = SSLLoss()
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device)
    if probe is None:
        from pmt.probe import Probe

        probe = Probe(cfg)
    loader = ssl_loader(splits, cfg)
    if not len(loader):
        raise ValueError('SSL TRAIN is empty')
    budget = cfg.train.steps
    if budget is not None and cfg.train.epoch_steps <= 0:
        raise ValueError('Steps per pseudo-epoch must be positive')
    epochs = math.ceil(budget / cfg.train.epoch_steps) if budget is not None else cfg.train.epochs
    optimizer, (gate_params, base_params) = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, cfg.train.epochs)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history, step, real_epoch, iterator = [], 0, 0, None
    best_scores: dict[float, tuple[float, float, int]] = {}
    scheduled = probe_epochs(cfg, epochs)

    def save_encoder(epoch: int, path: Path, **metadata: Any) -> None:
        temporary = path.with_suffix('.tmp')
        torch.save(
            dict(
                state_dict=model.encoder.state_dict(),
                cfg=cfg.to_dict(),
                epoch=epoch,
                mean=splits.mean,
                std=splits.std,
                **metadata,
            ),
            temporary,
        )
        temporary.replace(path)

    for epoch in range(epochs):
        model.train()
        count, totals = 0, {}
        epoch_lr = [group['lr'] for group in optimizer.param_groups]
        if budget is None:
            iterator, real_epoch = iter(loader), epoch
        limit = min(cfg.train.epoch_steps, budget - step) if budget is not None else len(loader)
        for _ in range(limit):
            if iterator is None:
                iterator = iter(loader)
            try:
                one, two = next(iterator)
            except StopIteration:
                real_epoch += 1
                iterator = iter(loader)
                one, two = next(iterator)
            model.current_epoch = real_epoch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                loss, terms = loss_fn(model(one.to(device), two.to(device)), cfg, real_epoch, epochs, step=step)
            loss_value = float(loss.detach())
            if not scaled and not math.isfinite(loss_value):
                raise FloatingPointError(f'Nonfinite SSL loss at step {step}')
            scaler.scale(loss).backward()
            # Reference's zero-weight L2 gives unused heads zero grads, so Adam still decays them.
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(gate_params, 100, error_if_nonfinite=not scaled)
            nn.utils.clip_grad_norm_(base_params, 50, error_if_nonfinite=not scaled)
            scaler.step(optimizer)
            scaler.update()
            step, count = step + 1, count + 1
            for key, value in {'loss': loss_value, **terms}.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        scheduler.step()
        # A skipped scaled update can leave nonfinite metrics, which JSON cannot encode.
        averages = {
            key: None if scaled and not math.isfinite(value) else value / count for key, value in totals.items()
        }
        row = dict(epoch=epoch + 1, step=step, lr=epoch_lr, **averages)
        retained = cfg.train.checkpoints
        if cfg.train.retain and epoch + 1 in scheduled and (not retained or epoch + 1 in retained):
            (run_dir / 'checkpoints').mkdir(exist_ok=True)
            save_encoder(epoch + 1, run_dir / 'checkpoints' / f'epoch{epoch + 1}.pt')
        if epoch + 1 in scheduled:
            training = model.training
            try:
                row.update(probe.evaluate(model.encoder, splits, epoch + 1))
            finally:
                model.train(training)
            for label_budget, name in zip(BUDGETS, ('best_encoder.pt', 'best_encoder_5pct.pt')):
                prefix = f'probe/svm/lp{label_budget:.3f}/'
                if prefix + 'top1_acc' not in row:
                    continue
                score = (row[prefix + 'top1_acc'], row[prefix + 'mf1'], -(epoch + 1))
                if label_budget not in best_scores or score > best_scores[label_budget]:
                    best_scores[label_budget] = score
                    save_encoder(epoch + 1, run_dir / name, svm_params=probe.best_params, selection_budget=label_budget)
        history.append(row)
        (run_dir / 'metrics.json').write_text(json.dumps({'history': history}, indent=2, allow_nan=False) + '\n')
        logger.info(
            'epoch=%d step=%d loss=%s', epoch + 1, step, 'unavailable' if row['loss'] is None else f"{row['loss']:.6f}"
        )
    save_encoder(epochs, run_dir / 'encoder.pt')
    torch.save(
        dict(
            model=model.state_dict(),
            loss=loss_fn.state_dict() if isinstance(loss_fn, nn.Module) else None,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            step=step,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        ),
        run_dir / 'training.pt',
    )
    return {'history': history}
