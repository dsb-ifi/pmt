"""A bounded CPU loop verifies cadence, parameter updates and persistent loss state."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch import nn

import pmt.train as training
from pmt.config import Config, load_run, runs
from pmt.data import Split, Splits
from pmt.probe import BUDGETS, Probe
from pmt.train import mixed_dtype, train


class Tiny(nn.Module):
    """Expose an encoder with the training model's two-view call shape."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(6, 2)
        self.unused = nn.Parameter(torch.ones(1))

    def forward(self, one, two):
        return self.encoder(one.flatten(1)) - self.encoder(two.flatten(1))


class Objective(nn.Module):
    """Track each global step as persistent loss state."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.register_buffer('count', torch.zeros((), dtype=torch.long))
        self.dtype = dtype

    def forward(self, out, cfg, epoch, total_epochs, *, step):
        assert step == self.count.item()
        assert out.dtype == self.dtype
        self.count += 1
        return out.square().mean(), {}


@pytest.mark.parametrize(
    'precision,dtype',
    [(None, torch.float32), ('none', torch.float32), ('bf16', torch.bfloat16), ('fp16', torch.float16)],
)
def test_cpu_loop(tmp_path: Path, precision: str | None, dtype: torch.dtype) -> None:
    values = load_run(next(iter(runs()))).to_dict()
    values['data'].update(noise=True, noise_std=0.2, time_warp=False, magnitude_warp=False)
    values['optim'].update(batch_size=2, lr=0.01, min_lr=0.001, weight_decay=1e-7, gate_lr=1.0, warmup=0)
    values['train'].update(seed=7, device='cpu', precision=precision, epochs=3, steps=5)
    values['train'].update(epoch_steps=2, probe_epochs=[], probe_interval=1, probe_start=1, retain=True, checkpoints=[])
    cfg = Config(values)
    split = Split(torch.arange(24).float().view(4, 1, 6), torch.tensor([0, 1, 0, 1]))
    model, objective, seen = Tiny(), Objective(dtype), []
    before = model.encoder.weight.detach().clone()
    probe = SimpleNamespace(evaluate=lambda encoder, splits, epoch: seen.append(epoch) or {'top1': 0.5})
    result = train(cfg, tmp_path, model=model, loss_fn=objective, probe=probe, splits=Splits(split, split, split))
    assert seen == [1, 2, 3] and objective.count.item() == 5
    assert not torch.equal(before, model.encoder.weight) and model.training and model.unused.item() < 1
    assert json.loads((tmp_path / 'metrics.json').read_text()) == result
    assert torch.load(tmp_path / 'training.pt', weights_only=False)['loss']['count'].item() == 5
    assert sorted(p.name for p in (tmp_path / 'checkpoints').glob('*.pt')) == ['epoch1.pt', 'epoch2.pt', 'epoch3.pt']


@pytest.mark.parametrize('precision', [None, 'none', 'bf16', 'fp16'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('supported', [False, True])
def test_mixed_dtype(precision: str | None, device: str, supported: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock capability detection without allocating a CUDA tensor or context."""

    def capability() -> bool:
        assert device == 'cuda' and precision in {None, 'none'}
        return supported

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', capability)
    expected = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(precision)
    if precision in {None, 'none'} and device == 'cuda':
        expected = torch.bfloat16 if supported else torch.float16
    assert mixed_dtype(precision, torch.device(device)) == expected
    with pytest.raises(ValueError, match='Unknown training precision'):
        mixed_dtype('invalid', torch.device(device))


@pytest.mark.parametrize('major', [7, 8])
def test_cuda_backends_before_model(major: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Check backend setup on both sides of the TF32 capability threshold without a CUDA context."""
    device = torch.device('cuda:0')
    cudnn = SimpleNamespace(benchmark=True, deterministic=True, allow_tf32=False)
    matmul = SimpleNamespace(allow_tf32=False)
    monkeypatch.setattr(torch, 'backends', SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)))
    selected = []
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda actual: (major, 0) if actual == device else (0, 0))

    def construct(cfg):
        assert selected == [device]
        assert not cudnn.benchmark and not cudnn.deterministic
        assert matmul.allow_tf32 == cudnn.allow_tf32 == (major >= 8)
        raise RuntimeError('model construction reached')

    monkeypatch.setitem(sys.modules, 'pmt.model', SimpleNamespace(SSLModel=construct))
    cfg = load_run(next(iter(runs())), ['train.device=cuda', 'train.precision=bf16'])
    with pytest.raises(RuntimeError, match='model construction reached'):
        train(cfg, tmp_path, splits=object())


@pytest.mark.parametrize('finite_loss', [True, False])
def test_fp16_overflow_recovers(finite_loss: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use real CPU GradScaler to exercise skip, backoff and the next finite training update."""
    model = nn.Module()
    model.encoder = nn.Linear(1, 1, bias=False)
    model.forward = lambda one, two: model.encoder(one)
    weight = model.encoder.weight
    with torch.no_grad():
        weight.fill_(1.0)
    scaler_type = torch.amp.GradScaler
    scalers, before_steps = [], []

    def cpu_scaler(device, *, enabled, **kwargs):
        scaler = scaler_type('cpu', **kwargs)
        scalers.append(scaler)
        return scaler

    def objective(out, cfg, epoch, total_epochs, *, step):
        before_steps.append((weight.item(), scalers[0].get_scale()))
        factor = (torch.finfo(torch.float32).max / 2 if finite_loss else float('inf')) if step == 0 else 1.0
        return weight.squeeze() * factor, {}

    monkeypatch.setattr(torch.amp, 'GradScaler', cpu_scaler)
    values = load_run(next(iter(runs()))).to_dict()
    values['train'].update(
        device='cpu', precision='fp16', epochs=1, steps=2, epoch_steps=2, probe_epochs=[1], retain=False
    )
    batch = torch.ones(2, 1, 1)
    monkeypatch.setattr(training, 'ssl_loader', lambda *args: [(batch, batch)] * 2)
    split = Split(batch, torch.zeros(2))
    metrics = train(
        Config(values),
        tmp_path,
        model=model,
        loss_fn=objective,
        probe=SimpleNamespace(evaluate=lambda *args: {}),
        splits=Splits(split, split, split),
    )
    assert before_steps == [(1.0, 128.0), (1.0, 64.0)]
    assert weight.item() < 1.0 and scalers[0].get_scale() == 64.0
    assert (metrics['history'][0]['loss'] is not None) == finite_loss
    assert json.loads((tmp_path / 'metrics.json').read_text()) == metrics


def test_best_encoders_and_undefined_cue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save each winning probe epoch and finish training when the cue half-life is undefined."""
    from pmt import cue

    labels = [0, 0, 1, 1]
    features = [[-2.0], [-1.0], [1.0], [2.0]]
    packet = cue.fit(
        dict(features=features, labels=labels), dict(features=features[::-1], labels=labels, gap_ids=[-1, -1, 1, 2])
    )
    assert np.isnan(packet['retention_metrics']['cue_eos_half_life@0.50'])
    monkeypatch.setattr(cue, 'evaluate', lambda *args, **kwargs: {'by_difficulty': {'_test': packet}})
    scores = [
        ((0.8, 0.6), (0.9, 0.6)),
        ((0.8, 0.7), (0.8, 0.9)),
        ((0.8, 0.7), (0.95, 0.5)),
        ((0.78, 0.95), (0.92, 0.99)),
    ]
    weights, params = [], []

    def classification(self, encoder, splits, *, reset, fixed):
        weights.append(encoder.weight.detach().clone())
        self.best_params = {b: dict(C=float(2 ** len(weights)), gamma=b) for b in BUDGETS}
        params.append(self.best_params)
        return {
            f'probe/svm/lp{b:.3f}/{metric}': value
            for b, pair in zip(BUDGETS, scores[len(weights) - 1])
            for metric, value in zip(('top1_acc', 'mf1'), pair)
        }

    monkeypatch.setattr(Probe, '_classification', classification)
    values = load_run(next(iter(runs()))).to_dict()
    values['train'].update(
        device='cpu', precision=None, epochs=4, steps=4, epoch_steps=1, probe_epochs=[1, 2, 3, 4], retain=False
    )
    values['probe']['tasks'] = ['linear', 'cue']
    batch = torch.arange(12).float().reshape(2, 1, 6)
    monkeypatch.setattr(training, 'ssl_loader', lambda *args: [(batch, torch.zeros_like(batch))])
    split = Split(batch, torch.tensor([0, 1]))
    metrics = train(
        Config(values), tmp_path, model=Tiny(), loss_fn=Objective(torch.float32), splits=Splits(split, split, split)
    )

    for name, epoch, budget in (('best_encoder.pt', 2, 0.01), ('best_encoder_5pct.pt', 3, 0.05)):
        checkpoint = torch.load(tmp_path / name, weights_only=True)
        assert checkpoint['epoch'] == epoch and checkpoint['selection_budget'] == budget
        assert torch.equal(checkpoint['state_dict']['weight'], weights[epoch - 1])
        assert checkpoint['svm_params'] == params[epoch - 1]
    assert torch.load(tmp_path / 'encoder.pt', weights_only=True)['epoch'] == 4
    assert json.loads((tmp_path / 'metrics.json').read_text()) == metrics
    for row in metrics['history']:
        assert row['cue/eos_test/cue_eos_half_life@0.50'] is None
        assert row['cue/eos_test/cue_eos_roc_auc'] == packet['retention_metrics']['cue_eos_roc_auc']
