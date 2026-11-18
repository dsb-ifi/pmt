"""Table 2 cue-retention probe."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import RandomSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pmt.probe import frozen, metrics

DIFFICULTIES = ((5.0, 26), (0.5, 100), (0.3, 48), (0.2, 24), (0.15, 16), (0.1, 12))


def inject(x: torch.Tensor, amplitude: float, width: int, generator: torch.Generator) -> tuple:
    """Inject the reference Hann cue and return centers, presence and raw window indices."""
    batch, _, length = x.shape
    if length < 25:
        raise ValueError('Cue sequences must contain at least one 25-sample window')
    windows = 1 + (length - 25) // 13
    width = int(max(4, min(width, max(length - 1, 1))))
    low = 1 if windows > 2 else 0
    presence = (torch.rand(batch, generator=generator) < 0.5).to(x.device)
    centers = torch.full((batch,), length, dtype=torch.long, device=x.device)
    indices = torch.full((batch,), windows, dtype=torch.long, device=x.device)
    modified = x.clone()
    if presence.any():
        count = int(presence.sum())
        ratios = torch.tensor([0.0, 0.25, 0.5, 0.75])[torch.randint(0, 4, (count,), generator=generator)].to(x.device)
        chosen = ((1 - ratios).clamp(0, 1) * (windows - 1)).round().long().clamp(low, windows - 1)
        starts = (chosen * 13 + (25 - width) // 2).clamp(0, length - width - 1)
        pulse = torch.hann_window(width, dtype=x.dtype, device=x.device)
        positives = torch.nonzero(presence, as_tuple=False).squeeze(-1)
        indices[positives] = chosen
        for row, start in zip(positives.tolist(), starts.tolist()):
            end = start + width
            outside = torch.cat((modified[row, 0, :start], modified[row, 0, end:]))
            scale = outside.std(unbiased=False).clamp_min(1e-6)
            centers[row] = start + width // 2
            modified[row, 0, start:end] += amplitude * scale * pulse
    return modified, centers, presence, indices


def memory_features(states: torch.Tensor) -> torch.Tensor:
    """Pool PMA state slots while retaining the reference window dimension."""
    if states.ndim != 4 or not states.size(1) or not torch.isfinite(states).all():
        raise ValueError('Cue probing requires finite [B,W,S,D] PMA states')
    return states.mean(dim=2).float()


def fit(train: dict, test: dict) -> dict:
    """Fit the EOS logistic probe and compute raw-gap retention summaries."""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs')
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train['features'], train['labels'])
    predicted = model.predict(test['features'])
    probs = model.predict_proba(test['features'])[:, 1]
    y, gaps = np.asarray(test['labels']), np.asarray(test['gap_ids'])
    train_pred, train_probs = model.predict(train['features']), model.predict_proba(train['features'])[:, 1]
    train_metrics, test_metrics = metrics(train['labels'], train_pred), metrics(y, predicted)
    train_metrics['loss'] = float(log_loss(train['labels'], train_probs))
    test_metrics['loss'] = float(log_loss(y, probs))
    positive = gaps >= 0
    if not positive.any():
        raise ValueError('Cue retention requires positive examples')
    per_gap, per_f1, pos_probs, neg_probs = {}, {}, {}, {}
    for gap in np.unique(gaps):
        mask = gaps == gap
        if gap >= 0:
            per_gap[int(gap)] = float((predicted[mask] == y[mask]).mean())
            per_f1[int(gap)] = float(f1_score(y[mask], predicted[mask], average='macro', zero_division=0))
            pos_probs[int(gap)] = float(probs[mask].mean())
        else:
            neg_probs[int(gap)] = float(probs[mask].mean())
    eligible = [gap for gap, accuracy in per_gap.items() if accuracy >= 0.5]
    mean = float(np.mean([per_gap[g] for g in sorted(per_gap)]))
    weighted = float(
        np.average([per_gap[g] for g in sorted(per_gap)], weights=[(gaps == g).sum() for g in sorted(per_gap)])
    )
    auc = float(roc_auc_score(y, probs)) if np.unique(y).size == 2 else float('nan')
    retention = dict(
        cue_eos_pos_acc=float((predicted[positive] == y[positive]).mean()),
        cue_eos_auc=mean,
        cue_eos_mean_acc_unweighted=mean,
        cue_eos_mean_acc_weighted=weighted,
        cue_eos_roc_auc=auc,
    )
    retention['cue_eos_acc'] = retention['cue_eos_pos_acc']
    retention['cue_eos_half_life@0.50'] = float(max(eligible)) if eligible else float('nan')
    test_metrics.update(retention)
    return dict(
        status='ok',
        train_metrics=train_metrics,
        val_metrics=test_metrics,
        val_per_gap=per_gap,
        val_per_gap_mf1=per_f1,
        val_prob_per_gap_pos=pos_probs,
        val_prob_per_gap_neg=neg_probs,
        retention_metrics=retention,
    )


def evaluate(
    encoder: torch.nn.Module,
    splits: object,
    cfg: object,
    *,
    generators: tuple[torch.Generator, torch.Generator] | None = None,
) -> dict:
    """Replay all Table 2 difficulties with common clean inputs and seeded raw gaps."""
    from pmt.data import transform

    rng, sampler = generators or (
        torch.Generator().manual_seed(cfg.train.seed),
        torch.Generator().manual_seed(cfg.train.seed),
    )
    device = next(encoder.parameters()).device
    results = {}
    with frozen(encoder):
        for amplitude, width in DIFFICULTIES:
            packets = []
            for split, training in ((splits.train, True), (splits.val, False)):
                indices = (
                    torch.tensor(list(RandomSampler(range(len(split.x)), generator=sampler)))
                    if training
                    else torch.arange(len(split.x))
                )
                features, labels, gaps = [], [], []
                for start in range(0, len(indices), cfg.optim.batch_size):
                    x = torch.stack(
                        [
                            transform(split.x[i], cfg, splits.mean, splits.std)
                            for i in indices[start : start + cfg.optim.batch_size]
                        ]
                    ).to(device)
                    modified, _, presence, positions = inject(x, amplitude, width, rng)
                    states = encoder(modified, return_all_states=True).pma_states
                    last = memory_features(states)[:, -1]
                    features.append(last.to('cpu', copy=True, memory_format=torch.contiguous_format))
                    labels.append(presence.long().cpu())
                    raw_windows = 1 + (x.shape[-1] - 25) // 13
                    gaps.append(torch.where(presence, raw_windows - 1 - positions, -1).cpu())
                packets.append(
                    dict(
                        features=torch.cat(features).numpy(),
                        labels=torch.cat(labels).numpy(),
                        gap_ids=torch.cat(gaps).numpy(),
                    )
                )
            results[f'_amp{amplitude:g}_w{width}_ro_mean'] = fit(*packets)
    return dict(status='ok', probe_mode='eos', by_difficulty=results, cue_difficulty_grid=list(DIFFICULTIES))
