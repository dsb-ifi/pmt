"""Table 3 forecasting probe."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import json

from pmt.probe import frozen

MILESTONES = (2, 6, 10, 20, 30, 40, 50)
WIDE23 = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
)
SOURCE13 = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def samples(reps: np.ndarray, data: np.ndarray, horizon: int, drop: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pair each origin with its next horizon steps without crossing split boundaries."""
    if horizon <= 0 or drop < 0 or len(data) <= horizon + drop or len(reps) != len(data):
        raise ValueError('Forecast split must contain at least horizon + drop + 1 aligned rows')
    data_b = data[np.newaxis]
    n = len(data)
    labels = np.stack([data_b[:, i : 1 + n + i - horizon] for i in range(horizon)], axis=2)[:, 1:]
    labels = labels[:, drop:]
    return reps[:-horizon][drop:], labels.reshape(-1, horizon * data.shape[1])


def fit(
    train: tuple, val: tuple, test: tuple | None, *, scalers: tuple, horizon: int, alphas: tuple = WIDE23
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Select a centered float64 Cholesky ridge using validation RMSE plus MAE."""
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Ridge alphas must be finite and positive')
    for x, y in (train, val, *((test,) if test is not None else ())):
        if x.ndim != 2 or y.ndim != 2 or not x.size or len(x) != len(y):
            raise ValueError('Ridge requires nonempty aligned [N,D] features and [N,H*C] targets')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Ridge inputs must be finite')
    x, y = np.asarray(train[0], dtype=np.float64), np.asarray(train[1], dtype=np.float64)
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered_x, centered_y = x - mean_x, y - mean_y
    gram, rhs = centered_x.T @ centered_x, centered_x.T @ centered_y

    def solve(alpha: float) -> tuple[np.ndarray, np.ndarray]:
        system = gram.copy()
        system.flat[:: x.shape[1] + 1] += float(alpha)
        chol = np.linalg.cholesky(system)
        weights = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        return weights, mean_y - mean_x @ weights

    best_alpha, best_score = None, float('inf')
    for alpha in alphas:
        weights, intercept = solve(alpha)
        pred = np.asarray(val[0], dtype=np.float64) @ weights + intercept
        score = np.sqrt(((pred - val[1]) ** 2).mean()) + np.abs(pred - val[1]).mean()
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite ridge validation score')
        if score < best_score:
            best_alpha, best_score = float(alpha), score
    weights, intercept = solve(best_alpha)
    result = {f'horizon_{horizon}_best_alpha': best_alpha}
    channels = y.shape[1] // horizon
    pred_mat, true_mat = np.empty((0, horizon, channels)), np.empty((0, horizon, channels))
    for name, (features, targets) in (('train', train), ('val', val), *((('test', test),) if test is not None else ())):
        predictions = np.asarray(features, dtype=np.float64) @ weights + intercept
        delta = predictions - targets
        prefix = f'horizon_{horizon}_{name}_'
        result.update({prefix + 'mse_norm': float((delta**2).mean()), prefix + 'mae_norm': float(np.abs(delta).mean())})
        pred_mat = predictions.reshape(-1, horizon, channels).copy()
        true_mat = targets.reshape(-1, horizon, channels).copy()
        for channel, scaler in enumerate(scalers[:channels]):
            pred_mat[:, :, channel] = scaler.inverse_transform(pred_mat[:, :, channel])
            true_mat[:, :, channel] = scaler.inverse_transform(true_mat[:, :, channel])
        delta = pred_mat - true_mat
        result.update({prefix + 'mse': float((delta**2).mean()), prefix + 'mae': float(np.abs(delta).mean())})
    if test is None:
        pred_mat, true_mat = np.empty((0, horizon, channels)), np.empty((0, horizon, channels))
    return result, pred_mat, true_mat


def sliding_encode(encoder: torch.nn.Module, series: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Encode one causal 201-step window per origin with 200 leading NaN rows."""
    if series.ndim != 2 or not len(series) or batch_size < 1:
        raise ValueError('Sliding encode requires a nonempty [T,C] series and positive batch size')
    padding = np.full((200, series.shape[1]), np.nan, dtype=series.dtype)
    padded = np.concatenate((padding, series))
    device = next(encoder.parameters()).device
    outputs = []
    with frozen(encoder):
        for start in range(0, len(series), batch_size):
            windows = [
                torch.tensor(padded[i : i + 201], dtype=torch.float32).T
                for i in range(start, min(start + batch_size, len(series)))
            ]
            last = encoder(torch.stack(windows).to(device)).tokens[:, -1, :]
            # Keep only the final token's storage, including singleton CPU batches.
            outputs.append(last.to('cpu', copy=True, memory_format=torch.contiguous_format).numpy())
    return np.concatenate(outputs)


def evaluate(
    encoder: torch.nn.Module, splits: object, cfg: object, *, test: bool | None = None, alphas: tuple | None = None
) -> dict[str, float]:
    """Score Table 3 horizons with TRAIN-only ridge fits and validation-selected alpha."""
    use_test = cfg.forecast.test if test is None else test
    train_slice, val_slice, test_slice = splits.slices
    stop = test_slice.stop if use_test else val_slice.stop
    series = splits.series[:stop]
    reps = sliding_encode(encoder, series)
    labels = series[:, splits.n_cov :]
    scalers = splits.scalers
    mode = 'uni' if cfg.data.univariate else 'multi'
    if cfg.data.univariate:
        labels = labels[:, splits.target_idx : splits.target_idx + 1]
        scalers = (scalers[splits.target_idx],)
    results = {}
    for horizon in cfg.data.horizons:
        train = samples(reps[train_slice], labels[train_slice], horizon, drop=200)
        val = samples(reps[val_slice], labels[val_slice], horizon)
        test_data = samples(reps[test_slice], labels[test_slice], horizon) if use_test else None
        metrics, _, _ = fit(
            train,
            val,
            test_data,
            scalers=scalers,
            horizon=horizon,
            alphas=tuple(cfg.forecast.alphas) if alphas is None else alphas,
        )
        results.update(metrics)
    for split in ('train', 'val', *(('test',) if use_test else ())):
        for metric in ('mse', 'mae', 'mse_norm', 'mae_norm'):
            results[f'avg_{split}_{metric}'] = float(
                np.mean([results[f'horizon_{h}_{split}_{metric}'] for h in cfg.data.horizons])
            )
    return {f'{mode}/{key}': value for key, value in results.items()}


def select(history: dict[int, tuple[float, float]]) -> int:
    """Select the milestone by validation MAE, then MSE, then earlier epoch."""
    if not history or any(not np.isfinite(values).all() for values in history.values()):
        raise ValueError('Forecast checkpoint history must be nonempty and finite')
    return min(history, key=lambda epoch: (*history[epoch], epoch))


def validation_milestone(rows: list[dict], mode: str, run_dir: Path) -> int:
    """Require every Table 3 validation probe before selecting its retained checkpoint."""
    mae = f'forecast/{mode}/avg_val_mae_norm'
    mse = f'forecast/{mode}/avg_val_mse_norm'
    history = {int(row['epoch']): (float(row[mae]), float(row[mse])) for row in rows if mae in row and mse in row}
    epochs = sorted(history)
    if tuple(epochs) != MILESTONES:
        raise ValueError(f'{run_dir}: expected Table 3 validation milestones {list(MILESTONES)}, found {epochs}')
    return select(history)


def replay(run_dir: str | Path, splits: object, *, device: str = 'cpu') -> dict:
    """Select a retained validation milestone and score both publication and source grids."""
    from pmt.config import Config
    from pmt.model import build_encoder

    run_dir = Path(run_dir)
    rows = json.loads((run_dir / 'metrics.json').read_text())['history']
    modes = {mode for mode in ('uni', 'multi') if any(f'forecast/{mode}/avg_val_mae_norm' in row for row in rows)}
    if len(modes) != 1:
        raise ValueError('Replay requires exactly one forecast mode in the validation history')
    mode = modes.pop()
    epoch = validation_milestone(rows, mode, run_dir)
    payload = torch.load(run_dir / f'checkpoints/epoch{epoch}.pt', map_location='cpu', weights_only=False)
    if int(payload['epoch']) != epoch:
        raise ValueError('Retained checkpoint epoch disagrees with the selected milestone')
    cfg = Config(payload['cfg'])
    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(payload['state_dict'], strict=True)
    return dict(
        epoch=epoch,
        wide23=evaluate(encoder, splits, cfg, test=True, alphas=WIDE23),
        source13=evaluate(encoder, splits, cfg, test=True, alphas=SOURCE13),
    )
