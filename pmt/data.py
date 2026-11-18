"""Paper datasets, train-fitted normalization, label budgets and SSL views."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pmt.config import Config


@dataclass(frozen=True)
class Split:
    """Raw channel-first samples and classification or multi-horizon labels."""

    x: Tensor
    y: Tensor | tuple[Tensor, ...]


@dataclass(frozen=True)
class Splits:
    """Three source splits, normalization and optional forecasting series metadata."""

    train: Split
    val: Split
    test: Split
    mean: Tensor | None = None
    std: Tensor | None = None
    series: np.ndarray | None = None
    slices: tuple[slice, ...] = ()
    scalers: tuple = ()
    n_cov: int = 0
    target_idx: int = 0
    ssl_train: Split | None = None
    shape: tuple[int, int, int] | None = None


def _shape(cfg: Config, parts: list[Split], *, forecast: bool = False) -> tuple[int, int, int]:
    expected = (cfg.data.channels, cfg.data.length, cfg.data.classes)
    dimensions = parts[0].x.shape[1:]
    for part in parts:
        if part.x.ndim != 3 or not part.x.numel() or part.x.shape[1:] != dimensions:
            raise ValueError(f'{cfg.data.name}: all sample splits require consistent, nonempty [N,C,T] tensors')
        if not forecast and (part.y.ndim != 1 or len(part.y) != len(part.x)):
            raise ValueError(f'{cfg.data.name}: each split requires one label per sample, with shape [N]')
    classes = 1 if forecast else torch.unique(parts[0].y).numel()
    observed = (*dimensions, classes)
    for key, given, actual in zip(('channels', 'length', 'classes'), expected, observed):
        if given is not None and (type(given) is not int or given != actual):
            raise ValueError(f'{cfg.data.name}: data.{key} expected {given}, observed {actual}')
    return observed


def infer_data_config(cfg: Config, splits: Splits) -> Config:
    """Fill unset data dimensions from validated splits before constructing the model."""
    if not isinstance(splits, Splits) or splits.shape is None:
        return cfg
    values = dict(zip(('channels', 'length', 'classes'), splits.shape))
    if all(getattr(cfg.data, key) == value for key, value in values.items()):
        return cfg
    return Config({**cfg, 'data': {**cfg.data, **values}})


def label_subset(labels: Tensor, proportion: float, seed: int) -> Tensor:
    """Return canonical-order indices from the reference's seeded class quotas."""
    if not 0 < proportion <= 1 or len(labels) == 0:
        raise ValueError('A label budget needs nonempty labels and 0 < proportion <= 1')
    n = len(labels)
    if proportion >= 0.9999:
        return torch.arange(n)
    k = max(1, round(proportion * n))
    generator = torch.Generator().manual_seed(seed + 99999 + round(proportion * 1_000_000) % (2**31 - 1))
    mask = torch.zeros(n, dtype=torch.bool)
    if labels.ndim != 1:
        mask[torch.randperm(n, generator=generator)[:k]] = True
    else:
        groups = [(labels == label).nonzero().flatten() for label in torch.unique(labels)]
        k = max(k, len(groups))  # At least one label per class; the paper's budgets already exceed this.
        targets = [min(k // len(groups), len(group)) for group in groups]
        remain, index = k - sum(targets), 0
        while remain:
            j = index % len(groups)
            if targets[j] < len(groups[j]):
                targets[j] += 1
                remain -= 1
            index += 1
        for group, target in zip(groups, targets):
            mask[group[torch.randperm(len(group), generator=generator)[:target]]] = True
    return mask.nonzero().flatten()


def _fit_scaler(x: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    sums = squares = counts = None
    # Matching the reference's shuffled batch reduction also matches float32 rounding.
    for batch in DataLoader(x, batch_size=batch_size, shuffle=True):
        valid = ~torch.isnan(batch)
        total = torch.where(valid, batch, 0.0).sum((0, 2))
        square = torch.where(valid, batch**2, 0.0).sum((0, 2))
        count = valid.sum((0, 2))
        if sums is None:
            sums, squares, counts = total, square, count
        else:
            sums += total
            squares += square
            counts += count
    if torch.any(counts == 0):
        raise ValueError('A training channel has no observed samples')
    mean = sums / counts
    std = (squares / counts - mean**2).clamp_min(1e-8).sqrt().clamp_min(0.02)
    return mean, std


def _forecast(cfg: Config, root: Path) -> Splits:
    name = cfg.data.name
    if type(cfg.data.length) is not int or cfg.data.length < 1:
        raise ValueError(f'{name}: forecast windows require a positive integer data.length')
    path = root / ('ETT-small' if name == 'ETTh1' else name) / f'{name}.csv'
    with path.open() as handle:
        columns = handle.readline().strip().split(',')[1:]
        dates = [datetime.fromisoformat(line.split(',', 1)[0]) for line in handle]
    values = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(1, len(columns) + 1), dtype=np.float32)
    target_idx = columns.index(cfg.data.target)
    if cfg.data.univariate:
        values = values[:, target_idx : target_idx + 1].copy()
        target_idx = 0
    lengths = (
        (8640, 2880, 2880)
        if name == 'ETTh1' and len(values) >= 14400
        else (
            int(len(values) * 0.6),
            int(len(values) * 0.2),
            len(values) - int(len(values) * 0.6) - int(len(values) * 0.2),
        )
    )
    offsets = np.cumsum((0, *lengths))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))
    cov = np.asarray(
        [[d.minute, d.hour, d.weekday(), d.day, d.timetuple().tm_yday, d.month, d.isocalendar().week] for d in dates],
        dtype=np.float32,
    )
    cov_scaler = StandardScaler().fit(cov[slices[0]])
    cov = cov_scaler.transform(cov)
    scalers = []
    for col in range(values.shape[1]):
        scaler = StandardScaler().fit(values[slices[0], col : col + 1])
        for selection in slices:
            values[selection, col : col + 1] = scaler.transform(values[selection, col : col + 1])
        scalers.append(scaler)
    series = np.concatenate((cov, values), axis=1)
    result = []
    for i, selection in enumerate(slices):
        data = torch.from_numpy(series[selection])
        stride = cfg.data.stride if i == 0 else cfg.data.eval_stride
        count = len(data) - cfg.data.length - max(cfg.data.horizons) + 1
        if count <= 0:
            raise ValueError('Forecast split is shorter than input plus maximum horizon')
        # Views avoid materializing overlapping electricity windows many times.
        x = data.unfold(0, cfg.data.length, stride)[: len(range(0, count, stride))]
        labels = []
        for horizon in sorted(cfg.data.horizons):
            y = data[cfg.data.length :, 7:].unfold(0, horizon, stride)[: len(x)].transpose(1, 2)
            labels.append(y.squeeze(-1) if cfg.data.univariate else y)
        result.append(Split(x, tuple(labels)))
    return Splits(
        *result,
        series=series,
        slices=slices,
        scalers=tuple(scalers),
        n_cov=7,
        target_idx=target_idx,
        shape=_shape(cfg, result, forecast=True),
    )


def load_dataset(cfg: Config) -> Splits:
    """Load the source splits and fit normalization on TRAIN.

    For custom classification data, put dictionaries with 'samples' [N,C,T]
    and 'labels' [N] at data/<name>/{train,val,test}.pt and set data.name in a
    recipe extending configs/base.yaml.
    """
    root = Path(cfg.data.root)
    name = cfg.data.name
    if name in {'ETTh1', 'electricity'}:
        splits = _forecast(cfg, root)
        # Replay the scaler scan's loader and sampler seeds (reference pmt/utils/augmentation_utils.py:177).
        torch.empty(2, dtype=torch.int64, device='cpu').random_()
        return splits
    if name.startswith('UCR_'):
        dataset = name.removeprefix('UCR_')
        path = root / 'UCR' / dataset
        arrays = [np.loadtxt(path / f'{dataset}_{split}.tsv', delimiter='\t') for split in ('TRAIN', 'TEST')]
        vocabulary = {label: i for i, label in enumerate(np.unique(arrays[0][:, 0]))}
        parts = [
            Split(
                torch.from_numpy(a[:, 1:].astype(np.float32)).unsqueeze(1),
                torch.tensor([vocabulary[y] for y in a[:, 0]], dtype=torch.long),
            )
            for a in arrays
        ]
        train, val, test = parts[0], parts[1], parts[1]
    else:
        parts = [
            torch.load(root / name / f'{split}.pt', map_location='cpu', weights_only=False)
            for split in ('train', 'val', 'test')
        ]
        train, val, test = [Split(torch.as_tensor(p['samples']).float(), torch.as_tensor(p['labels'])) for p in parts]
    shape = _shape(cfg, [train, val, test])
    mean, std = _fit_scaler(train.x, cfg.optim.batch_size)
    ssl_train = Split(torch.cat((train.x, val.x)), torch.cat((train.y, val.y))) if cfg.data.merge_val else train
    return Splits(train, val, test, mean, std, ssl_train=ssl_train, shape=shape)


def transform(
    x: Tensor,
    cfg: Config,
    mean: Tensor | None,
    std: Tensor | None,
    *,
    view: str = 'base',
    training: bool = False,
    ssl: bool = False,
) -> Tensor:
    """Apply one reference view to a single row, preserving augmentation draw order."""
    if view not in {'base', 'weak', 'strong'}:
        raise ValueError(f'Unknown view: {view}')
    x = x.clone()
    if mean is not None:
        x = ((x - mean.to(x.device)[:, None]) / std.to(x.device)[:, None]).clamp(-10, 10)
    noise = cfg.data.ssl_noise_std if ssl and cfg.data.ssl_noise_std is not None else cfg.data.noise_std
    if cfg.data.noise and (view != 'base' or training):
        x = x + torch.normal(0, noise, x.shape, device=x.device)
    if view == 'strong':
        if cfg.data.time_warp:
            c, length = x.shape
            indices = torch.arange(length, dtype=torch.float32, device=x.device)
            warped = (indices + torch.normal(0, cfg.data.time_scale, (length,), device=x.device)).clamp(0, length - 1)
            warped = warped / (length - 1) * 2 - 1
            grid = torch.stack((warped, torch.zeros_like(warped)), -1).view(1, 1, length, 2)
            x = F.grid_sample(
                x.view(1, c, 1, length), grid, mode='bilinear', padding_mode='border', align_corners=True
            ).view_as(x)
        if cfg.data.magnitude_warp:
            scale = cfg.data.magnitude_scale
            walk = torch.cumsum(torch.normal(0, scale / 10, (x.shape[-1],), device=x.device), 0)
            # CubicSpline is evaluated at its original knots in the reference.
            smooth = walk.float()
            multiplier = ((smooth - smooth.mean()) * scale + 1.0).clamp(0.5, 1.5)
            x = x * multiplier
    return x


class _Views(Dataset):
    def __init__(self, x: Tensor, cfg: Config, mean: Tensor | None, std: Tensor | None) -> None:
        self.x, self.cfg, self.mean, self.std = x, cfg, mean, std

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        kwargs = dict(cfg=self.cfg, mean=self.mean, std=self.std, training=True, ssl=True)
        # Base-view noise is consumed even though SSL does not use this view.
        transform(self.x[index], view='base', **kwargs)
        return transform(self.x[index], view='weak', **kwargs), transform(self.x[index], view='strong', **kwargs)


def ssl_loader(splits: Splits, cfg: Config) -> DataLoader:
    """Yield weak and strong SSL views with a shuffled, non-dropping sampler."""
    source = splits.ssl_train if splits.ssl_train is not None else splits.train
    return DataLoader(_Views(source.x, cfg, splits.mean, splits.std), batch_size=cfg.optim.batch_size, shuffle=True)
