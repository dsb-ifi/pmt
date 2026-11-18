"""Frozen-encoder classification with the reference RBF SVM probe."""

from __future__ import annotations

from contextlib import contextmanager
from math import isnan
import random
from statistics import mean, stdev
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BUDGETS = (0.01, 0.05)
TEST_METRICS = ('top1_acc', 'mf1', 'macro_acc')
WIDE_C = [2**e for e in (-7, -5, -3, -1, 1, 3, 5)]
WIDE_GAMMA = [2**e for e in (-17, -13, -9, -7, -5, -3, -1)]


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return the reference single-label classification metrics."""
    accuracy = float(accuracy_score(labels, predictions))
    return dict(
        top1_acc=accuracy,
        accuracy=accuracy,
        macro_acc=float(balanced_accuracy_score(labels, predictions)),
        mf1=float(f1_score(labels, predictions, average='macro', zero_division=0)),
        loss=float('nan'),
    )


@contextmanager
def frozen(encoder: torch.nn.Module) -> Iterator[None]:
    """Disable gradients and restore every submodule's original training flag."""
    modes = [(module, module.training) for module in encoder.modules()]
    try:
        encoder.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


@contextmanager
def observational(encoder: torch.nn.Module) -> Iterator[None]:
    """Restore the training random streams after an observational SVM/forecast probe."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        with frozen(encoder):
            yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _matrix(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x), np.asarray(y).reshape(-1)
    if x.ndim != 2 or not x.size or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('Probe requires nonempty finite [N,D] features and N labels')
    return x, y


def fit_svm(
    train: tuple,
    test: tuple,
    *,
    seed: int,
    previous: dict | None = None,
    tune: tuple | None = None,
    fixed: bool = False,
) -> tuple[object, dict, dict, dict]:
    """Fit the reference SVM-only linear probe, carrying its wide/narrow grid."""
    x, y = _matrix(*train)
    tx, ty = _matrix(*test)
    if fixed and previous is None:
        raise ValueError('A fixed SVM requires C and gamma')
    grid = dict(C=WIDE_C, gamma=WIDE_GAMMA)
    if previous is not None:
        mult = (1.0,) if fixed else (0.5, 0.75, 1.0, 1.5, 2.0)
        grid = {
            key: sorted({max(low, min(high, float(previous[key]) * m)) for m in mult})
            for key, low, high in (('C', 2**-19, 2**7), ('gamma', 2**-21, 2**5))
        }

    def pipeline(**params: float) -> object:
        return make_pipeline(
            StandardScaler(), SVC(kernel='rbf', class_weight='balanced', random_state=seed, max_iter=20000, **params)
        )

    if tune is not None:
        vx, vy = _matrix(*tune)
        best, params = -float('inf'), None
        for c in grid['C']:
            for gamma in grid['gamma']:
                candidate = pipeline(C=float(c), gamma=float(gamma)).fit(x, y)
                score = accuracy_score(vy, candidate.predict(vx))
                if score > best:
                    best, params = score, dict(C=float(c), gamma=float(gamma))
        model = pipeline(**params).fit(x, y)
    else:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            raise ValueError('SVM cross-validation requires at least two rows per class and two classes')
        cv = StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=seed)
        search = GridSearchCV(pipeline(), {f'svc__{key}': value for key, value in grid.items()}, cv=cv, n_jobs=1)
        search.fit(x, y)
        model = search.best_estimator_
        params = {key.removeprefix('svc__'): float(value) for key, value in search.best_params_.items()}
    return model, params, metrics(y, model.predict(x)), metrics(ty, model.predict(tx))


def readout(features: object, mode: str) -> torch.Tensor:
    """Extract CLS or valid-token maximum features from the encoder contract."""
    if mode == 'cls':
        result = features.cls.float()
    elif mode == 'max':
        tokens = features.tokens.float()
        mask = features.mask
        if mask is not None:
            mask = mask[:, 1:] if mask.shape[1] == tokens.shape[1] + 1 else mask
            if mask.dtype != torch.bool or mask.shape != tokens.shape[:2] or not mask.any(1).all():
                raise ValueError('Invalid token mask for classification readout')
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), -float('inf'))
        result = tokens.max(1).values
    else:
        raise ValueError(f'Unknown classification readout {mode!r}')
    if result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError('Encoder produced invalid classification features')
    return result


def collect(
    encoder: torch.nn.Module, split: object, splits: object, cfg: object, indices: torch.Tensor, *, training: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Collect the native compact/full-scan views, including iterator RNG and static padding."""
    from pmt.data import transform

    p = cfg.probe
    if p.input_view not in {'semantics', 'clean_base'}:
        raise ValueError(f'Unknown probe input view {p.input_view!r}')
    full = training and bool(p.full_scan)
    supervised = torch.zeros(len(split.x), dtype=torch.bool)
    supervised[indices] = True

    class Rows(Dataset):
        def __len__(self) -> int:
            return len(split.x)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            args = split.x[index], cfg, splits.mean, splits.std
            if not training or p.input_view == 'clean_base':
                x = transform(*args)
            else:
                if full:
                    transform(*args, view='base', training=True)
                x = transform(*args, view='weak', training=True)
                if full:
                    transform(*args, view='strong', training=True)
            return x, index

    sampler = range(len(split.x)) if full else indices.tolist()
    loader = DataLoader(Rows(), batch_size=p.batch_size, sampler=sampler, num_workers=0)
    device = next(encoder.parameters()).device
    values, labels = [], []
    for x, rows in loader:
        mask = supervised[rows]
        if not mask.any():
            continue
        if p.static_batch:
            padding = p.batch_size - len(x)
            if padding:
                x = torch.cat((x, x[:1].expand(padding, *x.shape[1:])))
                mask = torch.cat((mask, torch.zeros(padding, dtype=torch.bool)))
            features = readout(encoder(x.to(device)), p.readout).cpu()[mask]
        else:
            features = readout(encoder(x[mask].to(device)), p.readout)
            # A CLS view can hold the encoder's entire token batch on CPU.
            features = features.to('cpu', copy=p.readout == 'cls', memory_format=torch.contiguous_format)
        values.append(features)
        labels.append(split.y[rows][supervised[rows]])
    if not values:
        raise ValueError('Probe collection found no labelled rows')
    return torch.cat(values).numpy(), torch.cat(labels).numpy()


class Probe:
    """Keep per-budget SVM parameters across frozen-encoder evaluation calls."""

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg
        self.best_params: dict[float, dict] = {}
        self.best_scores: dict[float, tuple[float, float]] = {}
        self.calls = 0
        self.first_epoch: int | None = None
        self.cue_generators: tuple[torch.Generator, torch.Generator] | None = None

    def evaluate_features(
        self,
        train: tuple,
        test: tuple,
        budget: float,
        *,
        tune: tuple | None = None,
        reset: bool = False,
        fixed: bool = False,
    ) -> dict[str, float]:
        """Score already collected features without touching encoder or random streams."""
        previous = None if reset else self.best_params.get(budget)
        _, params, _, result = fit_svm(train, test, seed=self.cfg.train.seed, previous=previous, tune=tune, fixed=fixed)
        self.best_params[budget] = params
        return result

    def _classification(
        self, encoder: torch.nn.Module, splits: object, *, reset: bool, fixed: bool, clean: dict | None = None
    ) -> dict[str, float]:
        from pmt.data import label_subset

        cfg, p = self.cfg, self.cfg.probe

        def clean_features(name: str) -> tuple[np.ndarray, np.ndarray]:
            if clean is not None and name in clean:
                # A zero-worker DataLoader still draws its iterator base seed.
                iter(DataLoader((), num_workers=0))
                return clean[name]
            split = getattr(splits, name)
            packet = collect(encoder, split, splits, cfg, torch.arange(len(split.x)), training=False)
            if clean is not None:
                clean[name] = packet
            return packet

        results = {}
        for budget in BUDGETS:
            indices = label_subset(splits.train.y, budget, cfg.train.seed)
            train = collect(encoder, splits.train, splits, cfg, indices, training=True)
            test = clean_features('test')
            if p.tune == 'test':
                tune = test
            elif p.tune == 'val':
                tune = clean_features('val')
            elif p.tune == 'none':
                tune = None
            else:
                raise ValueError(f'Unknown SVM tuning split {p.tune!r}')
            result = self.evaluate_features(train, test, budget, tune=tune, reset=reset and not fixed, fixed=fixed)
            prefix = f'probe/svm/lp{budget:.3f}/'
            results.update({prefix + key: result[key] for key in TEST_METRICS})
            score = result['top1_acc'], result['mf1']
            self.best_scores[budget] = max(self.best_scores.get(budget, (-1.0, -1.0)), score)
            results.update(
                {
                    prefix + 'global_best_' + key: value
                    for key, value in zip(('top1_acc', 'mf1'), self.best_scores[budget])
                }
            )
        return results

    def _classification_tasks(self) -> set[str]:
        p = self.cfg.probe
        tasks = set(p.tasks or ['linear'])
        if tasks - {'linear', 'svm', 'cue', 'staged_ft'} or not tasks.intersection({'linear', 'svm'}):
            raise ValueError(f'Unsupported classification probe tasks {sorted(tasks)}')
        # The release omits the unreported staged_ft probe; it restored the encoder weights.
        return tasks

    def _cue_metrics(self, encoder: torch.nn.Module, splits: object, tasks: set[str]) -> dict[str, float | None]:
        if 'cue' not in tasks:
            return {}
        from pmt.cue import evaluate

        cfg = self.cfg
        if self.cue_generators is None:
            self.cue_generators = tuple(torch.Generator().manual_seed(cfg.train.seed) for _ in range(2))
        cue = evaluate(encoder, splits, cfg, generators=self.cue_generators)
        return {
            f'cue/eos{tag}/{key}': None if isnan(value) else float(value)
            for tag, packet in cue['by_difficulty'].items()
            for key, value in {**packet['val_metrics'], **packet['retention_metrics']}.items()
        }

    def evaluate(
        self, encoder: torch.nn.Module, splits: object, epoch: int, *, fixed: bool = False
    ) -> dict[str, float | None]:
        """Probe each label budget with the noisy labelled TRAIN and clean TEST views."""
        if splits.series is not None:
            from pmt.forecast import evaluate

            with observational(encoder):
                return {f'forecast/{key}': value for key, value in evaluate(encoder, splits, self.cfg).items()}
        p = self.cfg.probe
        tasks = self._classification_tasks()
        reset = self.calls == 0 or (p.reset_every > 0 and (self.calls + 1) % p.reset_every == 0)
        reset |= self.calls == 1 and self.first_epoch < p.early_reset
        with observational(encoder):
            results = self._classification(encoder, splits, reset=reset, fixed=fixed)
        self.first_epoch = epoch if self.first_epoch is None else self.first_epoch
        self.calls += 1
        return {**results, **self._cue_metrics(encoder, splits, tasks)}

    def evaluate_draws(
        self, encoder: torch.nn.Module, splits: object, draws: int, *, fixed: bool = False
    ) -> tuple[dict[str, float | None], dict]:
        """Fit independent SVMs from successive noise draws of the same labelled rows."""
        if draws < 1 or splits.series is not None:
            raise ValueError('Repeated SVM evaluation requires a positive draw count and classification data')
        tasks = self._classification_tasks()
        rows, clean = [], {}
        with observational(encoder):
            for _ in range(draws):
                probe = Probe(self.cfg)
                probe.best_params.update(self.best_params)
                rows.append(probe._classification(encoder, splits, reset=True, fixed=fixed, clean=clean))

        summaries = {}
        for budget in BUDGETS:
            label = f'lp{budget:.3f}'
            summaries[label] = {}
            for metric in TEST_METRICS:
                values = [row[f'probe/svm/{label}/{metric}'] for row in rows]
                summaries[label][metric] = dict(
                    first=values[0],
                    mean=mean(values),
                    sd=stdev(values) if draws > 1 else 0.0,
                    min=min(values),
                    max=max(values),
                    values=values,
                )
        first = {**rows[0], **Probe(self.cfg)._cue_metrics(encoder, splits, tasks)}
        return first, dict(count=draws, by_budget=summaries)
