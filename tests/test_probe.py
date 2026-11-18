"""Check SVM fitting and frozen readout invariants."""

from statistics import mean, stdev
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest
import torch

from pmt.probe import Probe, collect, fit_svm, frozen, readout



def test_invalid_and_readout() -> None:
    """Reject an impossible CV split and restore mixed module modes after errors."""
    with pytest.raises(ValueError, match='two rows'):
        fit_svm((np.ones((2, 2)), np.array([0, 1])), (np.ones((2, 2)), np.array([0, 1])), seed=44)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout()).train()
    model[0].eval()
    with frozen(model):
        assert not any(m.training for m in model.modules())
    assert model.training and not model[0].training
    out = NS(tokens=torch.tensor([[[2.0], [7.0]]]), mask=torch.tensor([[True, True, False]]))
    assert readout(out, 'max').item() == 2


def test_cls_collection_retains_compact_readouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copy selected CLS rows including a singleton tail without changing values or labels."""
    monkeypatch.setitem(sys.modules, 'pmt.data', NS(transform=lambda x, *args: x))
    encoder = torch.nn.Linear(4, 4)
    monkeypatch.setattr(encoder, 'forward', lambda batch: NS(cls=batch[:, 0]))
    split = NS(x=torch.arange(60.0).reshape(5, 3, 4), y=torch.arange(5))
    cfg = NS(probe=NS(readout='cls', batch_size=2, input_view='clean_base', full_scan=False, static_batch=False))
    original_cat = torch.cat
    batch_sizes = []

    def cat(tensors: list[torch.Tensor]) -> torch.Tensor:
        if tensors[0].ndim == 2:
            for tensor in tensors:
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                batch_sizes.append(len(tensor))
        return original_cat(tensors)

    monkeypatch.setattr(torch, 'cat', cat)
    with frozen(encoder):
        x, y = collect(encoder, split, NS(mean=None, std=None), cfg, torch.arange(5), training=False)
    np.testing.assert_array_equal(x, split.x[:, 0])
    np.testing.assert_array_equal(y, split.y)
    assert batch_sizes == [2, 2, 1]


def test_probe_rng_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect every budget and restore observational iterator and augmentation RNG draws."""
    counts = {True: 0, False: 0}

    def transform(x, *args, training=False, **kwargs):
        counts[training] += 1
        return x.clone()

    monkeypatch.setitem(
        sys.modules, 'pmt.data', NS(transform=transform, label_subset=lambda y, p, s: torch.arange(int(len(y) * p)))
    )
    x = torch.randn(1000, 1, 3, generator=torch.Generator().manual_seed(44))
    split = NS(x=x, y=torch.arange(1000) % 2)
    test = NS(x=x[:20], y=split.y[:20])
    splits = NS(train=split, val=test, test=test, mean=None, std=None, series=None)
    cfg = NS(
        train=NS(seed=44),
        probe=NS(
            tune='test',
            readout='cls',
            batch_size=16,
            reset_every=0,
            early_reset=0,
            input_view='semantics',
            full_scan=False,
            static_batch=False,
            tasks=[],
        ),
    )
    encoder = torch.nn.Linear(3, 3).train()
    monkeypatch.setattr(encoder, 'forward', lambda batch: NS(cls=batch[:, 0]))
    probe = Probe(cfg)
    state = torch.get_rng_state().clone()
    first = probe.evaluate(encoder, splits, 5)
    assert torch.equal(state, torch.get_rng_state())
    assert counts == {True: 60, False: 40} and encoder.training
    probe.evaluate(encoder, splits, 10)
    assert counts == {True: 120, False: 80}
    assert np.isfinite(first['probe/svm/lp0.010/mf1'])
    draws = []

    def cue(encoder, splits, cfg, *, generators):
        draws.append(float(torch.rand((), generator=generators[0])))
        return {'by_difficulty': {'example': {'val_metrics': {'top1_acc': 0.5}, 'retention_metrics': {'auc': 0.8}}}}

    monkeypatch.setitem(sys.modules, 'pmt.cue', NS(evaluate=cue))
    cfg.probe.tasks = ['linear', 'cue']
    assert probe.evaluate(encoder, splits, 15)['cue/eosexample/auc'] == 0.8
    probe.evaluate(encoder, splits, 20)
    assert draws[0] != draws[1]


@pytest.mark.parametrize('clean', [False, True])
def test_full_scan_and_static_batches(monkeypatch: pytest.MonkeyPatch, clean: bool) -> None:
    """Consume unused full-scan views, then mask labelled outputs after padded forwarding."""
    views, shapes = [], []

    def transform(x, *args, view='base', **kwargs):
        views.append(view)
        return x

    monkeypatch.setitem(sys.modules, 'pmt.data', NS(transform=transform))
    split = NS(x=torch.arange(15.0).reshape(5, 1, 3), y=torch.arange(5))
    cfg = NS(
        probe=NS(
            input_view='clean_base' if clean else 'semantics',
            full_scan=True,
            static_batch=True,
            batch_size=3,
            readout='cls',
        )
    )
    encoder = torch.nn.Linear(3, 3)

    def forward(batch):
        shapes.append(tuple(batch.shape))
        return NS(cls=batch[:, 0])

    monkeypatch.setattr(encoder, 'forward', forward)
    indices = torch.tensor([0, 4])
    with frozen(encoder):
        x, y = collect(encoder, split, NS(mean=None, std=None), cfg, indices, training=True)
    np.testing.assert_array_equal(x, split.x[indices, 0])
    np.testing.assert_array_equal(y, [0, 4])
    assert shapes == [(3, 1, 3), (3, 1, 3)]
    assert views == (['base'] * 5 if clean else ['base', 'weak', 'strong'] * 5)


@pytest.mark.parametrize('fixed', [False, True])
def test_successive_draws_match_uncached_single_probes(monkeypatch: pytest.MonkeyPatch, fixed: bool) -> None:
    """Match each independent draw and its noise against the full single-probe path."""
    from pmt import probe as module

    train_packets, clean_packets = [], []

    def record(*args, training):
        features, labels = collect(*args, training=training)
        (train_packets if training else clean_packets).append(features.copy())
        return features, labels

    def transform(x, *args, training=False, **kwargs):
        return x + torch.randn_like(x) if training else x

    monkeypatch.setattr(module, 'collect', record)
    monkeypatch.setitem(
        sys.modules, 'pmt.data', NS(transform=transform, label_subset=lambda y, p, s: torch.arange(int(len(y) * p)))
    )
    x = torch.randn(240, 1, 3, generator=torch.Generator().manual_seed(8))
    y = torch.arange(len(x)) % 2
    splits = NS(
        train=NS(x=x, y=y),
        val=NS(x=x[:20], y=y[:20]),
        test=NS(x=x[20:50], y=y[20:50]),
        mean=None,
        std=None,
        series=None,
    )
    cfg = NS(
        train=NS(seed=8),
        probe=NS(
            tune='val',
            readout='cls',
            batch_size=8,
            reset_every=0,
            early_reset=0,
            input_view='semantics',
            full_scan=False,
            static_batch=True,
            tasks=[],
        ),
    )
    encoder = torch.nn.Linear(3, 3).train()
    monkeypatch.setattr(encoder, 'forward', lambda batch: NS(cls=batch[:, 0]))
    state = torch.Generator().manual_seed(8).get_state()

    def new_probe():
        evaluator = Probe(cfg)
        if fixed:
            evaluator.best_params = {budget: dict(C=2.0, gamma=0.125) for budget in module.BUDGETS}
        return evaluator

    torch.set_rng_state(state)
    default = new_probe().evaluate(encoder, splits, 1, fixed=fixed)
    assert torch.equal(torch.get_rng_state(), state)
    train_packets.clear()
    clean_packets.clear()

    first, summary = new_probe().evaluate_draws(encoder, splits, 3, fixed=fixed)
    assert first == default and summary['count'] == 3
    assert torch.equal(torch.get_rng_state(), state) and encoder.training
    assert len(clean_packets) == 2
    cached_train = train_packets.copy()
    train_packets.clear()
    assert not np.array_equal(cached_train[0], cached_train[2])

    monkeypatch.setattr(module, 'observational', frozen)
    expected = [new_probe().evaluate(encoder, splits, 1, fixed=fixed) for _ in range(3)]
    assert len(train_packets) == len(cached_train) == 6
    for actual, native in zip(cached_train, train_packets):
        np.testing.assert_array_equal(actual, native)
    for budget, metrics in summary['by_budget'].items():
        samples = [row[f'probe/svm/{budget}/mf1'] for row in expected]
        assert metrics['mf1'] == dict(
            first=samples[0], mean=mean(samples), sd=stdev(samples), min=min(samples), max=max(samples), values=samples
        )


def test_small_budget_keeps_every_class() -> None:
    """A 1% budget below the class count still labels one example per class."""
    from pmt.data import label_subset

    labels = torch.arange(60) % 3
    assert torch.unique(labels[label_subset(labels, 0.01, 42)]).tolist() == [0, 1, 2]
