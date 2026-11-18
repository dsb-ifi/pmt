"""Immutable configuration and compact-run resolver checks."""

import pytest
from pmt.config import Config, load, load_run, runs


def test_config_guards(tmp_path):
    path = tmp_path / 'base.yaml'
    path.write_text('train: {seed: 42}\nitems: [1, 2]\nprobe: {full_scan: null, static_batch: false, tasks: []}\n')
    cfg = load(str(path), ['train.seed=7'])
    assert cfg.train.seed == 7 and cfg['items'] == (1, 2)
    assert cfg.probe.full_scan is None and cfg.probe.static_batch is False and cfg.probe.tasks == ()
    with pytest.raises(TypeError):
        cfg.train.seed = 2
    with pytest.raises(KeyError):
        load(str(path), ['train.typo=1'])
    with pytest.raises(TypeError):
        load(str(path), ['train.seed=true'])
    assert Config(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_run_precision() -> None:
    """Retain null automatic precision and allow explicit dtype overrides."""
    for precision in (None, 'bf16'):
        run_id = next(key for key, row in runs().items() if row['precision'] == precision)
        assert load_run(run_id).train.precision == precision
        assert load_run(run_id, ['train.precision=fp16']).train.precision == 'fp16'
