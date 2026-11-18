"""Check published SVM selection and checksum-verified atomic downloads offline."""

import hashlib
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.download_data import fetch


def test_custom_training_and_encoder_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train from a two-line recipe and load both kinds of saved encoder."""
    import torch

    import pmt
    from pmt.config import ROOT, load
    from pmt.data import load_dataset
    from scripts import train

    data_root = tmp_path / 'data'
    dataset = data_root / 'toy'
    dataset.mkdir(parents=True)
    generator = torch.Generator().manual_seed(13)
    for split, count in (('train', 160), ('val', 4), ('test', 4)):
        torch.save(
            {'samples': torch.randn(count, 2, 80, generator=generator), 'labels': torch.arange(count) % 2},
            dataset / f'{split}.pt',
        )
    recipe = tmp_path / 'toy.yaml'
    recipe.write_text(f"extends: '{ROOT / 'configs/base.yaml'}'\ndata: {{name: toy}}\n")
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    run_dir = tmp_path / 'runs' / 'toy_s7'
    overrides = [
        f'data.root={data_root}',
        'train.device=cpu',
        'train.steps=1',
        'optim.batch_size=2',
        'model.dim=16',
        'pma.layers=1',
        'probe.batch_size=4',
    ]
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'train',
            '--config',
            str(recipe),
            '--seed',
            '7',
            *(arg for value in overrides for arg in ('--override', value)),
        ],
    )
    train.main()
    record = json.loads((run_dir / 'run.json').read_text())
    assert record['experiment_id'] == 'toy_s7' and record['exit_code'] == 0
    assert [record['cfg']['data'][key] for key in ('channels', 'length', 'classes')] == [2, 80, 2]
    assert json.loads((run_dir / 'metrics.json').read_text())['history'][0]['step'] == 1
    inputs = torch.randn(2, 2, 80, generator=generator)
    outputs = []
    for name in ('encoder.pt', 'best_encoder.pt'):
        encoder = pmt.load_encoder(run_dir / name)
        assert not encoder.training
        with torch.no_grad():
            features = encoder(inputs)
        assert features.cls.shape == (2, 16)
        assert features.tokens.shape == (2, 81, 16)
        assert features.mask.shape == (2, 82)
        assert features.pma_states.size(0) == features.pma_final.size(0) == 2
        outputs.append(features.cls)
    assert torch.equal(*outputs)
    payload = torch.load(run_dir / 'best_encoder.pt', weights_only=True)
    normalized = ((inputs - payload['mean'][:, None]) / payload['std'][:, None]).clamp(-10, 10)
    with torch.no_grad():
        assert torch.equal(encoder.encoder(normalized).cls, outputs[-1])
    with pytest.raises(ValueError, match='data.channels expected 1, observed 2'):
        load_dataset(load(str(recipe), [f'data.root={data_root}', 'data.channels=1']))


def test_scripts(tmp_path: Path) -> None:
    """Reject corrupted bytes before target replacement."""
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'new data')
    target.write_bytes(b'old data')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch(source.as_uri(), target, '0' * 64)
    assert target.read_bytes() == b'old data'
    fetch(source.as_uri(), target, hashlib.sha256(b'new data').hexdigest())
    assert target.read_bytes() == b'new data'


def test_evaluate_refits_saved_svm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refit both saved parameter pairs at an epoch absent from the published table."""
    import numpy as np
    import torch
    from pmt import data, model, probe
    from pmt.config import load_run, runs
    from scripts import evaluate

    encoder = torch.nn.Linear(1, 1)
    values = load_run(next(iter(runs()))).to_dict()
    values['probe'].update(tasks=['linear'], tune='test')
    expected = {0.01: {'C': 0.75, 'gamma': 0.125}, 0.05: {'C': 3.0, 'gamma': 0.25}}
    path, output = tmp_path / 'best_encoder.pt', tmp_path / 'score.json'
    torch.save(dict(state_dict=encoder.state_dict(), cfg=values, epoch=999, svm_params=expected), path)
    split = data.Split(torch.zeros(400, 1, 1), torch.arange(400) % 2)
    monkeypatch.setattr(data, 'load_dataset', lambda cfg: data.Splits(split, split, split))
    monkeypatch.setattr(model, 'SSLModel', lambda cfg: NS(encoder=encoder))
    monkeypatch.setattr(
        probe, 'collect', lambda *args, **kwargs: (np.array([[-2], [-1], [1], [2]]), np.array([0, 0, 1, 1]))
    )
    fit_svm, fitted = probe.fit_svm, []

    def observed(*args, **kwargs):
        assert kwargs['fixed']
        result = fit_svm(*args, **kwargs)
        fitted.append(result[1])
        return result

    monkeypatch.setattr(probe, 'fit_svm', observed)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', str(path), '--run-svm', '--out', str(output)])
    evaluate.main()
    assert fitted == list(expected.values())
    assert 'probe/svm/lp0.010/top1_acc' in json.loads(output.read_text())['metrics']


def _schemas(root: Path) -> tuple[Path, list[str]]:
    from scripts.summarize import ROOT, write_csv

    shipped = root / 'shipped'
    for name in ('runs.csv', 'table3_per_seed.csv'):
        with (ROOT / 'results' / name).open() as handle:
            fields = next(csv.reader(handle))
        write_csv(shipped / name, fields, [])
    with (shipped / 'runs.csv').open() as handle:
        return shipped, next(csv.reader(handle))


def test_classification_summary(tmp_path: Path) -> None:
    """Select accuracy then F1 and compute paired seed summaries in the shipped schema."""
    from scripts.summarize import collect, summarize, write_csv

    shipped, fields = _schemas(tmp_path)
    reference = []
    for seed, accuracy, f1 in ((42, 0.9, 0.7), (43, 0.7, 0.5)):
        folder = tmp_path / 'runs' / f'arm_s{seed}'
        folder.mkdir(parents=True)
        record = dict(
            experiment_id=folder.name,
            source='cohort',
            dataset='UCR_FordA',
            seed=seed,
            tables='1',
            exit_code=0,
            cfg={'data': {'name': 'UCR_FordA'}},
        )
        (folder / 'run.json').write_text(json.dumps(record))
        history = []
        for epoch, acc, mf1 in ((1, accuracy - 0.1, 0.99), (2, accuracy, f1), (3, accuracy, f1 - 0.1)):
            row = {'epoch': epoch}
            for budget in (0.01, 0.05):
                row.update({f'probe/svm/lp{budget:.3f}/top1_acc': acc, f'probe/svm/lp{budget:.3f}/mf1': mf1})
            history.append(row)
        (folder / 'metrics.json').write_text(json.dumps({'history': history}))
        other = {key: record.get(key, '') for key in fields}
        for budget in (0.01, 0.05):
            other[f'probe/svm/lp{budget:.3f}/global_best_top1_acc'] = accuracy - 0.1
            other[f'probe/svm/lp{budget:.3f}/global_best_mf1'] = f1 - 0.1
        reference.append(other)
    write_csv(shipped / 'runs.csv', fields, reference)
    rows, forecasts = collect(tmp_path / 'runs', shipped)
    assert not forecasts and rows[0]['probe/svm/lp0.010/global_best_mf1'] == 0.7
    assert list(rows[0]) == fields
    assert any('n=2 mean=0.8 sd=0.14142136 delta=+0.1' in line for line in summarize(rows, reference))
    path = tmp_path / 'runs/arm_s42/run.json'
    record = json.loads(path.read_text())
    record['exit_code'] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='incomplete'):
        collect(tmp_path / 'runs', shipped)


def test_forecast_summary(tmp_path: Path) -> None:
    """Require the correct validation-selected replay and preserve per-horizon grid rows."""
    from scripts.summarize import collect

    shipped, _ = _schemas(tmp_path)
    folder = tmp_path / 'run'
    folder.mkdir()
    record = dict(
        experiment_id='forecast_s43',
        source='table3',
        dataset='ETTh1',
        seed=43,
        tables='3',
        exit_code=0,
        cfg={'data': {'name': 'ETTh1', 'univariate': True, 'horizons': [24]}},
    )
    (folder / 'run.json').write_text(json.dumps(record))
    history = [
        dict(
            epoch=epoch,
            step=epoch * 100,
            **{'forecast/uni/avg_val_mae_norm': mae, 'forecast/uni/avg_val_mse_norm': mse},
        )
        for epoch, mae, mse in (
            (2, 0.3, 0.4),
            (6, 0.2, 0.35),
            (10, 0.2, 0.35),
            (20, 0.3, 0.4),
            (30, 0.3, 0.4),
            (40, 0.3, 0.4),
            (50, 0.3, 0.4),
        )
    ]
    (folder / 'metrics.json').write_text(json.dumps({'history': history}))
    with pytest.raises(ValueError, match='replay-run'):
        collect(folder, shipped)
    packet = {
        f'uni/{h}_test_{metric}_norm': value
        for h in ('horizon_24', 'avg')
        for metric, value in (('mse', 0.1), ('mae', 0.2))
    }
    replay = dict(epoch=6, wide23=packet, source13=packet)
    path = folder / 'replay.json'
    path.write_text(json.dumps({'metrics': replay}))
    (folder / 'metrics.json').write_text(json.dumps({'history': history[:-1]}))
    with pytest.raises(ValueError, match=r'expected Table 3 validation milestones.*found \[2, 6, 10, 20, 30, 40\]'):
        collect(folder, shipped)
    (folder / 'metrics.json').write_text(json.dumps({'history': history}))
    rows, forecasts = collect(folder, shipped)
    assert len(rows) == 1 and len(forecasts) == 4
    assert all(
        r['selected_epoch'] == 6 and r['selected_step'] == 600 and r['selection_split'] == 'val' for r in forecasts
    )
    assert {r['grid'] for r in forecasts} == {'wide23', 'source13'}
    replay['epoch'] = 10
    path.write_text(json.dumps({'metrics': replay}))
    with pytest.raises(ValueError, match='validation-selected'):
        collect(folder, shipped)


@pytest.mark.parametrize('failed', [False, True])
def test_training_completion_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    """Mark a run complete only after the training call returns successfully."""
    from scripts import train as command

    folder = tmp_path / 'run'
    cfg = NS(train=NS(seed=42), data=NS(name='UCR_FordA'), to_dict=lambda: {'data': {'name': 'UCR_FordA'}})
    row = dict(seed=42, source='cohort', arm='arm', tables='1')
    monkeypatch.setitem(sys.modules, 'pmt.config', NS(load_run=lambda *a: cfg, runs=lambda: {'arm_s42': row}))

    def train(actual, run_dir):
        assert actual is cfg and run_dir == folder
        assert json.loads((folder / 'run.json').read_text())['exit_code'] is None
        if failed:
            raise RuntimeError('synthetic failure')

    monkeypatch.setitem(sys.modules, 'pmt.train', NS(train=train))
    monkeypatch.setattr(sys, 'argv', ['train', '--id', 'arm_s42', '--run-dir', str(folder)])
    if failed:
        with pytest.raises(RuntimeError, match='synthetic failure'):
            command.main()
    else:
        command.main()
    record = json.loads((folder / 'run.json').read_text())
    assert record['experiment_id'] == 'arm_s42' and record['seed'] == 42
    assert record['exit_code'] == (None if failed else 0)
