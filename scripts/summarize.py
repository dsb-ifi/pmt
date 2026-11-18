"""Summarize completed run histories and compare the same seeds with shipped results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Read a result schema and its rows without changing numerical values."""
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


def number(row: dict, key: str) -> float:
    """Reject missing or nonfinite scientific metrics."""
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {key}')
    return value


def collect(runs_dir: Path, results_dir: Path = ROOT / 'results') -> tuple[list[dict], list[dict]]:
    """Select scheduled classification probes and validated forecast replays from completed runs."""
    fields, _ = read_csv(results_dir / 'runs.csv')
    paths = [runs_dir / 'run.json'] if (runs_dir / 'run.json').exists() else sorted(runs_dir.glob('*/run.json'))
    if not paths:
        raise ValueError(f'No run.json records under {runs_dir}; use scripts/train.py to record run identities')
    rows, forecasts, identities = [], [], set()
    for path in paths:
        record = json.loads(path.read_text())
        identity = record['experiment_id']
        if identity in identities or record['exit_code'] != 0:
            raise ValueError(f'Duplicate or incomplete run: {identity}')
        identities.add(identity)
        history = json.loads(path.with_name('metrics.json').read_text())['history']
        if not history or len({r['epoch'] for r in history}) != len(history):
            raise ValueError(f'Empty or duplicated epoch history: {identity}')
        row = {key: record.get(key, '') for key in fields}
        row['status'] = 'completed'
        cfg = record['cfg']
        if cfg['data']['name'] in {'ETTh1', 'electricity'}:
            from pmt.forecast import validation_milestone

            mode = 'uni' if cfg['data']['univariate'] else 'multi'
            epoch = validation_milestone(history, mode, path.parent)
            selected = next(r for r in history if int(r['epoch']) == epoch)
            replay_path = path.with_name('replay.json')
            if not replay_path.exists():
                raise ValueError(f'Run scripts/evaluate.py --replay-run {path.parent} before summarizing forecasts')
            replay = json.loads(replay_path.read_text())['metrics']
            if replay['epoch'] != epoch:
                raise ValueError(f'Replay is not the validation-selected epoch: {identity}')
            for grid in ('wide23', 'source13'):
                packet = replay[grid]
                for horizon in (*cfg['data']['horizons'], 'avg'):
                    key = f'{mode}/avg_test_' if horizon == 'avg' else f'{mode}/horizon_{horizon}_test_'
                    forecasts.append(
                        dict(
                            method='pmt',
                            dataset=record['dataset'],
                            mode=mode,
                            grid=grid,
                            seed=record['seed'],
                            horizon=horizon,
                            mse=number(packet, key + 'mse_norm'),
                            mae=number(packet, key + 'mae_norm'),
                            selected_epoch=epoch,
                            selected_step=selected['step'],
                            selection_split='val',
                        )
                    )
        else:
            for budget in (0.01, 0.05):
                prefix = f'probe/svm/lp{budget:.3f}/'
                candidates = [r for r in history if prefix + 'top1_acc' in r]
                if not candidates:
                    raise ValueError(f'No scheduled SVM probes for {identity}, budget {budget}')
                selected = max(
                    candidates,
                    key=lambda r: (number(r, prefix + 'top1_acc'), number(r, prefix + 'mf1'), -int(r['epoch'])),
                )
                for key in ('top1_acc', 'mf1'):
                    row[prefix + 'global_best_' + key] = number(selected, prefix + key)
        rows.append(row)
    keys = [(r['method'], r['dataset'], r['mode'], r['grid'], r['seed'], r['horizon']) for r in forecasts]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate forecasting seed/horizon rows')
    return rows, forecasts


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write the shipped column order with unrounded per-run values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], shipped: list[dict], *, forecasting: bool = False) -> list[str]:
    """Report per-arm sample mean/SD and paired differences for the same identities."""
    keys = ('method', 'dataset', 'mode', 'grid', 'seed', 'horizon') if forecasting else ('experiment_id',)
    lookup = {tuple(str(row[key]) for key in keys): row for row in shipped}
    groups = defaultdict(list)
    for row in rows:
        identity = tuple(str(row[key]) for key in keys)
        if forecasting:
            arm = tuple(str(row[k]) for k in keys if k != 'seed')
            metrics = ('mse', 'mae')
        else:
            arm = (row['source'], row['experiment_id'].rsplit('_s', 1)[0], row['dataset'])
            metrics = [key for key in row if key.startswith('probe/') and row[key] != '']
        for key in metrics:
            reference = lookup.get(identity, {})
            expected = number(reference, key) if reference.get(key) not in (None, '') else None
            groups[(*arm, key)].append((number(row, key), expected))
    output = []
    for group, pairs in sorted(groups.items()):
        values = [p[0] for p in pairs]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else float('nan')
        comparable = [observed - reference for observed, reference in pairs if reference is not None]
        delta = f'{statistics.mean(comparable):+.8g}' if len(comparable) == len(values) else 'unavailable'
        output.append(f'{" | ".join(group)}: n={len(values)} mean={mean:.8g} sd={sd:.8g} delta={delta}')
    return output


def main() -> None:
    """Write per-run CSVs and print seed summaries against shipped rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/runs.csv')
    parser.add_argument('--forecast-out', type=Path, default=ROOT / 'runs/table3_per_seed.csv')
    args = parser.parse_args()
    rows, forecasts = collect(args.runs_dir, args.results_dir)
    fields, shipped = read_csv(args.results_dir / 'runs.csv')
    forecast_fields, forecast_shipped = read_csv(args.results_dir / 'table3_per_seed.csv')
    write_csv(args.out, fields, rows)
    write_csv(args.forecast_out, forecast_fields, forecasts)
    for line in (*summarize(rows, shipped), *summarize(forecasts, forecast_shipped, forecasting=True)):
        print(line)
    print(f'{args.out}: {len(rows)} completed runs; {args.forecast_out}: {len(forecasts)} forecast rows')


if __name__ == '__main__':
    main()
