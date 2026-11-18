"""Train a registered paper run, a dataset recipe, or a custom YAML configuration."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Resolve the selected recipe and call the single-device training loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--id', help='exact paper run identifier, including its seed')
    source.add_argument('--dataset', help='dataset name for its paper PMT recipe')
    source.add_argument('--config', type=Path, help='path to a YAML recipe, optionally extending configs/base.yaml')
    source.add_argument('--list', action='store_true', help='list the registered run identifiers')
    parser.add_argument('--seed', type=int, help='seed for --dataset or --config (default 42)')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    from pmt.config import load_run, runs

    if args.list:
        print('\n'.join(runs()))
        return
    if args.id:
        if args.seed is not None:
            parser.error('Choose the seed through --id')
        run_id = args.id
        cfg = load_run(run_id, args.override)
        row = runs()[run_id]
        if cfg.train.seed != row['seed']:
            parser.error('Choose the seed through --id so the recorded run identity stays correct')
    else:
        from pmt.config import Config, load

        if args.dataset and args.dataset not in {row['dataset'] for row in runs().values()}:
            parser.error(f'Unknown paper dataset {args.dataset!r}')
        path = args.config or ROOT / 'configs' / f'{args.dataset}.yaml'
        values = load(str(path), args.override).to_dict()
        name = values.get('data', {}).get('name')
        if not isinstance(name, str) or name in {'', '.', '..'} or Path(name).name != name:
            parser.error('Set data.name to a single dataset directory name')
        seed = args.seed if args.seed is not None else values['train']['seed']
        run_id = f'{name}_s{seed}'
        values['train'].update(seed=seed, run_id=run_id)
        cfg = Config(values)
        row = dict(source='dataset' if args.dataset else 'config', arm=name, tables='')

    from pmt.train import train

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_dir = args.run_dir or ROOT / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(
        experiment_id=run_id,
        source=row['source'],
        arm=row['arm'],
        dataset=cfg.data.name,
        seed=cfg.train.seed,
        tables=row['tables'],
        cfg=cfg.to_dict(),
        exit_code=None,
    )
    path = run_dir / 'run.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    train(cfg, run_dir)
    if not args.id and any(cfg.data.get(key) is None for key in ('channels', 'length', 'classes')):
        import torch

        record['cfg'] = torch.load(run_dir / 'encoder.pt', weights_only=True, mmap=True)['cfg']
    record['exit_code'] = 0
    path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
