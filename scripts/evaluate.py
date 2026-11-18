"""Score a frozen encoder using a fresh SVM grid or its saved SVM settings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_encoder(path: Path) -> dict:
    """Read an encoder written by training."""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not {'state_dict', 'cfg', 'epoch'} <= payload.keys():
        raise ValueError('Encoder checkpoint requires state_dict, cfg and epoch')
    return payload


def main() -> None:
    """Load weights strictly, evaluate without SSL updates and write metrics.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='score an encoder written by training')
    source.add_argument(
        '--replay-run', type=Path, help='score the validation-selected forecast milestone on both grids'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-svm', action='store_true', help='refit with the SVM settings saved with a best encoder')
    mode.add_argument('--cue', action='store_true', help='replay the Table 2 EOS cue probe')
    parser.add_argument('--draws', type=int, help='repeat classification from successive noise draws')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.draws is not None and (args.draws < 1 or args.cue):
        parser.error('--draws requires a positive count and excludes --cue')
    if args.replay_run and (args.cue or args.run_svm or args.draws is not None):
        parser.error('--replay-run excludes --cue/--run-svm/--draws')
    from pmt.config import Config
    from pmt.data import load_dataset
    from pmt.model import SSLModel
    from pmt.probe import Probe

    payload = load_encoder(args.checkpoint or args.replay_run / 'encoder.pt')
    values = payload['cfg']
    values['data']['root'] = str(args.data_root)
    cfg = Config(values)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    splits = load_dataset(cfg)
    if payload.get('mean') is not None:
        splits = replace(splits, mean=payload['mean'], std=payload['std'])
    if args.draws is not None and splits.series is not None:
        parser.error('--draws applies only to the classification probe')
    draw_summary = None
    if args.replay_run:
        if args.cue or args.run_svm or splits.series is None:
            parser.error('--replay-run requires a forecasting run and excludes --cue/--run-svm')
        from pmt.forecast import replay

        metrics = replay(args.replay_run, splits, device=args.device)
    else:
        # Follow training's normalization-before-model bootstrap before noisy probes.
        encoder = SSLModel(cfg).encoder.to(args.device)
        encoder.load_state_dict(payload['state_dict'], strict=True)
        probe = Probe(cfg)
        if args.run_svm:
            if args.cue or splits.series is not None:
                parser.error('--run-svm applies only to the classification probe')
            if 'svm_params' not in payload:
                parser.error('--run-svm requires best_encoder.pt or best_encoder_5pct.pt')
            probe.best_params = payload['svm_params']
        if args.cue:
            from pmt.cue import evaluate

            metrics = evaluate(encoder, splits, cfg)
        elif splits.series is not None:
            from pmt.forecast import evaluate

            metrics = evaluate(encoder, splits, cfg, test=True)
        elif args.draws is not None:
            metrics, draw_summary = probe.evaluate_draws(encoder, splits, args.draws, fixed=args.run_svm)
        else:
            metrics = probe.evaluate(encoder, splits, int(payload['epoch']) + 1, fixed=args.run_svm)
    out = args.out or (args.replay_run / 'replay.json' if args.replay_run else ROOT / 'runs/evaluate/metrics.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'metrics': metrics}
    if draw_summary is not None:
        result['draws'] = draw_summary
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    if draw_summary is not None:
        print(json.dumps({'draws': draw_summary}, indent=2))
    print(f'details: {out}')


if __name__ == '__main__':
    main()
