## E.1 fully supervised (250 epochs, test split, seeds 42-44)

| Dataset | PMT top-1 | PMT mF1 | Vanilla top-1 | Vanilla mF1 | Printed PMT | Printed vanilla | PMT - vanilla (top-1) |
|---|---|---|---|---|---|---|---|
| HAR | 94.56 ± 0.10 | 94.61 ± 0.09 | 94.27 ± 0.28 | 94.35 ± 0.29 | 97.8 / 98.0 | 97.7 / 97.9 | +0.29 (printed +0.1) |
| FordA | 92.88 ± 0.40 | 92.88 ± 0.40 | 92.63 ± 0.38 | 92.61 ± 0.38 | 91.5 / 91.5 | 85.5 / 85.5 | +0.25 (printed +6.0) |
| FordB | 81.93 ± 0.19 | 81.93 ± 0.18 | 79.55 ± 0.84 | 79.54 ± 0.84 | 79.3 / 79.2 | 80.5 / 80.5 | +2.39 (printed -1.2) |
| Wafer | 99.70 ± 0.07 | 99.23 ± 0.19 | 99.52 ± 0.07 | 98.75 ± 0.20 | 99.8 / 99.5 | 99.5 / 98.7 | +0.18 (printed +0.3) |

## E.2 HAR short run (50 epochs, seeds 42-44)

Supervised: best test epoch. SSL: SVM probe at 5% labels, best test probe (Table 1 rule).

| Model | Sup top-1 | Sup mF1 | SSL 5% top-1 | SSL 5% mF1 | Printed sup | Printed SSL 5% |
|---|---|---|---|---|---|---|
| PMT | 94.49 ± 0.09 | 94.58 ± 0.09 | 91.03 ± 1.98 | 91.12 ± 1.92 | 97.1 / 97.4 | 93.6 / 93.9 |
| Vanilla Transformer | 92.42 ± 0.26 | 92.53 ± 0.19 | 85.56 ± 1.92 | 85.19 ± 2.13 | 97.3 / 97.6 | 91.9 / 92.0 |
| Windowed Transformer (with temporal state pass) | 93.88 ± 0.65 | 93.96 ± 0.64 | 90.23 ± 1.24 | 90.17 ± 1.25 | 96.7 / 97.0 | 93.1 / 93.3 |
| Windowed Transformer (no state pass) | 94.22 ± 0.49 | 94.33 ± 0.51 | 89.13 ± 0.81 | 89.22 ± 0.81 | 96.8 / 97.1 | 93.0 / 93.2 |

## E.3 reset-state robustness (HAR, supervised, 250 epochs, global seed 42, test split)

| | Reset seed 42 | Reset seed 123 | Reset seed 456 | Reset seed 789 | Reset seed 2024 | Mean ± SD (population / sample) |
|---|---|---|---|---|---|---|
| Rerun top-1 | 93.69 | 94.30 | 94.91 | 93.96 | 94.71 | 94.31 ± 0.45 / 0.51 |
| Rerun mF1 | 93.74 | 94.37 | 94.95 | 94.02 | 94.79 | |
| Best epoch | 177 | 183 | 102 | 127 | 197 | |
| Printed top-1 | 96.13 | 96.19 | 96.06 | 95.99 | 95.99 | 96.07 ± 0.08 / 0.09 |

Repeat of reset seed 42: top-1 94.44, mF1 94.51, best epoch 76 (first run 93.69 / 93.74, epoch 177).
