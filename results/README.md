# Results

These files contain the paper's measurements and table data. See the
[table index](../paper/README.md#tables) for the link from each published
table to its result files.

| Path | Content |
| --- | --- |
| [runs.csv](runs.csv) | Per-run data for Tables 1, 4, 5, 6, and E.1–E.4 (884 runs), with cohort, run ID, dataset, seed, status, table membership, and metrics. |
| [inputs/](inputs/) | External baseline values, Table 2 scores, Appendix C measurements, tokenizer geometry, and the E.4 window plan. |
| [table3_per_seed.csv](table3_per_seed.csv) | Table 3 test MSE and MAE per method, dataset, mode, ridge grid, seed, and horizon, with the selected milestone. |
| [tables/](tables/) | Archived per-cell CSV and Markdown data for the paper's tables. |

## Metrics in `runs.csv`

Floats are written with full precision; an empty cell means the run did not record the metric
(SSL runs have no supervised metrics and vice versa).

- `probe/svm/lp0.010/global_best_top1_acc`, `..._mf1` and the same for `lp0.050`: frozen-encoder
  RBF-SVM probe at 1% and 5% of the training labels, from the best probe scheduled during training
  (ties: higher F1, then the earliest epoch), as reported in Table 1.
- `supervised/val/global_best_top1_acc`, `..._mf1`, `..._epoch`: fully supervised runs (Appendix E),
  best epoch by top-1 on the evaluation split (the official test split), F1 from the same epoch.

## Cohorts (`source`)

`source` names the original run cohort. The [run registry](../configs/runs.yaml)
groups runnable entries by arm. A run ID has the form `<arm>_s<seed>`.
To summarize new runs, use [scripts/summarize.py](../scripts/summarize.py).
