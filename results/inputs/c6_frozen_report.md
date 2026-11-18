# Composite C6 cost report on the frozen final recipes, 2 November 2026

**Status:** COMPLETE

Input for `experiments/tables/c6_cost_table.py` and `experiments/tables/build_tables.py`. The 18 non-HAR rows are copied verbatim from the GH200 benchmark of 23 October (frozen final recipes, GH200, 23 October). The three HAR rows come from the fresh-process one-slot measurements `appendix_c_raw/c6_har_*_report.md` (1 November), which replace the 23 October HAR cells (xLSTM there was measured at batch 128 after an out-of-memory retry). Every cell is at batch 256. Not a new measurement of the other six datasets.

## Results

| Dataset | Arm | Batch | Params (M) | Train ms/step median [Q1–Q3] | Peak train GB | Inference seq/s (median) | Inference ms/batch |
|---|---|---:|---:|---:|---:|---:|---:|
| HAR | PMT | 256 | 28.076 | 512.81 [507.20–516.53] | 25.18 | 988 | 259.06 |
| HAR | xLSTM adapter | 256 | 10.170 | 1427.79 [1410.93–1450.28] | 98.51 | 605 | 423.49 |
| HAR | Transformer-XL adapter | 256 | 18.350 | 206.42 [205.41–209.40] | 18.52 | 7265 | 35.24 |
| epilepsy | PMT | 256 | 18.701 | 572.87 [558.16–585.81] | 11.29 | 2335 | 109.65 |
| epilepsy | xLSTM adapter | 256 | 8.029 | 784.24 [771.87–795.13] | 45.16 | 1239 | 206.63 |
| epilepsy | Transformer-XL adapter | 256 | 12.205 | 68.63 [66.41–70.38] | 5.05 | 27458 | 9.32 |
| UCR_Wafer | PMT | 256 | 18.046 | 662.27 [652.74–677.44] | 10.07 | 1906 | 134.31 |
| UCR_Wafer | xLSTM adapter | 256 | 7.371 | 672.90 [662.51–691.15] | 38.35 | 1530 | 167.37 |
| UCR_Wafer | Transformer-XL adapter | 256 | 11.547 | 67.02 [64.93–69.37] | 4.22 | 28184 | 9.08 |
| UCR_FordA | PMT | 256 | 21.843 | 212.77 [211.04–215.08] | 11.15 | 4317 | 59.30 |
| UCR_FordA | xLSTM adapter | 256 | 14.734 | 527.87 [517.91–534.03] | 31.41 | 1916 | 133.61 |
| UCR_FordA | Transformer-XL adapter | 256 | 17.518 | 81.06 [79.68–81.79] | 6.51 | 22461 | 11.40 |
| UCR_FordB | PMT | 256 | 19.378 | 208.05 [207.04–209.44] | 10.73 | 4373 | 58.55 |
| UCR_FordB | xLSTM adapter | 256 | 12.269 | 521.85 [515.20–529.21] | 30.55 | 1928 | 132.81 |
| UCR_FordB | Transformer-XL adapter | 256 | 15.053 | 73.66 [70.52–76.51] | 5.66 | 24364 | 10.51 |
| UCR_PhalangesOutlinesCorrect | PMT | 256 | 22.439 | 428.43 [423.97–445.11] | 6.04 | 3180 | 80.50 |
| UCR_PhalangesOutlinesCorrect | xLSTM adapter | 256 | 13.553 | 329.91 [326.23–332.91] | 17.95 | 3090 | 82.84 |
| UCR_PhalangesOutlinesCorrect | Transformer-XL adapter | 256 | 17.033 | 69.47 [68.44–70.46] | 3.04 | 22834 | 11.21 |
| UCR_ElectricDevices | PMT | 256 | 24.207 | 329.94 [324.00–332.66] | 7.35 | 2672 | 95.82 |
| UCR_ElectricDevices | xLSTM adapter | 256 | 13.532 | 258.14 [255.49–259.80] | 13.19 | 3934 | 65.07 |
| UCR_ElectricDevices | Transformer-XL adapter | 256 | 17.708 | 63.87 [62.27–66.27] | 2.17 | 21107 | 12.13 |

