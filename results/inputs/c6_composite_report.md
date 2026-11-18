# Composite C6 cost report, 1 November 2026

**Status:** COMPLETE

Input of `experiments/tables/c6_cost_table.py`. Non-HAR rows come from the GH200 benchmark of 16 October; the three HAR rows come from the fresh-process one-slot measurements in `appendix_c_raw/c6_har_*_report.md`. Not a new measurement of the other datasets.

## Results

| Dataset | Arm | Batch | Params (M) | Train ms/step median [Q1–Q3] | Peak train GB | Inference seq/s (median) | Inference ms/batch |
|---|---|---:|---:|---:|---:|---:|---:|
| HAR | PMT | 256 | 28.076 | 512.81 [507.20–516.53] | 25.18 | 988 | 259.06 |
| HAR | xLSTM adapter | 256 | 10.170 | 1427.79 [1410.93–1450.28] | 98.51 | 605 | 423.49 |
| HAR | Transformer-XL adapter | 256 | 18.350 | 206.42 [205.41–209.40] | 18.52 | 7265 | 35.24 |
| epilepsy | PMT | 256 | 18.701 | 497.76 [488.37–504.36] | 11.20 | 2774 | 92.27 |
| epilepsy | xLSTM adapter | 256 | 8.029 | 655.44 [648.16–664.49] | 45.07 | 1353 | 189.22 |
| epilepsy | Transformer-XL adapter | 256 | 12.205 | 63.51 [61.89–64.38] | 4.96 | 31293 | 8.18 |
| UCR_Wafer | PMT | 256 | 18.046 | 773.25 [766.83–783.36] | 25.81 | 1330 | 192.46 |
| UCR_Wafer | xLSTM adapter | 128 (OOM retry) | 7.371 | 1479.31 [1471.18–1494.75] | 57.51 | 340 | 376.56 |
| UCR_Wafer | Transformer-XL adapter | 256 | 11.547 | 173.38 [172.61–176.06] | 14.77 | 8634 | 29.65 |
| UCR_FordA | PMT | 256 | 21.843 | 189.23 [186.65–192.81] | 11.07 | 4908 | 52.16 |
| UCR_FordA | xLSTM adapter | 256 | 14.734 | 440.77 [434.50–449.39] | 31.32 | 2074 | 123.43 |
| UCR_FordA | Transformer-XL adapter | 256 | 17.518 | 72.34 [71.01–73.87] | 6.42 | 24394 | 10.49 |
| UCR_FordB | PMT | 256 | 19.378 | 182.17 [179.46–184.18] | 10.65 | 4942 | 51.80 |
| UCR_FordB | xLSTM adapter | 256 | 12.269 | 436.41 [429.22–439.68] | 30.47 | 2076 | 123.32 |
| UCR_FordB | Transformer-XL adapter | 256 | 15.053 | 66.21 [65.37–67.77] | 5.57 | 27881 | 9.18 |
| UCR_PhalangesOutlinesCorrect | PMT | 256 | 22.439 | 280.78 [275.61–289.15] | 4.43 | 4674 | 54.77 |
| UCR_PhalangesOutlinesCorrect | xLSTM adapter | 256 | 13.553 | 205.67 [201.46–211.64] | 11.15 | 4806 | 53.27 |
| UCR_PhalangesOutlinesCorrect | Transformer-XL adapter | 256 | 17.033 | 60.93 [59.33–62.55] | 2.01 | 26086 | 9.81 |
| UCR_ElectricDevices | PMT | 256 | 24.209 | 480.47 [475.51–485.89] | 27.09 | 1701 | 150.48 |
| UCR_ElectricDevices | xLSTM adapter | 256 | 13.534 | 1036.92 [1022.96–1053.26] | 76.39 | 838 | 305.51 |
| UCR_ElectricDevices | Transformer-XL adapter | 256 | 17.710 | 128.37 [126.66–130.00] | 11.75 | 12863 | 19.90 |

