## Patch size (fraction of sequence length) at the recipe stride

| Patch size | FordA Top-1 | FordA mF1 | FordB Top-1 | FordB mF1 | Epilepsy Top-1 | Epilepsy mF1 |
|---|---:|---:|---:|---:|---:|---:|
| 2.5% | 76.1 | 76.1 | 68.9 | 67.6 | **96.8** | **94.9** |
| 5% | 85.7 | 85.7 | 70.3 | 69.1 | _96.7_ | _94.8_ |
| 7.5% | 88.5 | 88.5 | _76.2_ | _76.2_ | _96.7_ | _94.8_ |
| 10% | _89.1_ | _89.1_ | **76.9** | **76.8** | 96.5 | 94.5 |
| 12.5% | _89.1_ | _89.1_ | 75.7 | 75.6 | 96.4 | 94.2 |
| 15% | **89.5** | **89.4** | 75.8 | 75.6 | 96.3 | 94.2 |

## Stride (fraction of patch size) at the recipe patch size

| Stride | FordA Top-1 | FordA mF1 | FordB Top-1 | FordB mF1 | Epilepsy Top-1 | Epilepsy mF1 |
|---|---:|---:|---:|---:|---:|---:|
| 10% | **88.3** | **88.3** | **76.5** | **76.4** | **96.9** | **95.1** |
| 25% | _86.1_ | _86.1_ | _73.3_ | _72.9_ | **96.9** | **95.1** |
| 50% | 72.0 | 71.8 | 69.3 | 69.0 | _96.7_ | _94.8_ |
| 75% | 67.9 | 67.6 | 63.4 | 62.4 | 96.6 | 94.7 |
| 100% | 65.5 | 64.2 | 59.5 | 58.6 | 96.5 | 94.5 |

Mean of 1% and 5% over seeds 42-44; bold = best, italic = second best per column (one decimal).

## Quantities quoted in the prose

| Dataset | Patch sweep top-1 range (all) | Range from 7.5% | Stride 10% -> 100% top-1 change |
|---|---:|---:|---:|
| FordA | 13.3 | 1.0 | -22.8 |
| FordB | 8.0 | 1.2 | -17.1 |
| Epilepsy | 0.4 | 0.3 | -0.4 |
