# Datasets

The [dataset downloader](../scripts/download_data.py) prepares and verifies the
datasets under `data/`. Use `--only` to select the dataset families for your run.

```bash
python scripts/download_data.py --only har epilepsy etth1
python scripts/download_data.py --only electricity
```

The UCR briefing's password clue points to a sentence in the introduction of
the archive paper. [Briefing](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/BriefingDocument2018.pdf) · [Archive paper](https://arxiv.org/abs/1810.07758) · [UCR download command](../README.md#download-data)

Allow a few minutes for the three HAR downloads. Preparing Electricity from
the archive requires pandas. With an existing prepared file, pass
`--only electricity --electricity-csv /path/to/electricity.csv` to verify and
reuse it. `--root` selects the output directory. [Paper dataset table](../paper/tables/tableD1_datasets.tex) · [CSV](../results/tables/tableD1.csv)

| Dataset | Used in | Source | Layout under `data/` |
| --- | --- | --- | --- |
| HAR | Classification | [TS-TCC preprocessed, DR-NTU (Data)](https://doi.org/10.21979/N9/RMFXOX), CC BY-NC 4.0 | `HAR/{train,val,test}.pt` |
| Epilepsy | Classification | [TS-TCC preprocessed, DR-NTU (Data)](https://doi.org/10.21979/N9/4PZQJ7), CC BY-NC 4.0 | `epilepsy/{train,val,test}.pt` |
| FordA, FordB, PhalangesOutlinesCorrect, Wafer, ElectricDevices, StarLightCurves | Classification | [UCR Time Series Archive 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | `UCR/<name>/<name>_{TRAIN,TEST}.tsv` |
| ETTh1 | Forecasting | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | `ETT-small/ETTh1.csv` |
| Electricity | Forecasting | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), preprocessed as in the [TS2Vec repository](https://github.com/yuezhihan/ts2vec) | `electricity/electricity.csv` |

Table 1 uses the eight classification datasets above. Table 3 uses ETTh1 and Electricity, each in univariate and multivariate mode.
Each dataset keeps the licence and citation terms of its source. The paper's appendix cites them.

The HAR `test.pt` on Dataverse holds the same tensors as the copy the paper's runs read, in a different
serialization (the two files differ in bytes, not in values).
