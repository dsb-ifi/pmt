# Paper archive

This archive contains the final NeurIPS 2026 table and figure sources.
The [release README](../README.md) gives the commands for training and scoring
runs. The tables below link to the numerical results.

## Tables

| Paper table | Result files |
| --- | --- |
| [1. Classification](tables/table1_classification.tex) | [Per-run measurements](../results/runs.csv), [table cells](../results/tables/table1_cells.csv), [published baselines](../results/inputs/table1_published_baselines.json) |
| [2. Cue retention](tables/table2_cue_retention.tex) | [Cue retention scores](../results/inputs/table2_summary.csv) |
| [3. Forecasting](tables/table3_forecasting.tex) | [Per-seed, per-horizon results for both grids](../results/table3_per_seed.csv) |
| [4. Backbone comparison](tables/table4_architecture.tex) | [Per-run measurements](../results/runs.csv), [run mapping](../results/tables/table4_checks.json), [per-budget cells](../results/tables/table4.csv) |
| [5. Loss ablations](tables/table5_loss_ablation.tex) | [Per-run measurements](../results/runs.csv), [per-dataset means](../results/tables/table5.csv) |
| [6. Tokenizer patch size and stride](tables/table6_patch_stride.tex) | [Per-run measurements](../results/runs.csv), [table cells](../results/tables/table6.csv) |
| [C.1. Streaming](tables/tableC1_streaming.tex), [C.2. Vectorization](tables/tableC2_vectorized.tex), [C.3. Aggregation](tables/tableC3_aggregator.tex), and [C.6. Cost](tables/tableC6_cost.tex) | [Reported measurements](../results/inputs/appendix_c_paper_rows.json), [benchmark records](../results/inputs/appendix_c_raw/) |
| [D.1. Datasets](tables/tableD1_datasets.tex) | [Dataset statistics](../results/tables/tableD1.csv) |
| [E.1. Supervised](tables/tableE1_supervised_full.tex), [E.2. HAR short windows](tables/tableE2_har_short.tex), and [E.3. Reset robustness](tables/tableE3_reset_robustness.tex) | [Per-run measurements](../results/runs.csv), [table data](../results/tables/appendix_e123.csv) |
| [E.4. Window size](tables/tableE4_window_size.tex) | [Per-run measurements](../results/runs.csv), [table data](../results/tables/appendix_e4.csv), [window plan](../results/inputs/appendix_e4_plan.json) |

Table D.1 comes from the inline dataset table in the final manuscript.
The forecast alternatives are a [layout without seed spreads](tables/table3_forecasting_no_spreads.tex) and a [sensitivity check with SoftCLT's 13-value ridge grid](tables/table3_forecasting_source13.tex).
The archive also includes the [Wafer and POC variant of Table 5(b)](tables/table5_loss_ablation_b_poc_wafer.tex) and the [July recipes behind the Table C.6 cost comparison](tables/tableC6_cost_july_recipes.tex).

## Figures

The visual sources are in [figures/](figures/) and the plotted values are in
[data/](data/). Run `lualatex figures.tex` from this directory for a one-page-per-figure gallery using the paper's TikZ and PGFPlots styles.
[Gallery driver](figures.tex)

| Figure | Source | Data under `data/` |
| --- | --- | --- |
| Representation hierarchy, PMA information flow, PMT architecture | [Hierarchy](figures/representation-hierarchy.tikz), [PMA](figures/pma-information-flow.tikz), [PMT](figures/pmt-architecture.tikz) | Drawings |
| Hybrid similarities | [Source](figures/hybrid-similarities.tikz) | `slice_plot_har_17_13_12_*` |
| Different-class hybrid similarities | [Source](figures/hybrid-different-classes.tikz) | `slice_plot_har_6_5_24_*` |
| Same-class HAR | [Source](figures/har-same-class.tikz) | `rep_sim_6_12.txt`, `state_sim_6_12.txt`, `wave_{6,12}.txt` |
| Epilepsy similarities | [Source](figures/epilepsy-similarities.tikz) | `epilepsy-*` |
| HAR comparison with SoftCLT | [Source](figures/softclt-hybrid-similarities.tikz) | `slice_plot_har_17_13_12_*`, `softclt_ts2vec_slice_plot_*` |
| Standing figure (t-SNE of upstairs/downstairs representations) | [Source](figures/pca-standing.tikz) · [Paper float](figures-camera-ready/fig_pca_repr_standing.tex) | `rainbow_plot/` |
| Walking representations (PCA) | [Source](figures/pca-walking.tikz) · [Paper float](figures-camera-ready/fig_pca_repr.tex) | `*_hybrid_walking-walking_downstairs-walking.txt` |

For the paper's heat maps and scatter plots, representations are centered by subtracting the
clean validation mean at each sequence position.
The final source also retains an earlier HAR splice whose figure was removed from the camera-ready PDF.
[Ideal hybrid source](figures/ideal-hybrid-similarities.tikz) · [Plot data](data/slice_plot_har_91_1295_592_class-5_class-4_class-5/)
