# Code provenance

The scripts in `analysis/` are clean repository copies of the final analysis
scripts stored with the N = 43 manuscript package. The following names were
changed to make the execution order explicit:

| Repository script | Final-package source |
|---|---|
| `01_compute_erp.py` | `Compute_ERP_grand_averages_no1028.py` |
| `02_cluster_erp.py` | `Cluster_ERP_waveforms_no1028.py` |
| `03_plot_erp.py` | `Plotting_ERPS_18_12_no1028.py` |
| `04_compute_tfr.py` | `Compuring_TRFs_no1028.py` |
| `05_cluster_tfr.py` | `Running_cluster_29_01_no1028.py` |
| `06_cluster_cnv_interaction.py` | `Cluster_CNV_interaction_no1028.py` |
| `07_plot_cnv_tfr.py` | `Plot_CNV_TF_composite_no1028.py` |
| `08_plot_frn_tfr.py` | `Plot_FRN_TF_3way_composite_no1028.py` |
| `09_frn_reward_analysis.py` | `FRN_reward_vs_noreward_no1028.py` |

`Running_cluster_no1028.py` was an earlier cluster-analysis version superseded
by `Running_cluster_29_01_no1028.py`. `build_no1028_doc.py` updated manuscript
text programmatically but performed no scientific analysis. Neither is part of
the released scientific pipeline.

Repository-release changes are documented in `README.md`.
