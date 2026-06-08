# Final Table and Figure Captions

## Main tables

**Table 1. Dataset and split summary.** Source: `paper/final_preparation/source_data/table1_dataset_split_summary_source.csv`, derived from `results/summaries/phase5p_busi_vs_bus_uclm_dataset_summary.csv`. Reports included benign/malignant samples, deferred normal cases, available identity units, mask area statistics, and leakage-control rule. Main/appendix status: main Methods/Experiments dataset table.

**Table 2. BUSI moderate non-IID main comparison.** Source: `paper/final_preparation/source_data/table2_busi_moderate_main_results_source.csv`, derived from `results/summaries/phase5m_moderate_multiseed_aggregate.csv`. Three random seeds are reported as mean +/- SD. Main/appendix status: main result table; safe positive evidence for WCA-Comm on BUSI moderate non-IID.

**Table 3. Fairness and communication summary.** Source: `paper/final_preparation/source_data/table3_fairness_communication_summary_source.csv`, derived from `results/summaries/phase5m_moderate_multiseed_aggregate.csv`. Emphasizes worst-client Dice, client-level imbalance, best-worst gap, uploaded MB, and communication reduction. Main/appendix status: main fairness/communication analysis table.

## Appendix tables

**Appendix Table S1. BUS-UCLM moderate non-IID boundary results.** Sources: `phase5q_bus_uclm_moderate_pilot.csv` and `phase5s_bus_uclm_wca_bn_scoped_pilot.csv`. Single-seed pilot/diagnostic only; not final main evidence.

**Appendix Table S2. Hard non-IID and Kvasir boundary results.** Source: `phase5k_core_hard_multiseed_aggregate.csv`. Three-seed boundary/appendix analysis; should be used to discuss limitations and scope, not to claim the method solved hard non-IID generally.

## Figures

**Figure 2. BUSI moderate non-IID main result.** Source: `paper/final_preparation/source_data/figure2_busi_moderate_source.csv`, derived from `phase5m_moderate_multiseed_aggregate.csv`. Shows average Dice, worst-client Dice, best-worst client gap, and uploaded MB for FedAvg, FedProx, FedBN, and WCA-Comm. Main evidence; values are three-seed mean with SD where available.

**Figure 3. Fairness-communication trade-off on BUSI moderate non-IID.** Source: `paper/final_preparation/source_data/figure3_tradeoff_source.csv`, derived from `phase5m_moderate_multiseed_aggregate.csv`. Shows uploaded MB versus worst-client Dice, uploaded MB versus best-worst gap, and rank-style comparison of fairness/communication outcomes. Main evidence.

**Appendix Figure S1. Boundary diagnostics across BUS-UCLM, BUSI hard non-IID, and Kvasir-SEG.** Source: `paper/final_preparation/source_data/appendix_boundary_figure_source.csv`, derived from `phase5q`, `phase5s`, and `phase5k` summaries. BUS-UCLM rows are single-seed pilot/diagnostic; hard/Kvasir rows are boundary evidence. This figure should support limitations and scope discussion.
