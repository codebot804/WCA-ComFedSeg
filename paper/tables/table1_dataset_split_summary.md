# Table 1. Dataset and split summary

Source CSV: `results/summaries/phase5p_busi_vs_bus_uclm_dataset_summary.csv`. BUSI lacks patient or study identifiers in the local loader, so leakage control is sample-stem based. BUS-UCLM uses patient/study prefix disjointness. Normal cases are deferred in both ultrasound datasets.

| Dataset | Modality | Included labels | Included samples | Benign | Malignant | Malignant ratio | Deferred normal | Identity unit | Mean mask area | Leakage rule | Scope note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUSI | breast ultrasound | Benign;Malignant | 647 | 437 | 210 | 0.325 | 133 | not_available | 0.0944 | sample stem disjoint; no patient/study identifier available | Benign+Malignant only; Normal deferred. |
| BUS-UCLM | breast ultrasound | Benign;Malignant | 264 | 174 | 90 | 0.341 | 419 | 36 | 0.0758 | patient/study prefix disjoint across train/val/test and clients | Benign+Malignant only; Normal deferred. |
