# Phase 5E Real hard_noniid Split Validation

This report validates real-data split difficulty only. It does not run training, Tier 1 pilots, or failed variants.

## Scope

- Image size for mask-area diagnosis: 128
- BUSI classes used: benign and malignant only
- BUSI normal class: deferred
- Kvasir-SEG pairing: image/mask filename stem
- Leakage identity: original image stem namespaced by source folder

## Dataset Counts

- BUSI valid benign/malignant pairs: 647
- BUSI images with multiple masks merged: 17
- Kvasir-SEG valid pairs: 1000

## BUSI hard_noniid vs moderate_noniid

- sample_count_imbalance_ratio: moderate=1.893805, hard=2.494624, hard_greater=True
- mean_mask_area_client_range: moderate=0.065125, hard=0.113745, hard_greater=True
- malignant_ratio_client_range: moderate=0.683875, hard=0.752964, hard_greater=True
- Validation passed: True
- Issues:
- None

## Kvasir-SEG hard_noniid vs moderate_noniid

- sample_count_imbalance_ratio: moderate=1.333333, hard=1.866667, hard_greater=True
- mean_mask_area_client_range: moderate=0.228345, hard=0.242404, hard_greater=True
- large_mask_ratio_client_range: moderate=0.561905, hard=0.654762, hard_greater=True
- Validation passed: True
- Issues:
- None

## Output Files

- `results\summaries\phase5e_busi_hard_noniid_client_distribution.csv`
- `results\summaries\phase5e_kvasir_seg_hard_noniid_client_distribution.csv`
- `results\summaries\phase5e_busi_moderate_vs_hard_noniid_comparison.csv`
- `results\summaries\phase5e_kvasir_seg_moderate_vs_hard_noniid_comparison.csv`
- `results\figures\phase5e_real_hard_noniid_splits/`
