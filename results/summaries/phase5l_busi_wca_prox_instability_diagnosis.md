# Phase 5L BUSI WCA+FedProx Instability Diagnosis

Phase 5L is a BUSI hard_noniid diagnostic only. No new training runs were launched.

## Logs Read

- `fedprox` seed 42: `results\logs\phase5k_core_hard_multiseed\20260606_170521_fedprox_busi_hard_noniid_seed42`
- `wca_comfedseg_prox` seed 42: `results\logs\phase5k_core_hard_multiseed\20260606_171607_wca_comfedseg_prox_busi_hard_noniid_seed42`
- `fedprox` seed 123: `results\logs\phase5k_core_hard_multiseed\20260606_170856_fedprox_busi_hard_noniid_seed123`
- `wca_comfedseg_prox` seed 123: `results\logs\phase5k_core_hard_multiseed\20260606_171948_wca_comfedseg_prox_busi_hard_noniid_seed123`
- `fedprox` seed 2025: `results\logs\phase5k_core_hard_multiseed\20260606_171241_fedprox_busi_hard_noniid_seed2025`
- `wca_comfedseg_prox` seed 2025: `results\logs\phase5k_core_hard_multiseed\20260606_172329_wca_comfedseg_prox_busi_hard_noniid_seed2025`

## Seed-Level Diagnosis

| seed | FedProx worst | WCA+FedProx worst | FedProx gap | WCA+FedProx gap | final weak client comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| 42 | 0.4819 | 0.5294 | 0.2106 | 0.0486 | WCA weakest client 2, Dice 0.5294; paired FedProx client 2, Dice 0.5694 |
| 123 | 0.5085 | 0.4017 | 0.0160 | 0.1466 | WCA weakest client 0, Dice 0.4017; paired FedProx client 0, Dice 0.5085 |
| 2025 | 0.4746 | 0.4359 | 0.1910 | 0.2114 | WCA weakest client 0, Dice 0.4359; paired FedProx client 0, Dice 0.4746 |

## Client-Level Findings

- Seed 42: FedProx worst-round counts {0: 10, 1: 0, 2: 0}; WCA+FedProx worst-round counts {0: 9, 1: 0, 2: 1}. Val split: client 0: n=51, malignant=0.06, mask_mean=0.050; client 1: n=20, malignant=0.35, mask_mean=0.152; client 2: n=27, malignant=0.81, mask_mean=0.126.
- Seed 123: FedProx worst-round counts {0: 8, 1: 1, 2: 1}; WCA+FedProx worst-round counts {0: 5, 1: 4, 2: 1}. Val split: client 0: n=51, malignant=0.06, mask_mean=0.054; client 1: n=20, malignant=0.35, mask_mean=0.163; client 2: n=27, malignant=0.81, mask_mean=0.137.
- Seed 2025: FedProx worst-round counts {0: 9, 1: 0, 2: 1}; WCA+FedProx worst-round counts {0: 7, 1: 2, 2: 1}. Val split: client 0: n=51, malignant=0.06, mask_mean=0.049; client 1: n=20, malignant=0.35, mask_mean=0.157; client 2: n=27, malignant=0.81, mask_mean=0.120.

## WCA Weighting Findings

- Seed 42: max-weight client counts {'0': 10, '1': 0, '2': 0}; rounds with weight >= 0.70: [(3, '0', '0.721230765'), (4, '0', '0.7396321067'), (6, '0', '0.7554331812'), (7, '0', '0.7374152923')].
- Seed 123: max-weight client counts {'0': 5, '1': 3, '2': 2}; rounds with weight >= 0.70: [(3, '0', '0.7560706402'), (5, '0', '0.7560706402'), (7, '0', '0.7560706402'), (9, '0', '0.7560706402')].
- Seed 2025: max-weight client counts {'0': 8, '1': 1, '2': 1}; rounds with weight >= 0.70: [(3, '0', '0.7560706402'), (4, '0', '0.7560706402'), (5, '0', '0.7464086773'), (7, '0', '0.7265145832'), (9, '0', '0.7560706402')].

## Interpretation

- Seed 42 is strong because WCA weighting repeatedly reduces imbalance without leaving a persistent weak client at the end; final gap is much smaller than FedProx and the weakest client is improved.
- Seed 123 is unstable because WCA+FedProx sharply improves client 2 but sacrifices client 0 relative to FedProx; final worst-client Dice drops to 0.4017 although the gap remains lower than FedProx seed 42.
- Seed 2025 is unstable because client 0 remains the repeated weak client for many rounds under WCA+FedProx; WCA gives client 0 very high aggregation weight for several rounds, but recovery is incomplete and final gap becomes worse than FedProx.
- The main issue is not a logging or summary aggregation bug: final summaries match metrics.csv, all runs use BUSI hard_noniid, image_size=128, seed-specific logs, and no NaN/inf is observed.
- Split distribution contributes to instability: hard_noniid intentionally creates strong client imbalance, and seed-specific train/val/test sampling changes each client's malignant ratio and mask-area range. However, split distribution alone does not explain the failure, because the WCA weighting response can over-focus one weak client and move the weakness to another client.
- The current evidence points more to WCA weighting sensitivity under BUSI hard_noniid than to FedProx mu alone. The FedProx term is identical between FedProx and WCA+FedProx; the unstable behavior appears after adding WCA aggregation weights.

## Recommendation

- Do not run alpha/mu diagnostic experiments yet; the read-only evidence is sufficient to identify WCA weighting sensitivity as the immediate issue.
- Downgrade `wca_comfedseg_prox` for BUSI hard_noniid from scoped candidate to diagnostic appendix candidate unless a revised, more stable WCA weighting rule is designed.
- Do not use BUSI hard_noniid WCA+FedProx as a main new-method claim.
- Keep Kvasir WCA+BN as the stronger hard_noniid scoped candidate from Phase 5K.
- Next broad validation should not expand BUSI WCA+FedProx. Prefer moderate_noniid scoped multi-seed for WCA-Comm, or a focused BUSI WCA weighting redesign if BUSI hard_noniid remains central.

## Constraints

- No new diagnostic training runs were launched.
- No Kvasir experiments were run.
- No moderate_noniid experiments were run.
- No failed variants were run.
- `wca_comfedseg_prox_comm_cons` was not run.
- image_size=256 was not used.
- BUSI normal was not used.
- No large model / SAM / MedSAM / pretrained backbone was added.
- Existing Phase 5A-5K results were not overwritten.
