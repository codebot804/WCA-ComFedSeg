# Phase 5M Moderate Non-IID Multi-Seed Verification

Phase 5M is scoped real-dataset validation for moderate_noniid. It is not a final SCI claim.

## Run Scope

- Datasets: `busi`, `kvasir_seg`.
- Split: `moderate_noniid` only.
- Methods: `fedavg`, `fedprox`, `fedbn`, `wca_comfedseg_comm`.
- Seeds: 42, 123, 2025.
- Fixed settings: clients=3, rounds=10, local_epochs=2, batch_size=4, image_size=128.

## Completion Checks

- Expected run count: 24
- Completed unique run count: 24
- Missing runs: none
- Duplicate run groups: 0
- Failed or incomplete run directories: none
- Disallowed/extra runs under phase5m log root: none
- NaN/inf: no
- Only approved methods were run: yes
- No failed variants: yes
- No hard_noniid: yes
- image_size values observed: [128]
- image_size=128 only: yes
- image_size=256 avoided: yes
- BUSI normal avoided: yes
- No large model / SAM / MedSAM / pretrained backbone was added: yes
- Logs located under phase5m_moderate_multiseed: yes
- Each method has exactly 3 seeds per dataset: yes
- WCA-Comm communication logs exist: yes
- WCA-Comm communication reduction exists: yes

## BUSI Moderate Non-IID Interpretation

- WCA-Comm communication reduction mean: 30.00%.
- Average Dice: WCA-Comm=0.5275; best=wca_comfedseg_comm (0.5275); competitive=True.
- Worst-client Dice: WCA-Comm=0.4974; best=wca_comfedseg_comm (0.4974); competitive=True.
- Best-worst gap: WCA-Comm=0.0580; best=wca_comfedseg_comm (0.0580); competitive=True.

## Kvasir-SEG Moderate Non-IID Interpretation

- WCA-Comm communication reduction mean: 30.00%.
- Average Dice: WCA-Comm=0.4474; best=fedbn (0.4987); competitive=False.
- Worst-client Dice: WCA-Comm=0.3123; best=fedbn (0.3948); competitive=False.
- Best-worst gap: WCA-Comm=0.2289; best=fedbn (0.2131); competitive=True.

## Cross-Dataset Decision

- WCA-Comm stable enough to remain the moderate_noniid communication-efficient candidate: False.
- Communication-efficiency trade-off acceptable: True.
- Proceed directly to final table/figure integration: False.
- Additional diagnosis needed before final integration: True.

## Recommendation

- Keep `wca_comfedseg_comm` as a communication-saving candidate, but present the Dice/fairness trade-off honestly and run targeted diagnosis before final claims.
- Treat Phase 5M as scoped validation only, not as final SCI evidence.
