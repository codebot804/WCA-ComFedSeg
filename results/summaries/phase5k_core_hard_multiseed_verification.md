# Phase 5K Core hard_noniid Multi-Seed Verification

Phase 5K is the first staged scoped multi-seed validation subset. It is not a final paper claim.

## Run Scope

- BUSI hard_noniid: `fedprox`, `wca_comfedseg_prox`.
- Kvasir-SEG hard_noniid: `fedbn`, `wca_comfedseg_bn`.
- Seeds: 42, 123, 2025.
- Fixed settings: clients=3, rounds=10, local_epochs=2, batch_size=4, image_size=128.

## Completion Checks

- Expected run count: 12
- Completed unique run count: 12
- Missing runs: none
- Duplicate run groups: 0
- Disallowed/extra runs under phase5k log root: none
- Failed runs recorded: none
- NaN/inf: no
- Only approved methods were run: yes
- No failed variants: yes
- image_size values observed: [128]
- image_size=128 only: yes
- image_size=256 avoided: yes
- BUSI normal avoided: yes
- No large model / SAM / MedSAM / pretrained backbone was added: yes
- Logs located under phase5k_core_hard_multiseed: yes
- Each method has exactly 3 seeds: yes

## BUSI Interpretation

- WCA+FedProx worst-client Dice mean higher than FedProx: False.
- WCA+FedProx best-worst gap mean lower than FedProx: True.
- WCA+FedProx average Dice mean higher than FedProx: False.

## Kvasir Interpretation

- WCA+BN average Dice mean higher than FedBN: True.
- WCA+BN worst-client Dice mean higher than FedBN: True.
- WCA+BN best-worst gap mean lower than FedBN: True.

## Recommendation

- BUSI: `wca_comfedseg_prox` is not stable enough yet; inspect hard_noniid behavior before expanding.
- Kvasir: keep `wca_comfedseg_bn` as a scoped hard_noniid candidate.
- Next step: adjust hard_noniid candidates before broadening validation.
