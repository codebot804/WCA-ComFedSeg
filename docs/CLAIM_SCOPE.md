# Claim Scope

This project is a scoped empirical study, not a universal medical image segmentation benchmark.

Supported main claim:

- In BUSI benign/malignant moderate non-IID segmentation, WCA-Comm improves average Dice, worst-client Dice, and best-worst client gap over FedAvg, FedProx, and FedBN while reducing uploaded communication by 30%.

Boundary observations:

- BUS-UCLM moderate non-IID does not reproduce the BUSI WCA-Comm success pattern; FedBN is stronger in this setting.
- BUSI hard non-IID remains a failure boundary for the current WCA variants.
- Kvasir-SEG is a cross-modality boundary setting and should not be treated as breast ultrasound evidence.
- Normal cases in BUSI and BUS-UCLM are deferred.
