# Table 2. BUSI moderate non-IID main comparison

Source CSV: `results/summaries/phase5m_moderate_multiseed_aggregate.csv`. Values are three-seed mean +/- SD for BUSI moderate non-IID only. This is the main positive evidence table.

| Method | Seeds | Average Dice, mean +/- SD | Average IoU, mean +/- SD | Worst-client Dice, mean +/- SD | Client Dice SD, mean +/- SD | Best-worst gap, mean +/- SD | Uploaded MB | Communication reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | 42\|123\|2025 | 0.5105 +/- 0.0186 | 0.3718 +/- 0.0159 | 0.4631 +/- 0.0255 | 0.0364 +/- 0.0225 | 0.0822 +/- 0.0528 | 3.374 | 0.0% |
| FedProx | 42\|123\|2025 | 0.5112 +/- 0.0232 | 0.3720 +/- 0.0206 | 0.4633 +/- 0.0205 | 0.0373 +/- 0.0166 | 0.0892 +/- 0.0396 | 3.374 | 0.0% |
| FedBN | 42\|123\|2025 | 0.5006 +/- 0.0173 | 0.3737 +/- 0.0089 | 0.4450 +/- 0.0384 | 0.0486 +/- 0.0105 | 0.1118 +/- 0.0255 | 3.374 | 0.0% |
| WCA-Comm | 42\|123\|2025 | 0.5275 +/- 0.0265 | 0.3902 +/- 0.0223 | 0.4974 +/- 0.0260 | 0.0257 +/- 0.0055 | 0.0580 +/- 0.0109 | 2.362 | 30.0% |
