# Formula-Code Alignment for WCA-ComFedSeg

This file records the formulas that are safe to use in the manuscript and the code locations that support them. It also flags implementation details that should not be simplified away in the final draft.

## WCA aggregation

Code source: `methods/wca_comfedseg.py` and `federated/server.py`.

Implemented deficit:

```text
delta_i = max(mean({d_j: d_j is valid}) - d_i, 0)
```

This is the critical correction. The implementation does not use `max_j d_j - d_i` or `d_best - d_i`. A formula using the best client should be described only as a discarded conceptual variant, not as the implemented WCA-Comm method.

Implemented weights:

```text
p_i = n_i / sum_j n_j
q_i = delta_i / sum_j delta_j, if sum_j delta_j > 0
w_i = (1 - alpha) p_i + alpha q_i
```

Fallback behavior:

```text
if validation Dice are unavailable, invalid, or all deficits sum to zero:
    w_i = p_i
alpha is clipped to [0, 1]
w_i is normalized before aggregation
```

Aggregation:

```text
theta = sum_i w_i theta_i
```

`wca_aggregate(client_states, aggregation_weights, exclude_keys=None)` applies the normalized weighted state average. When FedBN-style variants are used, BatchNorm keys may be excluded.

## Communication scheduler

Code source: `federated/communication.py` and `federated/server.py`.

Round 1 selects all clients. Later rounds compute the same average-Dice deficit and select a bounded number of clients:

```text
N_t = min(K, max(1, floor(client_fraction * K), min_selected_clients))
```

The scheduler always includes the previous weakest valid client. It fills remaining slots by descending performance deficit. If no positive deficit exists, it falls back to larger client data size. Logged reasons are `all_clients_round1`, `worst_client`, `priority_deficit`, `data_size_fallback`, and `skipped`.

In `FederatedServer.run`, `wca_comfedseg_comm` selects clients before local training. Only selected clients train and upload in that round. WCA weights are computed over the selected clients using the previous validation Dice. Skipped clients are recorded in the WCA-weight log with aggregation weight 0. After aggregation and validation, `previous_validation_dice` is updated from current client metrics.

## Communication accounting

Code source: `federated/communication.py` and `federated/server.py`.

```text
uploaded_parameters = sum(number of tensor elements in uploaded state)
uploaded_MB = uploaded_parameters * 4 / 1024^2
```

The unit is MiB under a float32-equivalent assumption. The server records full-participation cost as one full model upload per client per round. For selected clients, the row-level log uses the full model parameter count; skipped clients upload 0. Summaries store total uploaded MB and percentage reduction against full participation.

## Baselines

FedAvg is implemented in `federated/fedavg.py`, not in the placeholder `methods/fedavg.py`. The formula is data-size-weighted model averaging, with optional excluded keys for FedBN.

FedProx is implemented in `federated/client.py` inside `train_from_global`. When `proximal_mu > 0`, the local objective adds:

```text
0.5 * mu * sum_l ||theta_l - theta_global,l||_2^2
```

`methods/fedprox.py` is a placeholder marker, not the executable implementation.

FedBN is implemented in `federated/fedbn.py` and `federated/server.py`. BatchNorm affine parameters and running statistics are identified from BatchNorm modules, excluded from aggregation, and merged back from each client's local state during training/evaluation. `methods/fedbn.py` is a placeholder marker.

## Dataset loading and leakage controls

BUSI code source: `datasets/busi.py`. The loader uses benign and malignant samples only; normal cases are deferred. Multiple masks for the same image are merged with logical OR. Images are converted to grayscale and resized with bilinear interpolation; masks use nearest-neighbor resizing. Train/val/test leakage is checked by sample stem because no patient/study identifier is available in the local loader.

BUS-UCLM code source: `datasets/bus_uclm.py`. The loader uses Benign and Malignant samples only and defers Normal. Patient/study identity is derived from the stem prefix before the first underscore. Splits are built to avoid patient-prefix leakage. RGB masks treat green and red as foreground and black as background/normal. Images are normalized through RGB and converted to a single grayscale channel.

## Experiment entry point

`run_experiment.py` routes method names to centralized, local, FedAvg/FedProx/FedBN, WCA, WCA-Comm, and diagnostic variants. Method names also control normalisation choice, communication-log paths, WCA-weight log paths, and summary fields. Final manuscript claims should cite the exact CSV source file for each table/figure rather than citing method names alone.
