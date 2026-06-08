# Algorithm 1. WCA-Comm Federated Training Procedure

This pseudocode describes the implemented `wca_comfedseg_comm` procedure at paper level. It is an algorithmic description only; it does not claim a convergence proof.

**Input:** client set C = {1, ..., K}; initial global model theta^0; number of rounds T; local epochs E; learning rate eta; WCA mixing parameter alpha in [0, 1]; client fraction rho; minimum selected clients m; validation split per client; test split per client.

**Output:** final global model theta^T; client-level test Dice/IoU; per-round WCA weights; per-round communication log; total uploaded parameters/MB and reduction against full participation.

```text
1  Initialize global model theta^0.
2  Initialize previous validation Dice records D_prev = empty.
3  Compute one-client upload size P_full from the full model state.
4  for round t = 1, ..., T do
5      if t = 1 or D_prev is empty then
6          Select all clients for upload.
7          Mark selection reason as all_clients_round1.
8      else
9          For every client k, compute deficit delta_k = max(mean(D_prev) - D_prev[k], 0).
10         Identify the previous weakest valid client k_worst = argmin_k D_prev[k].
11         Select N_t = min(K, max(1, floor(rho K), m)) clients.
12         Always include k_worst.
13         Fill remaining slots by descending deficit delta_k.
14         If all deficits are zero, fill remaining slots by client data size.
15         Mark non-selected clients as skipped.
16     end if
17
18     Send theta^{t-1} to selected clients.
19     for each selected client k in S_t do
20         Train locally for E epochs from theta^{t-1} on client k's training data.
21         Return local model theta_k^t and sample count n_k.
22         Add parameter count of theta_k^t to the round upload total.
23     end for
24
25     if D_prev is empty then
26         Use data-size weights a_k = n_k / sum_{j in S_t} n_j.
27     else
28         For selected clients, compute delta_k = max(mean(D_prev over selected valid clients) - D_prev[k], 0).
29         If sum(delta_k) > 0, set q_k = delta_k / sum_j delta_j; otherwise use q_k = 0.
30         Set data-size weights p_k = n_k / sum_{j in S_t} n_j.
31         If sum(delta_k) > 0, set a_k = (1 - alpha) p_k + alpha q_k; otherwise set a_k = p_k.
32         Normalize a_k over selected clients.
33     end if
34
35     Aggregate selected models theta^t = sum_{k in S_t} a_k theta_k^t.
36     Record WCA weights for selected clients; record zero aggregation weight for skipped clients.
37     Evaluate theta^t on every client's validation split.
38     Update D_prev with the current validation Dice for all clients.
39     Log per-client selection reason, previous Dice, deficit, uploaded parameters, uploaded MB, and cumulative communication.
40 end for
41
42 Evaluate the final model on each client's test split.
43 Report average Dice/IoU, worst-client Dice, client Dice standard deviation, best-worst gap, uploaded parameters/MB, and communication reduction.
```

**Implementation note:** the code computes deficits relative to the average valid validation Dice, not relative to the best client. Any manuscript formula should therefore use `max(mean Dice - client Dice, 0)` when describing the implemented method.
