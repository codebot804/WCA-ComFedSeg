"""Data-size-weighted FedAvg aggregation helper."""

from __future__ import annotations

from collections import OrderedDict

import torch


def fedavg_aggregate(
    client_states: list[dict[str, torch.Tensor]],
    client_num_samples: list[int],
    exclude_keys: set[str] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Data-size-weighted FedAvg over model state dictionaries."""

    if not client_states:
        raise ValueError("client_states must not be empty.")
    if len(client_states) != len(client_num_samples):
        raise ValueError("client_states and client_num_samples must have the same length.")

    total_samples = float(sum(client_num_samples))
    if total_samples <= 0:
        raise ValueError("Total number of client samples must be positive.")

    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    exclude_keys = exclude_keys or set()
    for key in client_states[0].keys():
        if key in exclude_keys:
            continue
        weighted_sum = None
        for state, num_samples in zip(client_states, client_num_samples):
            tensor = state[key].detach().cpu()
            weighted_tensor = tensor * (float(num_samples) / total_samples)
            weighted_sum = weighted_tensor if weighted_sum is None else weighted_sum + weighted_tensor
        aggregated[key] = weighted_sum
    return aggregated
