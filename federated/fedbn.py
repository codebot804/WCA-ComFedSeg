"""FedBN helpers.

FedBN keeps BatchNorm parameters and running statistics local to each client and
aggregates only non-BatchNorm state.
"""

from __future__ import annotations

import torch
from torch import nn


def batchnorm_state_keys(model: nn.Module) -> set[str]:
    """Return state_dict keys belonging to BatchNorm modules.

    FedBN excludes BatchNorm affine parameters (`weight`, `bias`) or
    running statistics (`running_mean`, `running_var`, `num_batches_tracked`).
    We identify those keys by walking every BatchNorm module and expanding its
    `state_dict` names into full model-level state keys.
    """

    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for name in module.state_dict().keys():
                keys.add(prefix + name)
    return keys


def partition_fedbn_state_keys(model: nn.Module) -> tuple[list[str], list[str]]:
    """Split model state keys into aggregated non-BN keys and local BN keys."""

    bn_keys = batchnorm_state_keys(model)
    all_keys = list(model.state_dict().keys())
    aggregated_keys = [key for key in all_keys if key not in bn_keys]
    local_bn_keys = [key for key in all_keys if key in bn_keys]
    return aggregated_keys, local_bn_keys


def validate_fedbn_key_partition(model: nn.Module) -> dict[str, object]:
    """Return a compact FedBN sanity summary.

    This is intended for debug output or tests. Normal training does not print
    these lists unless the caller enables a debug flag.
    """

    aggregated_keys, local_bn_keys = partition_fedbn_state_keys(model)
    has_batchnorm = len(local_bn_keys) > 0
    overlap = sorted(set(aggregated_keys).intersection(local_bn_keys))
    if overlap:
        raise RuntimeError(f"FedBN key partition overlap detected: {overlap}")
    return {
        "has_batchnorm": has_batchnorm,
        "aggregated_key_count": len(aggregated_keys),
        "local_batchnorm_key_count": len(local_bn_keys),
        "aggregated_key_examples": aggregated_keys[:8],
        "local_batchnorm_key_examples": local_bn_keys[:8],
    }


def merge_global_with_local_batchnorm(
    global_state: dict[str, torch.Tensor],
    local_state: dict[str, torch.Tensor] | None,
    bn_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Use global non-BN parameters and local BN state when available.

    During FedBN evaluation/training, global aggregated parameters are combined
    with each client's own BatchNorm state. This preserves client-specific BN
    statistics across federated rounds.
    """

    merged = {key: value.detach().cpu().clone() for key, value in global_state.items()}
    if local_state is None:
        return merged
    for key in bn_keys:
        if key in local_state:
            merged[key] = local_state[key].detach().cpu().clone()
    return merged
