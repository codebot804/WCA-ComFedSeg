"""Sanity checks for personalized BatchNorm support."""

import torch

from federated.fedbn import partition_fedbn_state_keys
from methods.wca_comfedseg import wca_aggregate
from models.unet import build_unet2d


def test_batchnorm_unet_has_local_and_aggregated_keys() -> None:
    model = build_unet2d(base_channels=4, norm="batch")
    aggregated_keys, local_bn_keys = partition_fedbn_state_keys(model)

    assert aggregated_keys
    assert local_bn_keys
    assert set(aggregated_keys).isdisjoint(local_bn_keys)


def test_wca_aggregate_can_exclude_batchnorm_keys() -> None:
    states = [
        {"weight": torch.tensor([1.0]), "bn.running_mean": torch.tensor([10.0])},
        {"weight": torch.tensor([3.0]), "bn.running_mean": torch.tensor([30.0])},
    ]

    aggregated = wca_aggregate(states, [0.25, 0.75], exclude_keys={"bn.running_mean"})

    assert torch.isclose(aggregated["weight"], torch.tensor([2.5])).all()
    assert "bn.running_mean" not in aggregated
