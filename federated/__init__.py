"""Federated learning infrastructure for WCA-ComFedSeg."""

from federated.client import FederatedClient
from federated.fedavg import fedavg_aggregate
from federated.server import FederatedServer

__all__ = ["FederatedClient", "FederatedServer", "fedavg_aggregate"]

