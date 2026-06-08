"""Training methods and federated algorithms for WCA-ComFedSeg."""

from methods.centralized import run_centralized_training
from methods.local import run_local_training

__all__ = ["run_centralized_training", "run_local_training"]
