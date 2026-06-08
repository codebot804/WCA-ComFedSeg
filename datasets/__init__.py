"""Dataset loaders and client split utilities for WCA-ComFedSeg."""

from datasets.busi import BUSISegmentationDataset, build_busi_client_loaders, discover_busi_samples
from datasets.bus_uclm import BUSUCLMSegmentationDataset, build_bus_uclm_client_loaders, discover_bus_uclm_samples
from datasets.kvasir_seg import KvasirSegmentationDataset, build_kvasir_client_loaders, discover_kvasir_samples
from datasets.synthetic_segmentation import SyntheticSegmentationDataset, build_synthetic_client_loaders

__all__ = [
    "BUSISegmentationDataset",
    "BUSUCLMSegmentationDataset",
    "KvasirSegmentationDataset",
    "SyntheticSegmentationDataset",
    "build_busi_client_loaders",
    "build_bus_uclm_client_loaders",
    "build_kvasir_client_loaders",
    "build_synthetic_client_loaders",
    "discover_busi_samples",
    "discover_bus_uclm_samples",
    "discover_kvasir_samples",
]
