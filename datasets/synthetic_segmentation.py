"""Synthetic medical-style binary segmentation dataset for pipeline checks.

The generator creates simple foreground objects with configurable client-level
profile differences. It is useful for fast sanity checks before running public
medical datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "val", "test"]
SyntheticSplitMode = Literal["iid", "moderate_noniid", "hard_noniid", "extreme_noniid"]


@dataclass(frozen=True)
class SyntheticClientProfile:
    """Distribution settings for one simulated client."""

    profile_name: str
    noise_std: float
    background_mean: float
    background_std: float
    foreground_intensity_min: float
    foreground_intensity_max: float
    radius_x_min: float
    radius_x_max: float
    radius_y_min: float
    radius_y_max: float
    gradient_x: float
    gradient_y: float
    irregularity: float
    sample_scale: float = 1.0

    @property
    def contrast_mean(self) -> float:
        return (self.foreground_intensity_min + self.foreground_intensity_max) / 2.0

    @property
    def object_area_fraction_estimate(self) -> float:
        mean_rx = (self.radius_x_min + self.radius_x_max) / 2.0
        mean_ry = (self.radius_y_min + self.radius_y_max) / 2.0
        return float(np.pi * mean_rx * mean_ry)


@dataclass(frozen=True)
class SyntheticClientConfig:
    client_id: int
    split: SplitName
    split_mode: SyntheticSplitMode
    num_samples: int
    image_size: int
    seed: int
    profile: SyntheticClientProfile


class SyntheticSegmentationDataset(Dataset):
    """Small deterministic image-mask dataset with medical-style blobs."""

    def __init__(self, config: SyntheticClientConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._make_sample(index)
        return torch.from_numpy(image), torch.from_numpy(mask)

    def _make_sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.image_size
        split_offset = {"train": 0, "val": 10_000, "test": 20_000}[self.config.split]
        sample_seed = self.config.seed + self.config.client_id * 1_000 + split_offset + index
        rng = np.random.default_rng(sample_seed)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        center_x = rng.uniform(0.35 * size, 0.65 * size)
        center_y = rng.uniform(0.35 * size, 0.65 * size)
        profile = self.config.profile
        radius_x = rng.uniform(profile.radius_x_min * size, profile.radius_x_max * size)
        radius_y = rng.uniform(profile.radius_y_min * size, profile.radius_y_max * size)

        ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        if profile.irregularity > 0:
            angle = np.arctan2(yy - center_y, xx - center_x)
            boundary = 1.0 + profile.irregularity * np.sin(3.0 * angle + rng.uniform(0, 2 * np.pi))
            boundary += 0.5 * profile.irregularity * np.sin(5.0 * angle + rng.uniform(0, 2 * np.pi))
            mask = (ellipse <= boundary).astype(np.float32)
        else:
            mask = (ellipse <= 1.0).astype(np.float32)

        background = rng.normal(loc=profile.background_mean, scale=profile.background_std, size=(size, size))
        gradient = profile.gradient_x * (xx / max(size - 1, 1)) + profile.gradient_y * (yy / max(size - 1, 1))
        foreground = mask * rng.uniform(profile.foreground_intensity_min, profile.foreground_intensity_max)
        image = background + gradient + foreground

        image += rng.normal(loc=0.0, scale=profile.noise_std, size=(size, size))
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        return image[None, ...], mask[None, ...]


def get_synthetic_client_profiles(split_mode: SyntheticSplitMode, num_clients: int) -> list[SyntheticClientProfile]:
    """Return deterministic client distribution profiles for a split mode."""

    if split_mode == "iid":
        base = SyntheticClientProfile(
            profile_name="iid_shared_profile",
            noise_std=0.025,
            background_mean=0.22,
            background_std=0.030,
            foreground_intensity_min=0.50,
            foreground_intensity_max=0.62,
            radius_x_min=0.12,
            radius_x_max=0.19,
            radius_y_min=0.12,
            radius_y_max=0.20,
            gradient_x=0.06,
            gradient_y=0.04,
            irregularity=0.00,
            sample_scale=1.0,
        )
        return [base for _ in range(num_clients)]

    if split_mode == "moderate_noniid":
        profiles = [
            SyntheticClientProfile("moderate_client0_low_noise_medium_objects", 0.018, 0.21, 0.026, 0.52, 0.64, 0.13, 0.20, 0.13, 0.21, 0.06, 0.04, 0.00, 1.00),
            SyntheticClientProfile("moderate_client1_medium_noise_small_objects", 0.035, 0.23, 0.034, 0.45, 0.56, 0.09, 0.15, 0.09, 0.16, 0.05, 0.04, 0.04, 0.85),
            SyntheticClientProfile("moderate_client2_higher_noise_large_variable_objects", 0.050, 0.25, 0.040, 0.48, 0.61, 0.14, 0.24, 0.12, 0.25, 0.08, 0.05, 0.08, 1.15),
        ]
        return [profiles[min(client_id, len(profiles) - 1)] for client_id in range(num_clients)]

    if split_mode == "hard_noniid":
        profiles = [
            SyntheticClientProfile("hard_client0_easy_clear_medium_objects", 0.020, 0.21, 0.028, 0.54, 0.66, 0.13, 0.21, 0.13, 0.22, 0.05, 0.04, 0.00, 1.05),
            SyntheticClientProfile("hard_client1_medium_noise_reduced_contrast", 0.070, 0.25, 0.045, 0.28, 0.40, 0.09, 0.16, 0.09, 0.17, 0.08, 0.06, 0.10, 0.85),
            SyntheticClientProfile("hard_client2_hard_learnable_small_low_contrast", 0.120, 0.29, 0.060, 0.13, 0.23, 0.055, 0.105, 0.055, 0.115, 0.10, 0.08, 0.20, 0.65),
        ]
        return [profiles[min(client_id, len(profiles) - 1)] for client_id in range(num_clients)]

    if split_mode == "extreme_noniid":
        profiles = [
            SyntheticClientProfile("extreme_client0_easy_clear_large_objects", 0.012, 0.20, 0.022, 0.60, 0.72, 0.16, 0.25, 0.16, 0.26, 0.04, 0.03, 0.00, 1.20),
            SyntheticClientProfile("extreme_client1_hard_noisy_small_objects", 0.090, 0.26, 0.050, 0.20, 0.30, 0.05, 0.11, 0.05, 0.12, 0.08, 0.06, 0.12, 1.00),
            SyntheticClientProfile("extreme_client2_hardest_low_contrast_irregular", 0.140, 0.30, 0.065, 0.05, 0.10, 0.035, 0.10, 0.035, 0.12, 0.10, 0.08, 0.25, 0.60),
        ]
        return [profiles[min(client_id, len(profiles) - 1)] for client_id in range(num_clients)]

    raise ValueError(f"Unsupported synthetic split mode: {split_mode}")


def _scaled_sample_count(base_count: int, sample_scale: float) -> int:
    return max(2, int(round(base_count * sample_scale)))


def build_synthetic_split_metadata(
    split_mode: SyntheticSplitMode,
    num_clients: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> dict:
    profiles = get_synthetic_client_profiles(split_mode, num_clients)
    client_metadata = []
    for client_id, profile in enumerate(profiles):
        sample_counts = {
            "train": _scaled_sample_count(train_samples, profile.sample_scale),
            "val": _scaled_sample_count(val_samples, profile.sample_scale),
            "test": _scaled_sample_count(test_samples, profile.sample_scale),
        }
        client_metadata.append(
            {
                "client_id": client_id,
                "sample_counts": sample_counts,
                "estimated_object_area_fraction": profile.object_area_fraction_estimate,
                "noise_std": profile.noise_std,
                "contrast_mean": profile.contrast_mean,
                "profile": asdict(profile),
            }
        )
    return {
        "split_mode": split_mode,
        "description": {
            "iid": "All clients share the same synthetic image quality, contrast, object size, and shape distribution.",
            "moderate_noniid": "Clients have moderate differences in noise, contrast, object size, and object variability.",
            "hard_noniid": "Clients have stronger but still learnable differences in noise, contrast, object size, and sample count.",
            "extreme_noniid": "Clients have clearly different difficulty levels, including low contrast, high noise, small objects, and irregular shapes.",
        }[split_mode],
        "clients": client_metadata,
    }


def build_synthetic_client_loaders(
    num_clients: int,
    batch_size: int,
    split_mode: SyntheticSplitMode = "iid",
    image_size: int = 64,
    train_samples: int = 24,
    val_samples: int = 8,
    test_samples: int = 8,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[Dict[int, dict[str, DataLoader]], dict]:
    """Create train/val/test loaders for each simulated client."""

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    loaders: Dict[int, dict[str, DataLoader]] = {}
    split_metadata = build_synthetic_split_metadata(split_mode, num_clients, train_samples, val_samples, test_samples)
    profiles = get_synthetic_client_profiles(split_mode, num_clients)

    for client_id, profile in enumerate(profiles):
        client_loaders: dict[str, DataLoader] = {}
        sample_counts = split_metadata["clients"][client_id]["sample_counts"]
        for split, num_samples in sample_counts.items():
            dataset = SyntheticSegmentationDataset(
                SyntheticClientConfig(
                    client_id=client_id,
                    split=split,  # type: ignore[arg-type]
                    split_mode=split_mode,
                    num_samples=num_samples,
                    image_size=image_size,
                    seed=seed,
                    profile=profile,
                )
            )
            generator = torch.Generator().manual_seed(seed + client_id * 31)
            client_loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                generator=generator if split == "train" else None,
            )
        loaders[client_id] = client_loaders

    return loaders, split_metadata
