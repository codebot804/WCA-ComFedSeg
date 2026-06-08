"""Kvasir-SEG polyp segmentation dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

SplitName = Literal["train", "val", "test"]
RealSplitMode = Literal["iid", "moderate_noniid", "hard_noniid"]

KVASIR_ROOT = Path("data/raw/Kvasir-SEG")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class KvasirSample:
    image_path: Path
    mask_path: Path
    sample_id: str
    mask_area_fraction: float
    image_mean: float
    image_std: float


class KvasirSegmentationDataset(Dataset):
    """Kvasir-SEG image-mask dataset with binary masks."""

    def __init__(self, samples: list[KvasirSample], image_size: int = 128) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = _load_image(sample.image_path, self.image_size)
        mask = _load_mask(sample.mask_path, self.image_size)
        return torch.from_numpy(image), torch.from_numpy(mask)


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("L")
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, ...]


def _load_mask(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as handle:
        mask = handle.convert("L")
        mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        array = (np.asarray(mask) > 0).astype(np.float32)
    return array[None, ...]


def _mask_area_fraction(path: Path, image_size: int = 128) -> float:
    mask = _load_mask(path, image_size)
    return float(mask.mean())


def _image_quality_stats(path: Path, image_size: int = 128) -> tuple[float, float]:
    image = _load_image(path, image_size)
    return float(image.mean()), float(image.std())


def discover_kvasir_samples(root: str | Path = KVASIR_ROOT) -> tuple[list[KvasirSample], dict]:
    """Pair Kvasir images and masks by filename stem."""

    root = Path(root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    warnings: list[str] = []
    folder_status = {
        "root_exists": root.exists(),
        "images_exists": image_dir.exists(),
        "masks_exists": mask_dir.exists(),
    }
    if not root.exists():
        warnings.append(f"Kvasir-SEG root missing: {root}")
    if root.exists() and not image_dir.exists():
        warnings.append(f"Kvasir-SEG images folder missing: {image_dir}")
    if root.exists() and not mask_dir.exists():
        warnings.append(f"Kvasir-SEG masks folder missing: {mask_dir}")

    image_files = _image_files(image_dir)
    mask_files = _image_files(mask_dir)
    images_by_stem = {path.stem: path for path in image_files}
    masks_by_stem = {path.stem: path for path in mask_files}
    paired_stems = sorted(set(images_by_stem).intersection(masks_by_stem))
    unmatched_images = sorted(set(images_by_stem).difference(masks_by_stem))
    unmatched_masks = sorted(set(masks_by_stem).difference(images_by_stem))

    samples = []
    for stem in paired_stems:
        image_mean, image_std = _image_quality_stats(images_by_stem[stem])
        samples.append(
            KvasirSample(
                image_path=images_by_stem[stem],
                mask_path=masks_by_stem[stem],
                sample_id=stem,
                mask_area_fraction=_mask_area_fraction(masks_by_stem[stem]),
                image_mean=image_mean,
                image_std=image_std,
            )
        )
    metadata = {
        "dataset": "kvasir_seg",
        "root": str(root),
        "folder_status": folder_status,
        "raw_image_count": len(image_files),
        "raw_mask_count": len(mask_files),
        "valid_pair_count": len(samples),
        "unmatched_image_count": len(unmatched_images),
        "unmatched_mask_count": len(unmatched_masks),
        "unmatched_images": unmatched_images[:20],
        "unmatched_masks": unmatched_masks[:20],
        "warnings": warnings,
    }
    return samples, metadata


def _split_counts(total: int) -> tuple[int, int, int]:
    train_count = max(1, int(round(total * 0.70)))
    val_count = max(1, int(round(total * 0.15)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, total - val_count - test_count)
    return train_count, val_count, test_count


def split_kvasir_samples(samples: list[KvasirSample], seed: int = 42) -> dict[str, list[KvasirSample]]:
    """Create deterministic train/val/test splits."""

    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    train_count, val_count, _ = _split_counts(len(indices))
    split_indices = {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count :],
    }
    return {
        split_name: sorted((samples[int(index)] for index in selected), key=lambda sample: sample.sample_id)
        for split_name, selected in split_indices.items()
    }


def _round_robin_partition(samples: list[KvasirSample], num_clients: int) -> list[list[int]]:
    partitions = [[] for _ in range(num_clients)]
    for index, _sample in enumerate(samples):
        partitions[index % num_clients].append(index)
    return partitions


def _moderate_noniid_partition(samples: list[KvasirSample], num_clients: int) -> list[list[int]]:
    if num_clients != 3:
        return _round_robin_partition(samples, num_clients)

    sorted_indices = sorted(range(len(samples)), key=lambda index: samples[index].mask_area_fraction)
    groups = np.array_split(np.asarray(sorted_indices, dtype=int), 3)
    partitions = [[] for _ in range(num_clients)]
    pattern = {
        0: [0.60, 0.30, 0.10],
        1: [0.20, 0.60, 0.20],
        2: [0.10, 0.30, 0.60],
    }
    for group_id, group in enumerate(groups):
        indices = [int(index) for index in group.tolist()]
        cursor = 0
        for client_id, ratio in enumerate(pattern[group_id]):
            if client_id == num_clients - 1:
                selected = indices[cursor:]
            else:
                take = int(round(len(indices) * ratio))
                selected = indices[cursor : cursor + take]
                cursor += take
            partitions[client_id].extend(selected)
    for partition in partitions:
        partition.sort()
    return partitions


def _hard_noniid_partition(samples: list[KvasirSample], num_clients: int) -> list[list[int]]:
    if num_clients != 3:
        return _round_robin_partition(samples, num_clients)

    sorted_indices = sorted(range(len(samples)), key=lambda index: samples[index].mask_area_fraction)
    groups = np.array_split(np.asarray(sorted_indices, dtype=int), 3)
    partitions = [[] for _ in range(num_clients)]
    pattern = {
        0: [0.80, 0.16, 0.04],
        1: [0.50, 0.35, 0.15],
        2: [0.14, 0.28, 0.58],
    }
    for group_id, group in enumerate(groups):
        indices = [int(index) for index in group.tolist()]
        cursor = 0
        for client_id, ratio in enumerate(pattern[group_id]):
            if client_id == num_clients - 1:
                selected = indices[cursor:]
            else:
                take = int(round(len(indices) * ratio))
                selected = indices[cursor : cursor + take]
                cursor += take
            partitions[client_id].extend(selected)
    for partition in partitions:
        partition.sort()
    return partitions


def _client_metadata(split_samples: dict[str, list[KvasirSample]], partitions: dict[str, list[list[int]]]) -> list[dict]:
    clients = []
    for client_id in range(len(next(iter(partitions.values())))):
        sample_counts = {}
        area_stats = {}
        for split_name, samples in split_samples.items():
            indices = partitions[split_name][client_id]
            areas = [samples[index].mask_area_fraction for index in indices]
            sample_counts[split_name] = len(indices)
            area_stats[split_name] = {
                "mean_mask_area_fraction": float(np.mean(areas)) if areas else 0.0,
                "min_mask_area_fraction": float(np.min(areas)) if areas else 0.0,
                "max_mask_area_fraction": float(np.max(areas)) if areas else 0.0,
            }
        clients.append(
            {
                "client_id": client_id,
                "sample_counts": sample_counts,
                "mask_area_stats": area_stats,
            }
        )
    return clients


def build_kvasir_client_loaders(
    num_clients: int,
    batch_size: int,
    split_mode: RealSplitMode = "iid",
    image_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    root: str | Path = KVASIR_ROOT,
) -> tuple[dict[int, dict[str, DataLoader]], dict]:
    """Create Kvasir-SEG train/val/test loaders for simulated clients."""

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if split_mode not in {"iid", "moderate_noniid", "hard_noniid"}:
        raise ValueError("Kvasir-SEG supports iid, moderate_noniid, and hard_noniid real-data splits.")

    samples, dataset_metadata = discover_kvasir_samples(root)
    if not samples:
        raise RuntimeError("No valid Kvasir-SEG image-mask pairs found.")

    split_samples = split_kvasir_samples(samples, seed=seed)
    partitions: dict[str, list[list[int]]] = {}
    for split_name, samples_for_split in split_samples.items():
        if split_mode == "iid":
            partitions[split_name] = _round_robin_partition(samples_for_split, num_clients)
        elif split_mode == "moderate_noniid":
            partitions[split_name] = _moderate_noniid_partition(samples_for_split, num_clients)
        else:
            partitions[split_name] = _hard_noniid_partition(samples_for_split, num_clients)

    loaders: dict[int, dict[str, DataLoader]] = {}
    for client_id in range(num_clients):
        client_loaders: dict[str, DataLoader] = {}
        for split_name, samples_for_split in split_samples.items():
            dataset = KvasirSegmentationDataset(samples_for_split, image_size=image_size)
            subset = Subset(dataset, partitions[split_name][client_id])
            generator = torch.Generator().manual_seed(seed + client_id * 31)
            client_loaders[split_name] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                generator=generator if split_name == "train" else None,
            )
        loaders[client_id] = client_loaders

    sample_ids_by_split = {
        split_name: [sample.sample_id for sample in samples_for_split]
        for split_name, samples_for_split in split_samples.items()
    }
    overlaps = {
        "train_val": sorted(set(sample_ids_by_split["train"]).intersection(sample_ids_by_split["val"])),
        "train_test": sorted(set(sample_ids_by_split["train"]).intersection(sample_ids_by_split["test"])),
        "val_test": sorted(set(sample_ids_by_split["val"]).intersection(sample_ids_by_split["test"])),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Kvasir-SEG split leakage detected: {overlaps}")

    split_metadata = {
        "split_mode": split_mode,
        "description": {
            "iid": "Kvasir-SEG samples are assigned round-robin across clients after train/val/test splitting.",
            "moderate_noniid": "Kvasir-SEG clients receive different mask-size distributions while preserving disjoint samples.",
            "hard_noniid": "Kvasir-SEG clients receive stronger mask-size and sample-count imbalance while preserving disjoint samples.",
        }[split_mode],
        "dataset": dataset_metadata,
        "clients": _client_metadata(split_samples, partitions),
        "leakage_check": {"overlaps": overlaps, "passed": True},
    }
    return loaders, split_metadata
