"""BUSI breast ultrasound dataset loader.

The current experiments use benign and malignant image-mask pairs. Normal cases
are discovered but deferred because empty-mask handling requires a separate
evaluation protocol. Multiple masks for one BUSI image are merged by logical OR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

SplitName = Literal["train", "val", "test"]
RealSplitMode = Literal["iid", "moderate_noniid", "hard_noniid"]

BUSI_ROOT = Path("data/raw/BUSI/Dataset_BUSI_with_GT")
BUSI_CLASSES = ("benign", "malignant")


@dataclass(frozen=True)
class BUSISample:
    image_path: Path
    mask_paths: tuple[Path, ...]
    class_name: str
    sample_id: str
    mask_area_fraction: float


class BUSISegmentationDataset(Dataset):
    """BUSI image-mask dataset with merged binary masks."""

    def __init__(self, samples: list[BUSISample], image_size: int = 128) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = _load_image(sample.image_path, self.image_size)
        mask = _load_merged_mask(sample.mask_paths, self.image_size)
        return torch.from_numpy(image), torch.from_numpy(mask)


def _is_mask_file(path: Path) -> bool:
    return "_mask" in path.stem


def _base_image_stem(path: Path) -> str:
    return re.sub(r"_mask(?:_\d+)?$", "", path.stem)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("L")
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, ...]


def _load_merged_mask(paths: tuple[Path, ...], image_size: int) -> np.ndarray:
    merged = np.zeros((image_size, image_size), dtype=bool)
    for path in paths:
        with Image.open(path) as handle:
            mask = handle.convert("L")
            mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
            merged |= np.asarray(mask) > 0
    return merged.astype(np.float32)[None, ...]


def _mask_area_fraction(paths: tuple[Path, ...], image_size: int = 128) -> float:
    mask = _load_merged_mask(paths, image_size)
    return float(mask.mean())


def discover_busi_samples(root: str | Path = BUSI_ROOT) -> tuple[list[BUSISample], dict]:
    """Discover BUSI benign/malignant image-mask pairs and metadata."""

    root = Path(root)
    warnings: list[str] = []
    folder_status = {
        "root_exists": root.exists(),
        "benign_exists": (root / "benign").exists(),
        "malignant_exists": (root / "malignant").exists(),
        "normal_exists": (root / "normal").exists(),
        "normal_used": False,
    }
    samples: list[BUSISample] = []
    class_counts = {class_name: 0 for class_name in BUSI_CLASSES}
    raw_image_counts = {class_name: 0 for class_name in BUSI_CLASSES}
    raw_mask_counts = {class_name: 0 for class_name in BUSI_CLASSES}
    unmatched_images: list[str] = []
    orphan_masks: list[str] = []
    images_with_multiple_masks = 0
    merged_extra_mask_count = 0

    if not root.exists():
        warnings.append(f"BUSI root missing: {root}")
        return samples, {
            "dataset": "busi",
            "root": str(root),
            "folder_status": folder_status,
            "class_counts": class_counts,
            "raw_image_counts": raw_image_counts,
            "raw_mask_counts": raw_mask_counts,
            "valid_pair_count": 0,
            "images_with_multiple_masks": 0,
            "merged_mask_count": 0,
            "unmatched_images": unmatched_images,
            "orphan_masks": orphan_masks,
            "warnings": warnings,
        }

    for class_name in BUSI_CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            warnings.append(f"BUSI class folder missing: {class_dir}")
            continue

        png_files = sorted(class_dir.glob("*.png"))
        image_files = [path for path in png_files if not _is_mask_file(path)]
        mask_files = [path for path in png_files if _is_mask_file(path)]
        raw_image_counts[class_name] = len(image_files)
        raw_mask_counts[class_name] = len(mask_files)

        masks_by_stem: dict[str, list[Path]] = {}
        for mask_path in mask_files:
            masks_by_stem.setdefault(_base_image_stem(mask_path), []).append(mask_path)

        image_stems = {image_path.stem for image_path in image_files}
        for image_path in image_files:
            mask_paths = tuple(sorted(masks_by_stem.get(image_path.stem, [])))
            if not mask_paths:
                unmatched_images.append(str(image_path))
                continue
            if len(mask_paths) > 1:
                images_with_multiple_masks += 1
                merged_extra_mask_count += len(mask_paths) - 1
            sample_id = f"{class_name}/{image_path.stem}"
            samples.append(
                BUSISample(
                    image_path=image_path,
                    mask_paths=mask_paths,
                    class_name=class_name,
                    sample_id=sample_id,
                    mask_area_fraction=_mask_area_fraction(mask_paths),
                )
            )
            class_counts[class_name] += 1

        for mask_stem, paths in masks_by_stem.items():
            if mask_stem not in image_stems:
                orphan_masks.extend(str(path) for path in paths)

    metadata = {
        "dataset": "busi",
        "root": str(root),
        "folder_status": folder_status,
        "classes_used": list(BUSI_CLASSES),
        "classes_deferred": ["normal"],
        "class_counts": class_counts,
        "raw_image_counts": raw_image_counts,
        "raw_mask_counts": raw_mask_counts,
        "valid_pair_count": len(samples),
        "images_with_multiple_masks": images_with_multiple_masks,
        "merged_mask_count": merged_extra_mask_count,
        "unmatched_image_count": len(unmatched_images),
        "orphan_mask_count": len(orphan_masks),
        "unmatched_images": unmatched_images[:20],
        "orphan_masks": orphan_masks[:20],
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


def split_busi_samples(samples: list[BUSISample], seed: int = 42) -> dict[str, list[BUSISample]]:
    """Create deterministic stratified train/val/test splits."""

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name in BUSI_CLASSES:
        class_samples = [sample for sample in samples if sample.class_name == class_name]
        indices = np.arange(len(class_samples))
        rng.shuffle(indices)
        train_count, val_count, _ = _split_counts(len(indices))
        split_indices = {
            "train": indices[:train_count],
            "val": indices[train_count : train_count + val_count],
            "test": indices[train_count + val_count :],
        }
        for split_name, selected in split_indices.items():
            splits[split_name].extend(class_samples[int(index)] for index in selected)

    for split_samples in splits.values():
        split_samples.sort(key=lambda sample: sample.sample_id)
    return splits


def _round_robin_partition(samples: list[BUSISample], num_clients: int) -> list[list[int]]:
    partitions = [[] for _ in range(num_clients)]
    for index, _sample in enumerate(samples):
        partitions[index % num_clients].append(index)
    return partitions


def _moderate_noniid_partition(samples: list[BUSISample], num_clients: int) -> list[list[int]]:
    if num_clients != 3:
        return _round_robin_partition(samples, num_clients)

    by_class = {
        class_name: [index for index, sample in enumerate(samples) if sample.class_name == class_name]
        for class_name in BUSI_CLASSES
    }
    partitions = [[] for _ in range(num_clients)]
    for class_name, ratios in {
        "benign": [0.65, 0.25, 0.10],
        "malignant": [0.10, 0.25, 0.65],
    }.items():
        indices = by_class[class_name]
        cursor = 0
        for client_id, ratio in enumerate(ratios):
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


def _hard_noniid_partition(samples: list[BUSISample], num_clients: int) -> list[list[int]]:
    if num_clients != 3:
        return _round_robin_partition(samples, num_clients)

    by_class = {
        class_name: [index for index, sample in enumerate(samples) if sample.class_name == class_name]
        for class_name in BUSI_CLASSES
    }
    by_class["benign"].sort(key=lambda index: samples[index].mask_area_fraction)
    by_class["malignant"].sort(key=lambda index: samples[index].mask_area_fraction, reverse=True)

    partitions = [[] for _ in range(num_clients)]
    for class_name, ratios in {
        "benign": [0.72, 0.20, 0.08],
        "malignant": [0.08, 0.22, 0.70],
    }.items():
        indices = by_class[class_name]
        cursor = 0
        for client_id, ratio in enumerate(ratios):
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


def _build_client_metadata(split_samples: dict[str, list[BUSISample]], partitions: dict[str, list[list[int]]]) -> list[dict]:
    clients = []
    for client_id in range(len(next(iter(partitions.values())))):
        sample_counts = {}
        class_counts = {}
        for split_name, samples in split_samples.items():
            indices = partitions[split_name][client_id]
            sample_counts[split_name] = len(indices)
            class_counts[split_name] = {
                class_name: sum(1 for index in indices if samples[index].class_name == class_name)
                for class_name in BUSI_CLASSES
            }
        clients.append(
            {
                "client_id": client_id,
                "sample_counts": sample_counts,
                "class_counts": class_counts,
            }
        )
    return clients


def build_busi_client_loaders(
    num_clients: int,
    batch_size: int,
    split_mode: RealSplitMode = "iid",
    image_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    root: str | Path = BUSI_ROOT,
) -> tuple[dict[int, dict[str, DataLoader]], dict]:
    """Create BUSI train/val/test loaders for simulated clients."""

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if split_mode not in {"iid", "moderate_noniid", "hard_noniid"}:
        raise ValueError("BUSI supports iid, moderate_noniid, and hard_noniid real-data splits.")

    samples, dataset_metadata = discover_busi_samples(root)
    if not samples:
        raise RuntimeError("No valid BUSI benign/malignant image-mask pairs found.")

    split_samples = split_busi_samples(samples, seed=seed)
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
            dataset = BUSISegmentationDataset(samples_for_split, image_size=image_size)
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
        raise RuntimeError(f"BUSI split leakage detected: {overlaps}")

    split_metadata = {
        "split_mode": split_mode,
        "description": {
            "iid": "BUSI benign/malignant samples are stratified and assigned round-robin across clients.",
            "moderate_noniid": "BUSI clients receive different benign/malignant proportions while preserving disjoint samples.",
            "hard_noniid": "BUSI clients receive stronger benign/malignant and lesion-area imbalance while preserving disjoint samples.",
        }[split_mode],
        "dataset": dataset_metadata,
        "clients": _build_client_metadata(split_samples, partitions),
        "leakage_check": {"overlaps": overlaps, "passed": True},
    }
    return loaders, split_metadata
