"""BUS-UCLM breast ultrasound lesion segmentation dataset loader.

The loader supports benign/malignant lesion segmentation and patient/study-prefix
split handling where identity information can be inferred from filenames. Normal
cases are discovered but deferred from the reported binary-lesion experiments.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


SplitName = Literal["train", "val", "test"]
RealSplitMode = Literal["iid", "moderate_noniid", "hard_noniid"]

BUS_UCLM_ROOT = Path("data/raw/BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM")
BUS_UCLM_INCLUDED_LABELS = ("Benign", "Malignant")
BUS_UCLM_DEFERRED_LABELS = ("Normal",)
BUS_UCLM_PHASE5P_CACHE = Path("data/cache/phase5p_bus_uclm_discovery_cache.json")
KNOWN_MASK_COLORS = {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
FOREGROUND_COLORS = {(0, 255, 0), (255, 0, 0)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class BUSUCLMSample:
    image_path: Path
    mask_path: Path
    label: str
    sample_id: str
    patient_id: str
    mask_area_fraction: float
    image_mode: str
    image_size: tuple[int, int]
    mask_mode: str
    mask_size: tuple[int, int]
    unknown_mask_colors: tuple[tuple[int, int, int], ...]


class BUSUCLMSegmentationDataset(Dataset):
    """BUS-UCLM image-mask dataset with binary lesion masks."""

    def __init__(self, samples: list[BUSUCLMSample], image_size: int = 128) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = _load_image(sample.image_path, self.image_size)
        mask = _load_binary_mask(sample.mask_path, self.image_size)
        return torch.from_numpy(image), torch.from_numpy(mask)


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)


def _sample_stem(filename: str | Path) -> str:
    return Path(filename).stem


def patient_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def _read_info_rows(info_path: Path) -> list[dict[str, str]]:
    if not info_path.exists():
        return []
    with info_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as handle:
        # BUS-UCLM contains RGB and RGBA images. Normalize through RGB first,
        # then return one channel to match the current BUSI/Kvasir U-Net input.
        image = handle.convert("RGB")
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray[None, ...].astype(np.float32)


def _mask_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def _foreground_mask(path: Path) -> np.ndarray:
    rgb = _mask_rgb_array(path)
    foreground = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for color in FOREGROUND_COLORS:
        foreground |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1).astype(np.uint8)
    return foreground


def _load_binary_mask(path: Path, image_size: int) -> np.ndarray:
    foreground = _foreground_mask(path) * 255
    mask = Image.fromarray(foreground, mode="L")
    mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.float32)[None, ...]


def _mask_area_fraction(path: Path) -> float:
    return float(_foreground_mask(path).mean())


def _mask_colors(path: Path) -> tuple[tuple[int, int, int], ...]:
    with Image.open(path) as handle:
        colors = handle.convert("RGB").getcolors(maxcolors=10_000_000)
    if colors is None:
        return tuple()
    return tuple(sorted({tuple(int(channel) for channel in color) for _count, color in colors}))


def _image_stats(image_path: Path, mask_path: Path) -> tuple[str, tuple[int, int], str, tuple[int, int]]:
    with Image.open(image_path) as image_handle, Image.open(mask_path) as mask_handle:
        return image_handle.mode, image_handle.size, mask_handle.mode, mask_handle.size


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_discovery_cache(cache_path: Path | None) -> dict:
    if cache_path is None or not cache_path.exists():
        return {"samples": {}}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and isinstance(loaded.get("samples"), dict):
            return loaded
    except (OSError, json.JSONDecodeError):
        pass
    return {"samples": {}}


def _save_discovery_cache(cache_path: Path | None, cache: dict) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def _cached_sample_stats(
    stem: str,
    image_path: Path,
    mask_path: Path,
    cache: dict,
) -> tuple[dict, bool]:
    image_signature = _file_signature(image_path)
    mask_signature = _file_signature(mask_path)
    cached = cache.get("samples", {}).get(stem)
    if (
        isinstance(cached, dict)
        and cached.get("image_signature") == image_signature
        and cached.get("mask_signature") == mask_signature
    ):
        return cached["stats"], True

    image_mode, image_size, mask_mode, mask_size = _image_stats(image_path, mask_path)
    stats = {
        "image_mode": image_mode,
        "image_size": list(image_size),
        "mask_mode": mask_mode,
        "mask_size": list(mask_size),
        "colors": [list(color) for color in _mask_colors(mask_path)],
        "mask_area_fraction": _mask_area_fraction(mask_path),
    }
    cache.setdefault("samples", {})[stem] = {
        "image_signature": image_signature,
        "mask_signature": mask_signature,
        "stats": stats,
    }
    return stats, False


def discover_bus_uclm_samples(
    root: str | Path = BUS_UCLM_ROOT,
    cache_path: str | Path | None = BUS_UCLM_PHASE5P_CACHE,
) -> tuple[list[BUSUCLMSample], dict]:
    """Discover BUS-UCLM image-mask pairs and dataset metadata."""

    root = Path(root)
    cache_path = Path(cache_path) if cache_path is not None else None
    cache = _load_discovery_cache(cache_path)
    cache_hits = 0
    cache_misses = 0
    image_dir = root / "images"
    mask_dir = root / "masks"
    info_path = root / "INFO.csv"
    warnings: list[dict[str, str]] = []

    image_files = _image_files(image_dir)
    mask_files = _image_files(mask_dir)
    info_rows = _read_info_rows(info_path)
    images_by_stem = {path.stem: path for path in image_files}
    masks_by_stem = {path.stem: path for path in mask_files}
    info_by_stem: dict[str, dict[str, str]] = {}
    duplicate_stems: list[str] = []
    for row in info_rows:
        stem = _sample_stem(row.get("Image", ""))
        if stem in info_by_stem:
            duplicate_stems.append(stem)
        info_by_stem[stem] = row

    all_stems = sorted(set(images_by_stem) | set(masks_by_stem) | set(info_by_stem))
    missing_images = sorted(stem for stem in all_stems if stem not in images_by_stem)
    missing_masks = sorted(stem for stem in all_stems if stem not in masks_by_stem)
    missing_info = sorted(stem for stem in all_stems if stem not in info_by_stem)
    matched_stems = sorted(set(images_by_stem) & set(masks_by_stem) & set(info_by_stem))

    samples: list[BUSUCLMSample] = []
    label_counts = Counter(row.get("Label", "") for row in info_rows)
    included_counts = Counter()
    deferred_counts = Counter()
    image_modes = Counter()
    image_sizes = Counter()
    mask_modes = Counter()
    mask_sizes = Counter()
    dimension_mismatches: list[dict[str, str]] = []
    unknown_mask_color_records: list[dict[str, str]] = []
    empty_included_masks: list[str] = []
    deferred_normal_examples: list[dict[str, str]] = []

    for stem in matched_stems:
        row = info_by_stem[stem]
        label = row.get("Label", "")
        image_path = images_by_stem[stem]
        mask_path = masks_by_stem[stem]
        stats, cache_hit = _cached_sample_stats(stem, image_path, mask_path, cache)
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
        image_mode = str(stats["image_mode"])
        image_size = tuple(int(value) for value in stats["image_size"])
        mask_mode = str(stats["mask_mode"])
        mask_size = tuple(int(value) for value in stats["mask_size"])
        image_modes[image_mode] += 1
        image_sizes[str(image_size)] += 1
        mask_modes[mask_mode] += 1
        mask_sizes[str(mask_size)] += 1
        if image_size != mask_size or image_mode != "RGB":
            dimension_mismatches.append(
                {
                    "sample_id": stem,
                    "label": label,
                    "image_mode": image_mode,
                    "image_size": str(image_size),
                    "mask_mode": mask_mode,
                    "mask_size": str(mask_size),
                }
            )
        colors = tuple(tuple(int(channel) for channel in color) for color in stats["colors"])
        unknown_colors = tuple(color for color in colors if color not in KNOWN_MASK_COLORS)
        for color in unknown_colors:
            unknown_mask_color_records.append(
                {
                    "sample_id": stem,
                    "label": label,
                    "unknown_color": str(color),
                }
            )
        if label in BUS_UCLM_DEFERRED_LABELS:
            deferred_counts[label] += 1
            if len(deferred_normal_examples) < 10:
                deferred_normal_examples.append(
                    {"sample_id": stem, "image_path": str(image_path), "mask_path": str(mask_path), "label": label}
                )
            continue
        if label not in BUS_UCLM_INCLUDED_LABELS:
            warnings.append({"category": "label", "sample_id": stem, "message": f"Unsupported label: {label}"})
            continue
        area = float(stats["mask_area_fraction"])
        if area == 0.0:
            empty_included_masks.append(stem)
        samples.append(
            BUSUCLMSample(
                image_path=image_path,
                mask_path=mask_path,
                label=label,
                sample_id=stem,
                patient_id=patient_id_from_stem(stem),
                mask_area_fraction=area,
                image_mode=image_mode,
                image_size=image_size,
                mask_mode=mask_mode,
                mask_size=mask_size,
                unknown_mask_colors=unknown_colors,
            )
        )
        included_counts[label] += 1

    for stem in missing_images:
        warnings.append({"category": "missing_image", "sample_id": stem, "message": "INFO/mask stem has no image file."})
    for stem in missing_masks:
        warnings.append({"category": "missing_mask", "sample_id": stem, "message": "INFO/image stem has no mask file."})
    for stem in missing_info:
        warnings.append({"category": "missing_info", "sample_id": stem, "message": "Image/mask stem has no INFO.csv row."})
    for stem in duplicate_stems:
        warnings.append({"category": "duplicate_info_stem", "sample_id": stem, "message": "Duplicate INFO.csv stem."})
    for record in unknown_mask_color_records:
        warnings.append(
            {
                "category": "unknown_mask_color",
                "sample_id": record["sample_id"],
                "message": f"Unknown mask color {record['unknown_color']}",
            }
        )
    for stem in empty_included_masks:
        warnings.append({"category": "empty_included_mask", "sample_id": stem, "message": "Included lesion mask is empty."})

    _save_discovery_cache(cache_path, cache)

    samples.sort(key=lambda sample: sample.sample_id)
    metadata = {
        "dataset": "bus_uclm",
        "root": str(root),
        "folder_status": {
            "root_exists": root.exists(),
            "images_exists": image_dir.exists(),
            "masks_exists": mask_dir.exists(),
            "info_exists": info_path.exists(),
        },
        "raw_image_count": len(image_files),
        "raw_mask_count": len(mask_files),
        "info_row_count": len(info_rows),
        "matched_pair_count": len(matched_stems),
        "missing_image_count": len(missing_images),
        "missing_mask_count": len(missing_masks),
        "missing_info_count": len(missing_info),
        "duplicate_stem_count": len(duplicate_stems),
        "duplicate_stems": duplicate_stems[:20],
        "classes_used": list(BUS_UCLM_INCLUDED_LABELS),
        "classes_deferred": list(BUS_UCLM_DEFERRED_LABELS),
        "label_counts": dict(label_counts),
        "included_counts": dict(included_counts),
        "deferred_counts": dict(deferred_counts),
        "included_total": len(samples),
        "deferred_total": sum(deferred_counts.values()),
        "patient_prefix_count_all": len({patient_id_from_stem(stem) for stem in matched_stems}),
        "patient_prefix_count_included": len({sample.patient_id for sample in samples}),
        "image_modes": dict(image_modes),
        "image_sizes": dict(image_sizes),
        "mask_modes": dict(mask_modes),
        "mask_sizes": dict(mask_sizes),
        "dimension_mismatch_count": len(dimension_mismatches),
        "dimension_mismatches": dimension_mismatches[:30],
        "unknown_mask_colors_count": len(unknown_mask_color_records),
        "unknown_mask_colors": unknown_mask_color_records[:30],
        "empty_included_mask_count": len(empty_included_masks),
        "empty_included_masks": empty_included_masks[:30],
        "missing_images": missing_images[:20],
        "missing_masks": missing_masks[:20],
        "missing_info": missing_info[:20],
        "deferred_normal_examples": deferred_normal_examples,
        "cache": {
            "path": str(cache_path) if cache_path is not None else "",
            "enabled": cache_path is not None,
            "hits": cache_hits,
            "misses": cache_misses,
            "sample_records": len(cache.get("samples", {})),
        },
        "warnings": warnings,
    }
    return samples, metadata


def _group_samples_by_patient(samples: list[BUSUCLMSample]) -> list[list[BUSUCLMSample]]:
    groups: dict[str, list[BUSUCLMSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.patient_id].append(sample)
    return [sorted(group, key=lambda sample: sample.sample_id) for _patient_id, group in sorted(groups.items())]


def _group_label_counts(group: list[BUSUCLMSample]) -> Counter:
    return Counter(sample.label for sample in group)


def _split_counts_by_label(samples: list[BUSUCLMSample]) -> Counter:
    return Counter(sample.label for sample in samples)


def split_bus_uclm_samples(samples: list[BUSUCLMSample], seed: int = 42) -> dict[str, list[BUSUCLMSample]]:
    """Create train/val/test splits without patient-prefix leakage."""

    rng = np.random.default_rng(seed)
    groups = _group_samples_by_patient(samples)
    rng.shuffle(groups)
    groups.sort(key=lambda group: (-len(group), group[0].patient_id))
    split_names = ["train", "val", "test"]
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    total = len(samples)
    total_labels = _split_counts_by_label(samples)
    target_totals = {split: total * ratio for split, ratio in ratios.items()}
    target_labels = {
        split: {label: total_labels[label] * ratio for label in BUS_UCLM_INCLUDED_LABELS}
        for split, ratio in ratios.items()
    }
    assigned: dict[str, list[list[BUSUCLMSample]]] = {split: [] for split in split_names}

    for group in groups:
        group_counts = _group_label_counts(group)
        best_split = None
        best_score = None
        for split in split_names:
            future_assigned = {name: list(items) for name, items in assigned.items()}
            future_assigned[split] = future_assigned[split] + [group]
            total_error = 0.0
            label_error = 0.0
            for future_split in split_names:
                future_samples = [sample for item in future_assigned[future_split] for sample in item]
                future_counts = _split_counts_by_label(future_samples)
                total_error += ((len(future_samples) - target_totals[future_split]) / max(total, 1)) ** 2
                for label in BUS_UCLM_INCLUDED_LABELS:
                    label_error += (
                        (future_counts[label] - target_labels[future_split][label]) / max(total_labels[label], 1)
                    ) ** 2
            # Prefer the split with the largest remaining target for the dominant label.
            dominant = "Malignant" if group_counts["Malignant"] > group_counts["Benign"] else "Benign"
            current_split_samples = [sample for item in assigned[split] for sample in item]
            current_counts = _split_counts_by_label(current_split_samples)
            remaining_label = target_labels[split][dominant] - current_counts[dominant]
            score = total_error + label_error - 0.001 * remaining_label
            if best_score is None or score < best_score:
                best_score = score
                best_split = split
        assigned[best_split or "train"].append(group)

    split_samples = {
        split: sorted((sample for group in groups_for_split for sample in group), key=lambda sample: sample.sample_id)
        for split, groups_for_split in assigned.items()
    }
    return split_samples


def _patient_groups_for_split(samples: list[BUSUCLMSample]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.patient_id].append(index)
    return [indices for _patient_id, indices in sorted(groups.items())]


def _group_stats(samples: list[BUSUCLMSample], indices: list[int]) -> dict[str, float]:
    labels = Counter(samples[index].label for index in indices)
    areas = [samples[index].mask_area_fraction for index in indices]
    count = len(indices)
    return {
        "count": float(count),
        "benign": float(labels["Benign"]),
        "malignant": float(labels["Malignant"]),
        "malignant_ratio": float(labels["Malignant"] / count) if count else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
    }


def _subset_stats(samples: list[BUSUCLMSample], indices: list[int]) -> dict[str, float]:
    labels = Counter(samples[index].label for index in indices)
    areas = [samples[index].mask_area_fraction for index in indices]
    count = len(indices)
    return {
        "count": float(count),
        "malignant_ratio": float(labels["Malignant"] / count) if count else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
    }


def _partition_score(
    samples: list[BUSUCLMSample],
    partitions: list[list[int]],
    target_count_fractions: list[float],
    target_malignant_ratios: list[float],
    target_area_means: list[float],
) -> float:
    total_count = max(len(samples), 1)
    area_scale = max(float(np.std([sample.mask_area_fraction for sample in samples])), 0.01)
    min_client_count = 1 if len(samples) < len(partitions) * 6 else 4
    iid_like = max(target_count_fractions) - min(target_count_fractions) < 1e-8
    count_weight = 8.0 if iid_like else 2.0
    ratio_weight = 1.0 if iid_like else 1.5
    area_weight = 0.05 if iid_like else 0.15
    score = 0.0
    for client_id, partition in enumerate(partitions):
        stats = _subset_stats(samples, partition)
        count_error = (stats["count"] - total_count * target_count_fractions[client_id]) / total_count
        ratio_error = stats["malignant_ratio"] - target_malignant_ratios[client_id]
        area_error = (stats["mean_area"] - target_area_means[client_id]) / area_scale
        score += count_weight * count_error**2 + ratio_weight * ratio_error**2 + area_weight * area_error**2
        if stats["count"] == 0:
            score += 100.0
        elif stats["count"] < min_client_count:
            score += (min_client_count - stats["count"]) * 2.0
    return score


def _targets_for_partition(samples: list[BUSUCLMSample], split_mode: RealSplitMode) -> tuple[list[float], list[float], list[float]]:
    total = max(len(samples), 1)
    malignant_count = sum(1 for sample in samples if sample.label == "Malignant")
    global_malignant_ratio = malignant_count / total
    global_area = float(np.mean([sample.mask_area_fraction for sample in samples])) if samples else 0.0
    if split_mode == "iid":
        return [1 / 3, 1 / 3, 1 / 3], [global_malignant_ratio] * 3, [global_area] * 3
    if split_mode == "moderate_noniid":
        return (
            [0.39, 0.34, 0.27],
            [
                max(global_malignant_ratio - 0.11, 0.02),
                global_malignant_ratio,
                min(global_malignant_ratio + 0.16, 0.98),
            ],
            [global_area * 0.88, global_area, global_area * 1.18],
        )
    return (
        [0.48, 0.32, 0.20],
        [
            max(global_malignant_ratio - 0.20, 0.02),
            min(global_malignant_ratio + 0.04, 0.98),
            min(global_malignant_ratio + 0.30, 0.98),
        ],
        [global_area * 0.72, global_area * 1.04, global_area * 1.42],
    )


def _partition_from_assignment(groups: list[list[int]], assignment: tuple[int, ...], num_clients: int) -> list[list[int]]:
    partitions = [[] for _ in range(num_clients)]
    for group, client_id in zip(groups, assignment):
        partitions[client_id].extend(group)
    return partitions


def _optimize_prefix_partition(samples: list[BUSUCLMSample], num_clients: int, split_mode: RealSplitMode) -> list[list[int]]:
    if num_clients != 3:
        partitions = [[] for _ in range(num_clients)]
        groups = _patient_groups_for_split(samples)
        groups.sort(key=lambda indices: (-len(indices), samples[indices[0]].patient_id))
        for group in groups:
            target_client = min(range(num_clients), key=lambda client_id: (len(partitions[client_id]), client_id))
            partitions[target_client].extend(group)
        for partition in partitions:
            partition.sort()
        return partitions

    groups = _patient_groups_for_split(samples)
    groups.sort(
        key=lambda indices: (
            -abs(_group_stats(samples, indices)["malignant_ratio"] - _targets_for_partition(samples, split_mode)[1][1]),
            -len(indices),
            samples[indices[0]].patient_id,
        )
    )
    target_count_fractions, target_malignant_ratios, target_area_means = _targets_for_partition(samples, split_mode)

    if len(groups) <= 10:
        best_assignment: tuple[int, ...] | None = None
        best_score: float | None = None
        for assignment in itertools.product(range(num_clients), repeat=len(groups)):
            partitions = _partition_from_assignment(groups, assignment, num_clients)
            score = _partition_score(samples, partitions, target_count_fractions, target_malignant_ratios, target_area_means)
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = assignment
        partitions = _partition_from_assignment(groups, best_assignment or tuple(), num_clients)
    else:
        partitions = [[] for _ in range(num_clients)]
        for group in groups:
            best_client = 0
            best_score = None
            for client_id in range(num_clients):
                future_partitions = [list(partition) for partition in partitions]
                future_partitions[client_id] = future_partitions[client_id] + group
                score = _partition_score(
                    samples,
                    future_partitions,
                    target_count_fractions,
                    target_malignant_ratios,
                    target_area_means,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_client = client_id
            partitions[best_client].extend(group)

    for partition in partitions:
        partition.sort()
    return partitions


def _balanced_prefix_partition(samples: list[BUSUCLMSample], num_clients: int) -> list[list[int]]:
    return _optimize_prefix_partition(samples, num_clients, "iid")


def _targeted_prefix_partition(samples: list[BUSUCLMSample], num_clients: int, split_mode: RealSplitMode) -> list[list[int]]:
    return _optimize_prefix_partition(samples, num_clients, split_mode)


def partition_bus_uclm_clients(
    split_samples: dict[str, list[BUSUCLMSample]],
    num_clients: int,
    split_mode: RealSplitMode,
) -> dict[str, list[list[int]]]:
    return {
        split_name: _targeted_prefix_partition(samples, num_clients, split_mode)
        for split_name, samples in split_samples.items()
    }


def _area_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_mask_area_fraction": 0.0,
            "median_mask_area_fraction": 0.0,
            "min_mask_area_fraction": 0.0,
            "max_mask_area_fraction": 0.0,
        }
    return {
        "mean_mask_area_fraction": float(np.mean(values)),
        "median_mask_area_fraction": float(np.median(values)),
        "min_mask_area_fraction": float(np.min(values)),
        "max_mask_area_fraction": float(np.max(values)),
    }


def summarize_client_distribution(
    split_samples: dict[str, list[BUSUCLMSample]],
    partitions: dict[str, list[list[int]]],
) -> list[dict]:
    rows = []
    for split_name, samples in split_samples.items():
        for client_id, indices in enumerate(partitions[split_name]):
            labels = Counter(samples[index].label for index in indices)
            areas = [samples[index].mask_area_fraction for index in indices]
            patient_ids = {samples[index].patient_id for index in indices}
            area_stats = _area_stats(areas)
            total = len(indices)
            rows.append(
                {
                    "split": split_name,
                    "client_id": client_id,
                    "sample_count": total,
                    "benign_count": labels["Benign"],
                    "malignant_count": labels["Malignant"],
                    "malignant_ratio": float(labels["Malignant"] / total) if total else 0.0,
                    "patient_prefix_count": len(patient_ids),
                    **area_stats,
                    "empty_client": total == 0,
                }
            )
    return rows


def leakage_report(
    split_samples: dict[str, list[BUSUCLMSample]],
    partitions: dict[str, list[list[int]]] | None = None,
) -> dict:
    split_stems = {split: {sample.sample_id for sample in samples} for split, samples in split_samples.items()}
    split_patients = {split: {sample.patient_id for sample in samples} for split, samples in split_samples.items()}
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    stem_overlap = {
        f"{left}_{right}": sorted(split_stems[left] & split_stems[right])
        for left, right in split_pairs
    }
    patient_overlap = {
        f"{left}_{right}": sorted(split_patients[left] & split_patients[right])
        for left, right in split_pairs
    }
    client_overlap: dict[str, dict[str, list[str]]] = {}
    if partitions is not None:
        for split, samples in split_samples.items():
            client_sets = []
            client_patient_sets = []
            for indices in partitions[split]:
                client_sets.append({samples[index].sample_id for index in indices})
                client_patient_sets.append({samples[index].patient_id for index in indices})
            for i in range(len(client_sets)):
                for j in range(i + 1, len(client_sets)):
                    key = f"{split}_client{i}_client{j}"
                    client_overlap[key] = {
                        "sample_overlap": sorted(client_sets[i] & client_sets[j]),
                        "patient_overlap": sorted(client_patient_sets[i] & client_patient_sets[j]),
                    }
    passed = not any(stem_overlap.values()) and not any(patient_overlap.values())
    if partitions is not None:
        passed = passed and all(
            not value["sample_overlap"] and not value["patient_overlap"]
            for value in client_overlap.values()
        )
    return {
        "stem_overlap": stem_overlap,
        "patient_overlap": patient_overlap,
        "client_overlap": client_overlap,
        "passed": passed,
    }


def build_bus_uclm_client_loaders(
    num_clients: int,
    batch_size: int,
    split_mode: RealSplitMode = "iid",
    image_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    root: str | Path = BUS_UCLM_ROOT,
) -> tuple[dict[int, dict[str, DataLoader]], dict]:
    """Create BUS-UCLM train/val/test loaders for simulated clients."""

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if split_mode not in {"iid", "moderate_noniid", "hard_noniid"}:
        raise ValueError("BUS-UCLM supports iid, moderate_noniid, and hard_noniid splits.")

    samples, dataset_metadata = discover_bus_uclm_samples(root)
    if not samples:
        raise RuntimeError("No valid BUS-UCLM benign/malignant image-mask pairs found.")

    split_samples = split_bus_uclm_samples(samples, seed=seed)
    partitions = partition_bus_uclm_clients(split_samples, num_clients, split_mode)
    leakage = leakage_report(split_samples, partitions)
    if not leakage["passed"]:
        raise RuntimeError(f"BUS-UCLM split leakage detected: {leakage}")

    loaders: dict[int, dict[str, DataLoader]] = {}
    for client_id in range(num_clients):
        client_loaders: dict[str, DataLoader] = {}
        for split_name, samples_for_split in split_samples.items():
            dataset = BUSUCLMSegmentationDataset(samples_for_split, image_size=image_size)
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

    split_metadata = {
        "split_mode": split_mode,
        "description": {
            "iid": "BUS-UCLM patient prefixes are assigned to clients with balanced counts.",
            "moderate_noniid": "BUS-UCLM clients receive moderate benign/malignant, mask-area, and count imbalance without prefix leakage.",
            "hard_noniid": "BUS-UCLM clients receive stronger but conservative benign/malignant, mask-area, and count imbalance without prefix leakage.",
        }[split_mode],
        "dataset": dataset_metadata,
        "clients": summarize_client_distribution(split_samples, partitions),
        "leakage_check": leakage,
    }
    return loaders, split_metadata
