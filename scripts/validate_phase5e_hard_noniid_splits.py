"""Validate Phase 5E real-data hard_noniid splits without training."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.busi import (  # noqa: E402
    BUSI_CLASSES,
    _hard_noniid_partition as busi_hard_partition,
    _moderate_noniid_partition as busi_moderate_partition,
    discover_busi_samples,
    split_busi_samples,
)
from datasets.kvasir_seg import (  # noqa: E402
    _hard_noniid_partition as kvasir_hard_partition,
    _moderate_noniid_partition as kvasir_moderate_partition,
    discover_kvasir_samples,
    split_kvasir_samples,
)

SUMMARY_DIR = PROJECT_ROOT / "results/summaries"
FIGURE_DIR = PROJECT_ROOT / "results/figures/phase5e_real_hard_noniid_splits"
REPORT_PATH = SUMMARY_DIR / "phase5e_real_hard_noniid_split_validation.md"
BUSI_HARD_CSV = SUMMARY_DIR / "phase5e_busi_hard_noniid_client_distribution.csv"
KVASIR_HARD_CSV = SUMMARY_DIR / "phase5e_kvasir_seg_hard_noniid_client_distribution.csv"
BUSI_COMPARISON_CSV = SUMMARY_DIR / "phase5e_busi_moderate_vs_hard_noniid_comparison.csv"
KVASIR_COMPARISON_CSV = SUMMARY_DIR / "phase5e_kvasir_seg_moderate_vs_hard_noniid_comparison.csv"

SPLIT_NAMES = ("train", "val", "test")
SPLIT_MODES = ("moderate_noniid", "hard_noniid")
NUM_CLIENTS = 3
MIN_CLIENT_SAMPLES = 10
MIN_MEAN_MASK_AREA = 0.001
MAX_MEAN_MASK_AREA = 0.70


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _area_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_mask_area_fraction": 0.0,
            "median_mask_area_fraction": 0.0,
            "min_mask_area_fraction": 0.0,
            "q25_mask_area_fraction": 0.0,
            "q75_mask_area_fraction": 0.0,
            "max_mask_area_fraction": 0.0,
        }
    return {
        "mean_mask_area_fraction": float(np.mean(values)),
        "median_mask_area_fraction": float(median(values)),
        "min_mask_area_fraction": float(min(values)),
        "q25_mask_area_fraction": _quantile(values, 0.25),
        "q75_mask_area_fraction": _quantile(values, 0.75),
        "max_mask_area_fraction": float(max(values)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _identity(sample: Any) -> str:
    if hasattr(sample, "image_path"):
        return f"{sample.image_path.parent.name}/{sample.image_path.stem}"
    return sample.sample_id


def _leakage_check(split_samples: dict[str, list[Any]]) -> dict[str, Any]:
    ids = {split_name: {_identity(sample) for sample in samples} for split_name, samples in split_samples.items()}
    overlaps = {
        "train_val": sorted(ids["train"].intersection(ids["val"])),
        "train_test": sorted(ids["train"].intersection(ids["test"])),
        "val_test": sorted(ids["val"].intersection(ids["test"])),
    }
    return {"overlaps": overlaps, "passed": not any(overlaps.values())}


def _partitions(
    split_samples: dict[str, list[Any]],
    moderate_fn: Callable[[list[Any], int], list[list[int]]],
    hard_fn: Callable[[list[Any], int], list[list[int]]],
) -> dict[str, dict[str, list[list[int]]]]:
    return {
        "moderate_noniid": {
            split_name: moderate_fn(samples, NUM_CLIENTS) for split_name, samples in split_samples.items()
        },
        "hard_noniid": {split_name: hard_fn(samples, NUM_CLIENTS) for split_name, samples in split_samples.items()},
    }


def _kvasir_group_thresholds(samples: list[Any]) -> tuple[float, float]:
    areas = [float(sample.mask_area_fraction) for sample in samples]
    return _quantile(areas, 1 / 3), _quantile(areas, 2 / 3)


def _kvasir_group(sample: Any, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if sample.mask_area_fraction <= low:
        return "small"
    if sample.mask_area_fraction <= high:
        return "medium"
    return "large"


def _client_rows(
    dataset: str,
    split_samples: dict[str, list[Any]],
    partitions_by_mode: dict[str, dict[str, list[list[int]]]],
    kvasir_thresholds: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_mode in SPLIT_MODES:
        for split_name in SPLIT_NAMES:
            samples = split_samples[split_name]
            total = len(samples)
            for client_id, indices in enumerate(partitions_by_mode[split_mode][split_name]):
                selected = [samples[index] for index in indices]
                areas = [float(sample.mask_area_fraction) for sample in selected]
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "split_mode": split_mode,
                    "subset": split_name,
                    "client_id": client_id,
                    "sample_count": len(selected),
                    "sample_pct": round(len(selected) / total, 6) if total else 0.0,
                    "unique_identity_count": len({_identity(sample) for sample in selected}),
                    "empty_client": len(selected) == 0,
                    **{key: round(value, 6) for key, value in _area_stats(areas).items()},
                }
                if dataset == "busi":
                    for class_name in BUSI_CLASSES:
                        count = sum(1 for sample in selected if sample.class_name == class_name)
                        row[f"{class_name}_count"] = count
                        row[f"{class_name}_ratio"] = round(count / len(selected), 6) if selected else 0.0
                else:
                    groups = [_kvasir_group(sample, kvasir_thresholds or (0.0, 1.0)) for sample in selected]
                    for group_name in ("small", "medium", "large"):
                        count = groups.count(group_name)
                        row[f"{group_name}_mask_count"] = count
                        row[f"{group_name}_mask_ratio"] = round(count / len(selected), 6) if selected else 0.0
                    image_means = [float(sample.image_mean) for sample in selected]
                    image_stds = [float(sample.image_std) for sample in selected]
                    row["mean_image_intensity"] = round(float(np.mean(image_means)), 6) if image_means else 0.0
                    row["mean_image_std"] = round(float(np.mean(image_stds)), 6) if image_stds else 0.0
                rows.append(row)
    return rows


def _comparison_rows(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        subset_rows = [row for row in rows if row["subset"] == split_name]
        values: dict[str, dict[str, float]] = {}
        for split_mode in SPLIT_MODES:
            mode_rows = [row for row in subset_rows if row["split_mode"] == split_mode]
            counts = [float(row["sample_count"]) for row in mode_rows]
            mean_areas = [float(row["mean_mask_area_fraction"]) for row in mode_rows]
            values[split_mode] = {
                "sample_count_imbalance_ratio": max(counts) / min(counts) if counts and min(counts) > 0 else 0.0,
                "mean_mask_area_client_range": max(mean_areas) - min(mean_areas) if mean_areas else 0.0,
            }
            if dataset == "busi":
                malignant_ratios = [float(row["malignant_ratio"]) for row in mode_rows]
                values[split_mode]["malignant_ratio_client_range"] = (
                    max(malignant_ratios) - min(malignant_ratios) if malignant_ratios else 0.0
                )
            else:
                large_ratios = [float(row["large_mask_ratio"]) for row in mode_rows]
                values[split_mode]["large_mask_ratio_client_range"] = (
                    max(large_ratios) - min(large_ratios) if large_ratios else 0.0
                )
        for metric in values["hard_noniid"]:
            moderate_value = values["moderate_noniid"][metric]
            hard_value = values["hard_noniid"][metric]
            comparisons.append(
                {
                    "dataset": dataset,
                    "subset": split_name,
                    "metric": metric,
                    "moderate_noniid": round(moderate_value, 6),
                    "hard_noniid": round(hard_value, 6),
                    "hard_greater_than_moderate": hard_value > moderate_value,
                }
            )
    return comparisons


def _validate(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], leakage: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    hard_rows = [row for row in rows if row["split_mode"] == "hard_noniid"]
    for row in hard_rows:
        if row["empty_client"]:
            issues.append(f"{row['dataset']} {row['subset']} client {row['client_id']} is empty.")
        if int(row["sample_count"]) < MIN_CLIENT_SAMPLES:
            issues.append(f"{row['dataset']} {row['subset']} client {row['client_id']} has too few samples.")
        mean_area = float(row["mean_mask_area_fraction"])
        if mean_area < MIN_MEAN_MASK_AREA or mean_area > MAX_MEAN_MASK_AREA:
            issues.append(f"{row['dataset']} {row['subset']} client {row['client_id']} has extreme mean mask area.")
    if not leakage["passed"]:
        issues.append("Train/val/test leakage detected.")
    for row in comparisons:
        if not bool(row["hard_greater_than_moderate"]):
            issues.append(f"{row['dataset']} {row['subset']} {row['metric']} is not harder than moderate_noniid.")
    return not issues, issues


def _draw_bar_figure(path: Path, title: str, labels: list[str], values: list[float], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((32, 24), title, fill="black", font=font)
    chart_left, chart_top, chart_right, chart_bottom = 90, 80, width - 40, height - 120
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(210, 210, 210))
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-6)
    gap = 12
    bar_width = max(18, int((chart_right - chart_left - gap * (len(values) + 1)) / max(1, len(values))))
    for idx, value in enumerate(values):
        x0 = chart_left + gap + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_height = int((value / max_value) * (chart_bottom - chart_top - 20))
        y0 = chart_bottom - bar_height
        draw.rectangle((x0, y0, x1, chart_bottom), fill=color)
        draw.text((x0, max(chart_top + 4, y0 - 16)), f"{value:.3f}", fill="black", font=font)
        draw.text((x0, chart_bottom + 10), labels[idx], fill="black", font=font)
    image.save(path)


def _hard_distribution_figure(dataset: str, rows: list[dict[str, Any]], path: Path) -> None:
    train_rows = [row for row in rows if row["split_mode"] == "hard_noniid" and row["subset"] == "train"]
    labels = [f"C{row['client_id']} n" for row in train_rows] + [f"C{row['client_id']} area" for row in train_rows]
    values = [float(row["sample_count"]) for row in train_rows] + [
        float(row["mean_mask_area_fraction"]) * 100 for row in train_rows
    ]
    _draw_bar_figure(path, f"Phase 5E {dataset} hard_noniid train distribution", labels, values, (92, 139, 168))


def _comparison_figure(dataset: str, comparisons: list[dict[str, Any]], path: Path) -> None:
    train_rows = [row for row in comparisons if row["subset"] == "train"]
    labels: list[str] = []
    values: list[float] = []
    for row in train_rows:
        labels.extend([f"mod {row['metric'][:8]}", f"hard {row['metric'][:8]}"])
        values.extend([float(row["moderate_noniid"]), float(row["hard_noniid"])])
    _draw_bar_figure(path, f"Phase 5E {dataset} moderate vs hard train comparison", labels, values, (178, 112, 86))


def _report_lines(
    busi_status: tuple[bool, list[str]],
    kvasir_status: tuple[bool, list[str]],
    busi_metadata: dict[str, Any],
    kvasir_metadata: dict[str, Any],
    busi_comparison: list[dict[str, Any]],
    kvasir_comparison: list[dict[str, Any]],
) -> list[str]:
    def comparison_lines(rows: list[dict[str, Any]]) -> list[str]:
        lines = []
        for row in rows:
            if row["subset"] == "train":
                lines.append(
                    f"- {row['metric']}: moderate={row['moderate_noniid']}, "
                    f"hard={row['hard_noniid']}, hard_greater={row['hard_greater_than_moderate']}"
                )
        return lines

    def issue_lines(issues: list[str]) -> list[str]:
        if not issues:
            return ["- None"]
        return [f"- {issue}" for issue in issues]

    busi_passed, busi_issues = busi_status
    kvasir_passed, kvasir_issues = kvasir_status
    return [
        "# Phase 5E Real hard_noniid Split Validation",
        "",
        "This report validates real-data split difficulty only. It does not run training, Tier 1 pilots, or failed variants.",
        "",
        "## Scope",
        "",
        "- Image size for mask-area diagnosis: 128",
        "- BUSI classes used: benign and malignant only",
        "- BUSI normal class: deferred",
        "- Kvasir-SEG pairing: image/mask filename stem",
        "- Leakage identity: original image stem namespaced by source folder",
        "",
        "## Dataset Counts",
        "",
        f"- BUSI valid benign/malignant pairs: {busi_metadata['valid_pair_count']}",
        f"- BUSI images with multiple masks merged: {busi_metadata['images_with_multiple_masks']}",
        f"- Kvasir-SEG valid pairs: {kvasir_metadata['valid_pair_count']}",
        "",
        "## BUSI hard_noniid vs moderate_noniid",
        "",
        *comparison_lines(busi_comparison),
        f"- Validation passed: {busi_passed}",
        "- Issues:",
        *issue_lines(busi_issues),
        "",
        "## Kvasir-SEG hard_noniid vs moderate_noniid",
        "",
        *comparison_lines(kvasir_comparison),
        f"- Validation passed: {kvasir_passed}",
        "- Issues:",
        *issue_lines(kvasir_issues),
        "",
        "## Output Files",
        "",
        f"- `{BUSI_HARD_CSV.relative_to(PROJECT_ROOT)}`",
        f"- `{KVASIR_HARD_CSV.relative_to(PROJECT_ROOT)}`",
        f"- `{BUSI_COMPARISON_CSV.relative_to(PROJECT_ROOT)}`",
        f"- `{KVASIR_COMPARISON_CSV.relative_to(PROJECT_ROOT)}`",
        f"- `{FIGURE_DIR.relative_to(PROJECT_ROOT)}/`",
    ]


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    busi_samples, busi_metadata = discover_busi_samples()
    busi_split_samples = split_busi_samples(busi_samples, seed=42)
    busi_partitions = _partitions(busi_split_samples, busi_moderate_partition, busi_hard_partition)
    busi_rows = _client_rows("busi", busi_split_samples, busi_partitions)
    busi_comparison = _comparison_rows("busi", busi_rows)
    busi_status = _validate(busi_rows, busi_comparison, _leakage_check(busi_split_samples))

    kvasir_samples, kvasir_metadata = discover_kvasir_samples()
    kvasir_split_samples = split_kvasir_samples(kvasir_samples, seed=42)
    kvasir_partitions = _partitions(kvasir_split_samples, kvasir_moderate_partition, kvasir_hard_partition)
    kvasir_thresholds = _kvasir_group_thresholds(kvasir_samples)
    kvasir_rows = _client_rows("kvasir_seg", kvasir_split_samples, kvasir_partitions, kvasir_thresholds)
    kvasir_comparison = _comparison_rows("kvasir_seg", kvasir_rows)
    kvasir_status = _validate(kvasir_rows, kvasir_comparison, _leakage_check(kvasir_split_samples))

    _write_csv(BUSI_HARD_CSV, [row for row in busi_rows if row["split_mode"] == "hard_noniid"])
    _write_csv(KVASIR_HARD_CSV, [row for row in kvasir_rows if row["split_mode"] == "hard_noniid"])
    _write_csv(BUSI_COMPARISON_CSV, busi_comparison)
    _write_csv(KVASIR_COMPARISON_CSV, kvasir_comparison)

    _hard_distribution_figure("BUSI", busi_rows, FIGURE_DIR / "busi_hard_noniid_client_distribution.png")
    _hard_distribution_figure(
        "Kvasir-SEG",
        kvasir_rows,
        FIGURE_DIR / "kvasir_seg_hard_noniid_client_distribution.png",
    )
    _comparison_figure("BUSI", busi_comparison, FIGURE_DIR / "busi_moderate_vs_hard_noniid_comparison.png")
    _comparison_figure(
        "Kvasir-SEG",
        kvasir_comparison,
        FIGURE_DIR / "kvasir_seg_moderate_vs_hard_noniid_comparison.png",
    )

    lines = _report_lines(
        busi_status,
        kvasir_status,
        busi_metadata,
        kvasir_metadata,
        busi_comparison,
        kvasir_comparison,
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Phase 5E report saved to: {REPORT_PATH}")
    print(f"BUSI hard_noniid validation passed: {busi_status[0]}")
    print(f"Kvasir-SEG hard_noniid validation passed: {kvasir_status[0]}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
