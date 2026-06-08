"""Validate BUS-UCLM loader, splits, leakage checks, and sample visualizations.

This Phase 5O script does not train models.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bus_uclm import (  # noqa: E402
    BUS_UCLM_INCLUDED_LABELS,
    BUS_UCLM_ROOT,
    build_bus_uclm_client_loaders,
    discover_bus_uclm_samples,
    leakage_report,
    partition_bus_uclm_clients,
    split_bus_uclm_samples,
    summarize_client_distribution,
)


SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5o_bus_uclm_loader"
SPLIT_MODES = ["iid", "moderate_noniid", "hard_noniid"]
SPLIT_NAMES = ["train", "val", "test"]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def area_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def split_label_counts(split_samples: dict[str, list]) -> list[dict[str, Any]]:
    rows = []
    for split_name, samples in split_samples.items():
        labels = Counter(sample.label for sample in samples)
        areas = [sample.mask_area_fraction for sample in samples]
        stats = area_summary(areas)
        rows.append(
            {
                "split_mode": "base_train_val_test",
                "split": split_name,
                "client_id": "",
                "sample_count": len(samples),
                "benign_count": labels["Benign"],
                "malignant_count": labels["Malignant"],
                "malignant_ratio": fmt(labels["Malignant"] / len(samples)) if samples else "0",
                "patient_prefix_count": len({sample.patient_id for sample in samples}),
                "mean_mask_area_fraction": fmt(stats["mean"]),
                "median_mask_area_fraction": fmt(stats["median"]),
                "min_mask_area_fraction": fmt(stats["min"]),
                "max_mask_area_fraction": fmt(stats["max"]),
                "empty_client": "",
            }
        )
    return rows


def client_rows_for_mode(split_samples: dict[str, list], split_mode: str) -> list[dict[str, Any]]:
    partitions = partition_bus_uclm_clients(split_samples, num_clients=3, split_mode=split_mode)
    rows = summarize_client_distribution(split_samples, partitions)
    output = []
    for row in rows:
        output.append(
            {
                "split_mode": split_mode,
                "split": row["split"],
                "client_id": row["client_id"],
                "sample_count": row["sample_count"],
                "benign_count": row["benign_count"],
                "malignant_count": row["malignant_count"],
                "malignant_ratio": fmt(row["malignant_ratio"]),
                "patient_prefix_count": row["patient_prefix_count"],
                "mean_mask_area_fraction": fmt(row["mean_mask_area_fraction"]),
                "median_mask_area_fraction": fmt(row["median_mask_area_fraction"]),
                "min_mask_area_fraction": fmt(row["min_mask_area_fraction"]),
                "max_mask_area_fraction": fmt(row["max_mask_area_fraction"]),
                "empty_client": row["empty_client"],
            }
        )
    return output


def range_for(rows: list[dict[str, Any]], split_mode: str, split_name: str, field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if row["split_mode"] == split_mode and row["split"] == split_name and row["client_id"] != ""
    ]
    if not values:
        return 0.0
    return max(values) - min(values)


def validate_tensor_loading() -> list[dict[str, str]]:
    rows = []
    for split_mode in SPLIT_MODES:
        loaders, _metadata = build_bus_uclm_client_loaders(
            num_clients=3,
            batch_size=4,
            split_mode=split_mode,
            image_size=128,
            seed=42,
        )
        for client_id, split_loaders in loaders.items():
            for split_name, loader in split_loaders.items():
                batch = next(iter(loader))
                image, mask = batch
                rows.append(
                    {
                        "split_mode": split_mode,
                        "split": split_name,
                        "client_id": str(client_id),
                        "image_shape": "x".join(str(dim) for dim in image.shape),
                        "mask_shape": "x".join(str(dim) for dim in mask.shape),
                        "mask_min": fmt(float(mask.min())),
                        "mask_max": fmt(float(mask.max())),
                        "passed": str(tuple(image.shape[1:]) == (1, 128, 128) and tuple(mask.shape[1:]) == (1, 128, 128)),
                    }
                )
    return rows


def binary_mask_array(mask_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    foreground = np.zeros(rgb.shape[:2], dtype=bool)
    for color in [(0, 255, 0), (255, 0, 0)]:
        foreground |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)
    return foreground


def _panel_with_title(image: Image.Image, title: str, size: tuple[int, int] = (256, 192)) -> Image.Image:
    image = image.convert("RGB").resize(size, resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title[:38], fill=(0, 0, 0))
    return canvas


def save_overlay_figure(samples: list, title: str, output_path: Path, max_items: int = 3) -> None:
    selected = samples[:max_items]
    if not selected:
        return
    row_images = []
    for row_index, sample in enumerate(selected):
        image = Image.open(sample.image_path).convert("RGB")
        mask = binary_mask_array(sample.mask_path)
        resized = image.resize(mask.shape[::-1], resample=Image.Resampling.BILINEAR)
        overlay_arr = np.asarray(resized).copy()
        overlay_arr[mask] = (0.55 * overlay_arr[mask] + 0.45 * np.asarray([255, 0, 0])).astype(np.uint8)
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")
        panels = [
            _panel_with_title(image, f"{sample.sample_id} image"),
            _panel_with_title(mask_img, f"{sample.label} mask"),
            _panel_with_title(Image.fromarray(overlay_arr), "overlay"),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        row_images.append(row)
    header = Image.new("RGB", (row_images[0].width, 34), "white")
    ImageDraw.Draw(header).text((6, 10), title, fill=(0, 0, 0))
    canvas = Image.new("RGB", (row_images[0].width, header.height + sum(row.height for row in row_images)), "white")
    canvas.paste(header, (0, 0))
    y = header.height
    for row in row_images:
        canvas.paste(row, (0, y))
        y += row.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def normal_proxy(metadata: dict[str, Any]) -> Any:
    examples = metadata.get("deferred_normal_examples", [])
    if not examples:
        return None
    example = examples[0]

    class SampleProxy:
        sample_id = example["sample_id"]
        image_path = Path(example["image_path"])
        mask_path = Path(example["mask_path"])
        label = "Normal-deferred"

    return SampleProxy()


def make_visualizations(samples: list, metadata: dict[str, Any]) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    benign = [sample for sample in samples if sample.label == "Benign"]
    malignant = [sample for sample in samples if sample.label == "Malignant"]
    hesn = [sample for sample in samples if sample.sample_id.startswith("HESN")]
    normal = normal_proxy(metadata)
    paths = [
        FIGURE_DIR / "phase5o_bus_uclm_benign_overlay_examples.png",
        FIGURE_DIR / "phase5o_bus_uclm_malignant_overlay_examples.png",
        FIGURE_DIR / "phase5o_bus_uclm_normal_deferred_reference.png",
        FIGURE_DIR / "phase5o_bus_uclm_hesn_mismatch_resize_example.png",
    ]
    save_overlay_figure(benign, "BUS-UCLM benign included examples", paths[0])
    save_overlay_figure(malignant, "BUS-UCLM malignant included examples", paths[1])
    if normal is not None:
        save_overlay_figure([normal], "BUS-UCLM normal deferred reference", paths[2], max_items=1)
    if hesn:
        save_overlay_figure(hesn, "BUS-UCLM HESN mismatch safely readable/resizable", paths[3], max_items=2)
    return [path for path in paths if path.exists()]


def build_validation_rows(samples: list, metadata: dict[str, Any], distribution_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    split_samples = split_bus_uclm_samples(samples, seed=42)
    validation_rows: list[dict[str, str]] = []
    dataset_checks = {
        "root_exists": metadata["folder_status"]["root_exists"],
        "images_exists": metadata["folder_status"]["images_exists"],
        "masks_exists": metadata["folder_status"]["masks_exists"],
        "info_exists": metadata["folder_status"]["info_exists"],
        "matched_pairs_equal_info_rows": metadata["matched_pair_count"] == metadata["info_row_count"],
        "included_benign_malignant_only": set(Counter(sample.label for sample in samples)) <= set(BUS_UCLM_INCLUDED_LABELS),
        "normal_deferred": metadata["deferred_counts"].get("Normal", 0) > 0,
        "no_missing_images": metadata["missing_image_count"] == 0,
        "no_missing_masks": metadata["missing_mask_count"] == 0,
        "no_duplicate_stems": metadata["duplicate_stem_count"] == 0,
        "no_unknown_mask_colors": metadata["unknown_mask_colors_count"] == 0,
        "no_empty_included_masks": metadata["empty_included_mask_count"] == 0,
    }
    for check, passed in dataset_checks.items():
        validation_rows.append({"category": "dataset", "check": check, "value": str(passed), "status": "passed" if passed else "failed"})

    for split_mode in SPLIT_MODES:
        partitions = partition_bus_uclm_clients(split_samples, num_clients=3, split_mode=split_mode)
        leakage = leakage_report(split_samples, partitions)
        empty_clients = [
            row
            for row in distribution_rows
            if row["split_mode"] == split_mode and str(row["empty_client"]) == "True"
        ]
        validation_rows.append(
            {
                "category": "leakage",
                "check": f"{split_mode}_patient_and_stem_leakage",
                "value": str(leakage["passed"]),
                "status": "passed" if leakage["passed"] else "failed",
            }
        )
        validation_rows.append(
            {
                "category": "split",
                "check": f"{split_mode}_empty_client_check",
                "value": str(len(empty_clients)),
                "status": "passed" if not empty_clients else "needs_adjustment",
            }
        )

    moderate_sample_range = range_for(distribution_rows, "moderate_noniid", "train", "sample_count")
    hard_sample_range = range_for(distribution_rows, "hard_noniid", "train", "sample_count")
    moderate_ratio_range = range_for(distribution_rows, "moderate_noniid", "train", "malignant_ratio")
    hard_ratio_range = range_for(distribution_rows, "hard_noniid", "train", "malignant_ratio")
    moderate_area_range = range_for(distribution_rows, "moderate_noniid", "train", "mean_mask_area_fraction")
    hard_area_range = range_for(distribution_rows, "hard_noniid", "train", "mean_mask_area_fraction")
    hard_harder = (
        hard_sample_range > moderate_sample_range
        or hard_ratio_range > moderate_ratio_range
        or hard_area_range > moderate_area_range
    )
    hard_too_extreme = any(
        row["split_mode"] == "hard_noniid"
        and row["split"] == "train"
        and row["client_id"] != ""
        and int(row["sample_count"]) < 10
        for row in distribution_rows
    )
    validation_rows.extend(
        [
            {
                "category": "split",
                "check": "hard_harder_than_moderate_train",
                "value": (
                    f"sample_range {fmt(hard_sample_range)} vs {fmt(moderate_sample_range)}; "
                    f"ratio_range {fmt(hard_ratio_range)} vs {fmt(moderate_ratio_range)}; "
                    f"area_range {fmt(hard_area_range)} vs {fmt(moderate_area_range)}"
                ),
                "status": "passed" if hard_harder else "needs_adjustment",
            },
            {
                "category": "split",
                "check": "hard_not_too_extreme_train_min_count",
                "value": str(not hard_too_extreme),
                "status": "passed" if not hard_too_extreme else "needs_adjustment",
            },
        ]
    )
    return validation_rows


def warning_rows(metadata: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for record in metadata.get("warnings", []):
        rows.append(
            {
                "category": record.get("category", ""),
                "sample_id": record.get("sample_id", ""),
                "message": record.get("message", ""),
            }
        )
    for record in metadata.get("dimension_mismatches", []):
        rows.append(
            {
                "category": "dimension_or_mode_mismatch",
                "sample_id": record["sample_id"],
                "message": (
                    f"image {record['image_mode']} {record['image_size']} vs "
                    f"mask {record['mask_mode']} {record['mask_size']}; label={record['label']}"
                ),
            }
        )
    if not rows:
        rows.append({"category": "none", "sample_id": "", "message": "No warnings."})
    return rows


def write_markdown(
    metadata: dict[str, Any],
    split_samples: dict[str, list],
    distribution_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, str]],
    tensor_rows: list[dict[str, str]],
    figures: list[Path],
) -> None:
    failed = [row for row in validation_rows if row["status"] == "failed"]
    needs_adjustment = [row for row in validation_rows if row["status"] == "needs_adjustment"]
    loader_status = "failed" if failed else "needs adjustment" if needs_adjustment else "passed"
    phase5p_recommended = loader_status == "passed"
    hard_adjustment = any(row["status"] == "needs_adjustment" and "hard" in row["check"] for row in validation_rows)
    lines = [
        "# Phase 5O BUS-UCLM Loader Validation",
        "",
        "This validation integrates BUS-UCLM as a second breast ultrasound dataset candidate. It does not train models.",
        "",
        "## Dataset Discovery",
        "",
        f"- Root: `{metadata['root']}`",
        f"- Raw image count: {metadata['raw_image_count']}",
        f"- Raw mask count: {metadata['raw_mask_count']}",
        f"- INFO.csv row count: {metadata['info_row_count']}",
        f"- Matched pairs: {metadata['matched_pair_count']}",
        f"- Missing images: {metadata['missing_image_count']}",
        f"- Missing masks: {metadata['missing_mask_count']}",
        f"- Duplicate stems: {metadata['duplicate_stem_count']}",
        f"- Included benign count: {metadata['included_counts'].get('Benign', 0)}",
        f"- Included malignant count: {metadata['included_counts'].get('Malignant', 0)}",
        f"- Deferred normal count: {metadata['deferred_counts'].get('Normal', 0)}",
        f"- Total included count: {metadata['included_total']}",
        f"- Included patient/study prefix count: {metadata['patient_prefix_count_included']}",
        "",
        "## Image/Mask Sanity",
        "",
        f"- Image modes: {metadata['image_modes']}",
        f"- Image sizes: {metadata['image_sizes']}",
        f"- Mask modes: {metadata['mask_modes']}",
        f"- Mask sizes: {metadata['mask_sizes']}",
        f"- Image/mask dimension or mode mismatches: {metadata['dimension_mismatch_count']}",
        f"- Unknown mask colors count: {metadata['unknown_mask_colors_count']}",
        f"- Empty lesion mask count among included benign/malignant samples: {metadata['empty_included_mask_count']}",
        "",
        "The known HESN mismatch samples are recorded in `phase5o_bus_uclm_warnings.csv`. The loader converts images through RGB and resizes image/mask pairs safely; mask resizing uses nearest-neighbor.",
        "",
        "## Split Counts",
        "",
    ]
    for split_name in SPLIT_NAMES:
        samples = split_samples[split_name]
        labels = Counter(sample.label for sample in samples)
        lines.append(
            f"- {split_name}: n={len(samples)}, benign={labels['Benign']}, malignant={labels['Malignant']}, prefixes={len({sample.patient_id for sample in samples})}"
        )
    lines.extend(
        [
            "",
            "## Leakage and Split Validation",
            "",
        ]
    )
    for row in validation_rows:
        lines.append(f"- {row['category']} / {row['check']}: {row['status']} ({row['value']})")
    lines.extend(
        [
            "",
            "## Tensor Loading Smoke Validation",
            "",
        ]
    )
    for row in tensor_rows[:9]:
        lines.append(
            f"- {row['split_mode']} {row['split']} client {row['client_id']}: image {row['image_shape']}, mask {row['mask_shape']}, passed={row['passed']}"
        )
    lines.extend(
        [
            "",
            "## Visualization Outputs",
            "",
        ]
    )
    for figure in figures:
        lines.append(f"- `{figure.relative_to(PROJECT_ROOT)}`")
    lines.extend(
        [
            "",
            "## Phase 5O Conclusion",
            "",
            f"- Loader validation status: {loader_status}.",
            f"- Recommend entering Phase 5P: {'yes' if phase5p_recommended else 'no'}.",
            f"- Phase 5P can perform BUSI vs BUS-UCLM ultrasound split validation: {'yes' if phase5p_recommended else 'no'}.",
            "- Normal remains deferred: yes.",
            f"- hard_noniid split design needs adjustment: {'yes' if hard_adjustment else 'no'}.",
            f"- Blocking issue exists: {'yes' if failed else 'no'}.",
        ]
    )
    (SUMMARY_DIR / "phase5o_bus_uclm_loader_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    samples, metadata = discover_bus_uclm_samples(BUS_UCLM_ROOT)
    split_samples = split_bus_uclm_samples(samples, seed=42)
    distribution_rows = split_label_counts(split_samples)
    for split_mode in SPLIT_MODES:
        distribution_rows.extend(client_rows_for_mode(split_samples, split_mode))
    validation_rows = build_validation_rows(samples, metadata, distribution_rows)
    tensor_rows = validate_tensor_loading()
    figures = make_visualizations(samples, metadata)
    warnings = warning_rows(metadata)

    write_csv(
        SUMMARY_DIR / "phase5o_bus_uclm_client_distribution.csv",
        distribution_rows,
        [
            "split_mode",
            "split",
            "client_id",
            "sample_count",
            "benign_count",
            "malignant_count",
            "malignant_ratio",
            "patient_prefix_count",
            "mean_mask_area_fraction",
            "median_mask_area_fraction",
            "min_mask_area_fraction",
            "max_mask_area_fraction",
            "empty_client",
        ],
    )
    write_csv(
        SUMMARY_DIR / "phase5o_bus_uclm_split_validation.csv",
        validation_rows + [
            {
                "category": "tensor",
                "check": f"{row['split_mode']}_{row['split']}_client_{row['client_id']}",
                "value": f"image={row['image_shape']}; mask={row['mask_shape']}; mask_range={row['mask_min']}-{row['mask_max']}",
                "status": "passed" if row["passed"] == "True" else "failed",
            }
            for row in tensor_rows
        ],
        ["category", "check", "value", "status"],
    )
    write_csv(SUMMARY_DIR / "phase5o_bus_uclm_warnings.csv", warnings, ["category", "sample_id", "message"])
    write_markdown(metadata, split_samples, distribution_rows, validation_rows, tensor_rows, figures)

    print(SUMMARY_DIR / "phase5o_bus_uclm_loader_validation.md")
    print(SUMMARY_DIR / "phase5o_bus_uclm_client_distribution.csv")
    print(SUMMARY_DIR / "phase5o_bus_uclm_split_validation.csv")
    print(SUMMARY_DIR / "phase5o_bus_uclm_warnings.csv")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
