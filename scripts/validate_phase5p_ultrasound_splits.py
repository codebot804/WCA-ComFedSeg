"""Validate Phase 5P ultrasound split design without model training."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import busi as busi_module  # noqa: E402
from datasets.bus_uclm import (  # noqa: E402
    BUS_UCLM_ROOT,
    build_bus_uclm_client_loaders,
    discover_bus_uclm_samples,
    leakage_report as bus_uclm_leakage_report,
    partition_bus_uclm_clients,
    split_bus_uclm_samples,
)
from datasets.busi import BUSI_ROOT, discover_busi_samples, split_busi_samples  # noqa: E402


SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
SPLIT_MODES = ["iid", "moderate_noniid", "hard_noniid"]
SPLIT_NAMES = ["train", "val", "test"]
DATASETS = ["BUSI", "BUS-UCLM"]


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def area_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def busi_normal_count(root: Path = BUSI_ROOT) -> int:
    normal_dir = Path(root) / "normal"
    if not normal_dir.exists():
        return 0
    return len([path for path in normal_dir.glob("*.png") if "_mask" not in path.stem])


def class_name(sample: Any) -> str:
    if hasattr(sample, "class_name"):
        return "Benign" if sample.class_name == "benign" else "Malignant"
    return sample.label


def sample_identity(sample: Any) -> str:
    return sample.sample_id


def patient_identity(sample: Any) -> str:
    return getattr(sample, "patient_id", "")


def mask_area(sample: Any) -> float:
    return float(sample.mask_area_fraction)


def busi_partitions(split_samples: dict[str, list], mode: str) -> dict[str, list[list[int]]]:
    partitions: dict[str, list[list[int]]] = {}
    for split_name, samples in split_samples.items():
        if mode == "iid":
            partitions[split_name] = busi_module._round_robin_partition(samples, 3)
        elif mode == "moderate_noniid":
            partitions[split_name] = busi_module._moderate_noniid_partition(samples, 3)
        else:
            partitions[split_name] = busi_module._hard_noniid_partition(samples, 3)
    return partitions


def busi_leakage(split_samples: dict[str, list], partitions: dict[str, list[list[int]]]) -> dict[str, Any]:
    split_ids = {split: {sample.sample_id for sample in samples} for split, samples in split_samples.items()}
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    split_overlap = {
        f"{left}_{right}": sorted(split_ids[left] & split_ids[right])
        for left, right in split_pairs
    }
    client_overlap: dict[str, list[str]] = {}
    for split_name, samples in split_samples.items():
        client_sets = [
            {samples[index].sample_id for index in indices}
            for indices in partitions[split_name]
        ]
        for left in range(len(client_sets)):
            for right in range(left + 1, len(client_sets)):
                client_overlap[f"{split_name}_client{left}_client{right}"] = sorted(
                    client_sets[left] & client_sets[right]
                )
    passed = not any(split_overlap.values()) and not any(client_overlap.values())
    return {"passed": passed, "split_overlap": split_overlap, "client_overlap": client_overlap}


def dataset_summary_row(dataset: str, samples: list, metadata: dict[str, Any]) -> dict[str, Any]:
    labels = Counter(class_name(sample) for sample in samples)
    areas = [mask_area(sample) for sample in samples]
    stats = area_summary(areas)
    if dataset == "BUSI":
        identity_count = "not_available"
        deferred_normal = busi_normal_count()
        image_sizes = "not_logged_by_loader"
        leakage_rule = "sample stem disjoint; no patient/study identifier available"
    else:
        identity_count = metadata.get("patient_prefix_count_included", 0)
        deferred_normal = metadata.get("deferred_counts", {}).get("Normal", 0)
        image_sizes = metadata.get("image_sizes", {})
        leakage_rule = "patient/study prefix disjoint across train/val/test and clients"
    total = len(samples)
    return {
        "dataset": dataset,
        "modality": "breast_ultrasound",
        "included_labels": "Benign;Malignant",
        "included_sample_count": total,
        "benign_count": labels["Benign"],
        "malignant_count": labels["Malignant"],
        "malignant_ratio": fmt(labels["Malignant"] / total) if total else "0",
        "deferred_normal_count": deferred_normal,
        "patient_study_identity_count": identity_count,
        "image_size_distribution": image_sizes,
        "mean_mask_area_fraction": fmt(stats["mean"]),
        "median_mask_area_fraction": fmt(stats["median"]),
        "min_mask_area_fraction": fmt(stats["min"]),
        "max_mask_area_fraction": fmt(stats["max"]),
        "leakage_identity_rule": leakage_rule,
        "notes": "Benign+Malignant only; Normal deferred.",
    }


def split_mode_partitions(dataset: str, split_samples: dict[str, list], mode: str) -> dict[str, list[list[int]]]:
    if dataset == "BUSI":
        return busi_partitions(split_samples, mode)
    return partition_bus_uclm_clients(split_samples, 3, mode)


def leakage_status(dataset: str, split_samples: dict[str, list], partitions: dict[str, list[list[int]]]) -> dict[str, Any]:
    if dataset == "BUSI":
        return busi_leakage(split_samples, partitions)
    return bus_uclm_leakage_report(split_samples, partitions)


def client_distribution_rows(dataset: str, split_samples: dict[str, list]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in SPLIT_MODES:
        partitions = split_mode_partitions(dataset, split_samples, mode)
        leakage = leakage_status(dataset, split_samples, partitions)
        for split_name in SPLIT_NAMES:
            samples = split_samples[split_name]
            split_client_rows = []
            for client_id, indices in enumerate(partitions[split_name]):
                labels = Counter(class_name(samples[index]) for index in indices)
                areas = [mask_area(samples[index]) for index in indices]
                stats = area_summary(areas)
                patients = {patient_identity(samples[index]) for index in indices if patient_identity(samples[index])}
                total = len(indices)
                split_client_rows.append(
                    {
                        "dataset": dataset,
                        "split_mode": mode,
                        "subset": split_name,
                        "client_id": client_id,
                        "sample_count": total,
                        "sample_count_imbalance_ratio": "",
                        "benign_count": labels["Benign"],
                        "malignant_count": labels["Malignant"],
                        "malignant_ratio": fmt(labels["Malignant"] / total) if total else "0",
                        "mean_mask_area_fraction": fmt(stats["mean"]),
                        "median_mask_area_fraction": fmt(stats["median"]),
                        "patient_study_identity_count": len(patients) if dataset == "BUS-UCLM" else "not_available",
                        "stem_leakage": not leakage["passed"],
                        "patient_leakage": (not leakage["passed"]) if dataset == "BUS-UCLM" else "not_available",
                        "empty_client": total == 0,
                    }
                )
            counts = [row["sample_count"] for row in split_client_rows]
            imbalance = max(counts) / max(min(counts), 1) if counts else 0.0
            for row in split_client_rows:
                row["sample_count_imbalance_ratio"] = fmt(imbalance)
            rows.extend(split_client_rows)
    return rows


def metric_ranges(rows: list[dict[str, Any]], dataset: str, mode: str, subset: str) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["dataset"] == dataset and row["split_mode"] == mode and row["subset"] == subset
    ]
    counts = [float(row["sample_count"]) for row in selected]
    ratios = [float(row["malignant_ratio"]) for row in selected]
    areas = [float(row["mean_mask_area_fraction"]) for row in selected]
    return {
        "sample_count_range": max(counts) - min(counts) if counts else 0.0,
        "sample_count_imbalance_ratio": max(counts) / max(min(counts), 1.0) if counts else 0.0,
        "malignant_ratio_range": max(ratios) - min(ratios) if ratios else 0.0,
        "mean_mask_area_range": max(areas) - min(areas) if areas else 0.0,
        "min_client_count": min(counts) if counts else 0.0,
        "empty_client_count": sum(1 for row in selected if str(row["empty_client"]) == "True"),
        "stem_leakage": any(str(row["stem_leakage"]) == "True" for row in selected),
        "patient_leakage": any(str(row["patient_leakage"]) == "True" for row in selected),
    }


def split_comparison_rows(dataset: str, distribution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_by_subset: dict[str, dict[str, float]] = {}
    for mode in SPLIT_MODES:
        for subset in SPLIT_NAMES:
            current = metric_ranges(distribution_rows, dataset, mode, subset)
            if mode == "iid":
                harder = "not_applicable"
            else:
                previous = previous_by_subset[subset]
                harder_bool = (
                    current["sample_count_range"] > previous["sample_count_range"]
                    or current["malignant_ratio_range"] > previous["malignant_ratio_range"]
                    or current["mean_mask_area_range"] > previous["mean_mask_area_range"]
                )
                harder = "yes" if harder_bool else "no"
            too_extreme = (
                current["empty_client_count"] > 0
                or (subset == "train" and current["min_client_count"] < 10)
                or (subset != "train" and current["min_client_count"] < 3)
            )
            if current["stem_leakage"] or current["patient_leakage"] or current["empty_client_count"]:
                status = "failed"
            elif mode != "iid" and harder == "no":
                status = "needs_adjustment"
            elif too_extreme:
                status = "needs_caution"
            else:
                status = "passed"
            rows.append(
                {
                    "dataset": dataset,
                    "split_mode": mode,
                    "subset": subset,
                    "sample_count_range": fmt(current["sample_count_range"]),
                    "sample_count_imbalance_ratio": fmt(current["sample_count_imbalance_ratio"]),
                    "malignant_ratio_range": fmt(current["malignant_ratio_range"]),
                    "mean_mask_area_range": fmt(current["mean_mask_area_range"]),
                    "min_client_count": fmt(current["min_client_count"]),
                    "empty_client_count": int(current["empty_client_count"]),
                    "stem_leakage": current["stem_leakage"],
                    "patient_leakage": current["patient_leakage"] if dataset == "BUS-UCLM" else "not_available",
                    "harder_than_previous": harder,
                    "too_extreme": too_extreme,
                    "status": status,
                    "notes": comparison_note(dataset, mode, subset, current, harder),
                }
            )
            previous_by_subset[subset] = current
    return rows


def comparison_note(dataset: str, mode: str, subset: str, metrics: dict[str, float], harder: str) -> str:
    if mode == "iid":
        if dataset == "BUS-UCLM" and subset != "train" and metrics["malignant_ratio_range"] > 0.20:
            return "Residual prefix-granularity imbalance remains in small validation/test subsets."
        return "Reference client split."
    if harder == "yes":
        return "More heterogeneous than the previous split mode by at least one tracked metric."
    return "Does not exceed previous split mode; inspect before using for claims."


def bus_uclm_warning_rows(
    metadata: dict[str, Any],
    distribution_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache = metadata.get("cache", {})
    rows.append(
        {
            "category": "discovery_cache",
            "sample_id": "",
            "message": (
                f"path={cache.get('path', '')}; hits={cache.get('hits', 0)}; "
                f"misses={cache.get('misses', 0)}; records={cache.get('sample_records', 0)}"
            ),
        }
    )
    for record in metadata.get("warnings", []):
        rows.append(
            {
                "category": record.get("category", ""),
                "sample_id": record.get("sample_id", ""),
                "message": record.get("message", ""),
            }
        )
    iid_train = metric_ranges(distribution_rows, "BUS-UCLM", "iid", "train")
    rows.append(
        {
            "category": "iid_balance",
            "sample_id": "",
            "message": (
                "BUS-UCLM iid train after Phase 5P: "
                f"count_range={fmt(iid_train['sample_count_range'])}, "
                f"malignant_ratio_range={fmt(iid_train['malignant_ratio_range'])}, "
                f"area_range={fmt(iid_train['mean_mask_area_range'])}."
            ),
        }
    )
    for row in comparison_rows:
        if row["dataset"] == "BUS-UCLM" and row["status"] != "passed":
            rows.append(
                {
                    "category": "split_validation",
                    "sample_id": "",
                    "message": (
                        f"{row['split_mode']} {row['subset']} status={row['status']}; "
                        f"harder_than_previous={row['harder_than_previous']}; notes={row['notes']}"
                    ),
                }
            )
    if len(rows) == 1:
        rows.append({"category": "none", "sample_id": "", "message": "No BUS-UCLM warnings."})
    return rows


def validate_bus_uclm_tensor_compatibility() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in SPLIT_MODES:
        loaders, metadata = build_bus_uclm_client_loaders(
            num_clients=3,
            batch_size=4,
            split_mode=mode,
            image_size=128,
            seed=42,
        )
        for client_id, split_loaders in loaders.items():
            for split_name, loader in split_loaders.items():
                image, mask = next(iter(loader))
                rows.append(
                    {
                        "split_mode": mode,
                        "subset": split_name,
                        "client_id": client_id,
                        "image_shape": "x".join(str(dim) for dim in image.shape),
                        "mask_shape": "x".join(str(dim) for dim in mask.shape),
                        "mask_min": fmt(float(mask.min())),
                        "mask_max": fmt(float(mask.max())),
                        "leakage_passed": metadata["leakage_check"]["passed"],
                        "passed": tuple(image.shape[1:]) == (1, 128, 128)
                        and tuple(mask.shape[1:]) == (1, 128, 128)
                        and metadata["leakage_check"]["passed"],
                    }
                )
    return rows


def write_markdown(
    dataset_rows: list[dict[str, Any]],
    busi_comparison: list[dict[str, Any]],
    bus_uclm_comparison: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
) -> None:
    bus_uclm_iid = metric_ranges(distribution_rows, "BUS-UCLM", "iid", "train")
    hard_checks = [
        row for row in bus_uclm_comparison
        if row["split_mode"] == "hard_noniid" and row["subset"] in SPLIT_NAMES
    ]
    hard_all_pass = all(row["harder_than_previous"] == "yes" and row["status"] == "passed" for row in hard_checks)
    leak_ok = all(str(row["stem_leakage"]) == "False" for row in distribution_rows)
    bus_uclm_leak_ok = all(
        str(row["patient_leakage"]) in {"False", "not_available"}
        for row in distribution_rows
        if row["dataset"] == "BUS-UCLM"
    )
    empty_count = sum(1 for row in distribution_rows if str(row["empty_client"]) == "True")
    compat_ok = all(row["passed"] for row in compatibility_rows)
    phase_status = "passed" if hard_all_pass and leak_ok and bus_uclm_leak_ok and not empty_count and compat_ok else "needs caution"

    lines = [
        "# Phase 5P Ultrasound Split Validation",
        "",
        "This phase validates ultrasound dataset positioning and split design only. No model training, optimizer changes, or federated training logic changes were run.",
        "",
        "## Status",
        "",
        f"- Phase 5P validation status: {phase_status}.",
        "- BUSI scope: benign and malignant only; normal remains deferred.",
        "- BUS-UCLM scope: Benign and Malignant only; Normal remains deferred.",
        f"- BUS-UCLM loader tensor compatibility after split refinement: {'passed' if compat_ok else 'failed'}.",
        f"- Leakage status: {'passed' if leak_ok and bus_uclm_leak_ok else 'failed'}.",
        f"- Empty client count: {empty_count}.",
        "",
        "## Dataset Summary",
        "",
    ]
    for row in dataset_rows:
        lines.append(
            f"- {row['dataset']}: n={row['included_sample_count']}, benign={row['benign_count']}, "
            f"malignant={row['malignant_count']}, normal_deferred={row['deferred_normal_count']}, "
            f"mean_mask_area={row['mean_mask_area_fraction']}."
        )
    lines.extend(
        [
            "",
            "## BUS-UCLM Phase 5P Refinement",
            "",
            "- Discovery now records/reuses a Phase 5P cache for image/mask stats, mask colors, and mask area fractions. This does not change sample inclusion, mask semantics, or loader tensor output.",
            "- IID client partition now optimizes patient-prefix assignment using sample count, malignant ratio, and mask area instead of count-only assignment.",
            "- moderate_noniid and hard_noniid client partition now use stronger target separation for hard_noniid and exhaustive assignment for small prefix sets where feasible.",
            f"- BUS-UCLM iid train count range: {fmt(bus_uclm_iid['sample_count_range'])}; malignant ratio range: {fmt(bus_uclm_iid['malignant_ratio_range'])}; mask area range: {fmt(bus_uclm_iid['mean_mask_area_range'])}.",
            "",
            "## Hard-vs-Moderate Check",
            "",
        ]
    )
    for row in hard_checks:
        lines.append(
            f"- BUS-UCLM hard_noniid {row['subset']}: harder_than_moderate={row['harder_than_previous']}; "
            f"status={row['status']}; count_imbalance={row['sample_count_imbalance_ratio']}; "
            f"malignant_ratio_range={row['malignant_ratio_range']}."
        )
    lines.extend(
        [
            "",
            "## BUSI Position",
            "",
            "- BUSI remains the main ultrasound dataset for the current paper line.",
            "- BUSI split validation is retained as a reference because its benign/malignant sample-level split has no patient/study identifier available in the loader.",
            "- BUSI hard_noniid remains a boundary/failure-risk setting rather than the primary success setting.",
            "",
            "## BUS-UCLM Position",
            "",
            "- BUS-UCLM is feasible as a second breast ultrasound dataset candidate after Phase 5P split refinement.",
            "- Its patient/study-prefix split should be kept for leakage control.",
            "- Because validation/test subsets have few prefixes, residual client imbalance should be reported instead of hidden.",
            "",
            "## Warning Summary",
            "",
        ]
    )
    for row in warnings[:12]:
        lines.append(f"- {row['category']}: {row['message']}")
    lines.extend(
        [
            "",
            "## Recommended Next Phase",
            "",
            "- Recommend Phase 5Q: BUS-UCLM no-training smoke and small pilot planning, then explicitly approved real training only after the split files are accepted.",
            "- Phase 5Q should not include Normal, foundation models, or historical failed variants.",
        ]
    )
    (SUMMARY_DIR / "phase5p_ultrasound_split_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    busi_samples, busi_metadata = discover_busi_samples(BUSI_ROOT)
    bus_uclm_samples, bus_uclm_metadata = discover_bus_uclm_samples(BUS_UCLM_ROOT)

    busi_splits = split_busi_samples(busi_samples, seed=42)
    bus_uclm_splits = split_bus_uclm_samples(bus_uclm_samples, seed=42)

    dataset_rows = [
        dataset_summary_row("BUSI", busi_samples, busi_metadata),
        dataset_summary_row("BUS-UCLM", bus_uclm_samples, bus_uclm_metadata),
    ]
    distribution_rows = client_distribution_rows("BUSI", busi_splits)
    distribution_rows.extend(client_distribution_rows("BUS-UCLM", bus_uclm_splits))

    busi_comparison = split_comparison_rows("BUSI", distribution_rows)
    bus_uclm_comparison = split_comparison_rows("BUS-UCLM", distribution_rows)
    bus_uclm_warnings = bus_uclm_warning_rows(bus_uclm_metadata, distribution_rows, bus_uclm_comparison)
    compatibility_rows = validate_bus_uclm_tensor_compatibility()

    write_csv(
        SUMMARY_DIR / "phase5p_busi_vs_bus_uclm_dataset_summary.csv",
        dataset_rows,
        [
            "dataset",
            "modality",
            "included_labels",
            "included_sample_count",
            "benign_count",
            "malignant_count",
            "malignant_ratio",
            "deferred_normal_count",
            "patient_study_identity_count",
            "image_size_distribution",
            "mean_mask_area_fraction",
            "median_mask_area_fraction",
            "min_mask_area_fraction",
            "max_mask_area_fraction",
            "leakage_identity_rule",
            "notes",
        ],
    )
    write_csv(
        SUMMARY_DIR / "phase5p_ultrasound_client_distribution.csv",
        distribution_rows,
        [
            "dataset",
            "split_mode",
            "subset",
            "client_id",
            "sample_count",
            "sample_count_imbalance_ratio",
            "benign_count",
            "malignant_count",
            "malignant_ratio",
            "mean_mask_area_fraction",
            "median_mask_area_fraction",
            "patient_study_identity_count",
            "stem_leakage",
            "patient_leakage",
            "empty_client",
        ],
    )
    comparison_fields = [
        "dataset",
        "split_mode",
        "subset",
        "sample_count_range",
        "sample_count_imbalance_ratio",
        "malignant_ratio_range",
        "mean_mask_area_range",
        "min_client_count",
        "empty_client_count",
        "stem_leakage",
        "patient_leakage",
        "harder_than_previous",
        "too_extreme",
        "status",
        "notes",
    ]
    write_csv(SUMMARY_DIR / "phase5p_busi_split_comparison.csv", busi_comparison, comparison_fields)
    write_csv(SUMMARY_DIR / "phase5p_bus_uclm_split_comparison.csv", bus_uclm_comparison, comparison_fields)
    write_csv(
        SUMMARY_DIR / "phase5p_bus_uclm_warnings.csv",
        bus_uclm_warnings,
        ["category", "sample_id", "message"],
    )
    write_markdown(
        dataset_rows,
        busi_comparison,
        bus_uclm_comparison,
        distribution_rows,
        bus_uclm_warnings,
        compatibility_rows,
    )

    print(SUMMARY_DIR / "phase5p_ultrasound_split_validation.md")
    print(SUMMARY_DIR / "phase5p_busi_vs_bus_uclm_dataset_summary.csv")
    print(SUMMARY_DIR / "phase5p_bus_uclm_split_comparison.csv")
    print(SUMMARY_DIR / "phase5p_busi_split_comparison.csv")
    print(SUMMARY_DIR / "phase5p_ultrasound_client_distribution.csv")
    print(SUMMARY_DIR / "phase5p_bus_uclm_warnings.csv")


if __name__ == "__main__":
    main()
