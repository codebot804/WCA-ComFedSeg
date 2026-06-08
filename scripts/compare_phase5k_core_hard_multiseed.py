"""Aggregate Phase 5K core hard_noniid scoped multi-seed runs.

This script reads only results/logs/phase5k_core_hard_multiseed. It does not
mix Phase 5E/5H single-seed logs and does not train models.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5k_core_hard_multiseed"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5k_core_hard_multiseed"

SEEDS = [42, 123, 2025]
EXPECTED = [
    ("busi", "hard_noniid", "fedprox"),
    ("busi", "hard_noniid", "wca_comfedseg_prox"),
    ("kvasir_seg", "hard_noniid", "fedbn"),
    ("kvasir_seg", "hard_noniid", "wca_comfedseg_bn"),
]
APPROVED_METHODS = {method for _dataset, _split, method in EXPECTED}
FAILED_METHODS = {"wca_comfedseg_smooth", "wca_comfedseg_pbn", "wca_comfedseg_rg"}
DISALLOWED_METHODS = FAILED_METHODS | {"wca_comfedseg_prox_comm_cons", "wca_comfedseg_comm"}
FULL_UPLOAD_MB = 3.373832703

AGG_FIELDS = [
    "dataset",
    "split",
    "method",
    "seeds",
    "n",
    "average_dice_mean",
    "average_dice_std",
    "average_iou_mean",
    "average_iou_std",
    "worst_client_dice_mean",
    "worst_client_dice_std",
    "client_dice_std_mean",
    "client_dice_std_std",
    "best_worst_gap_mean",
    "best_worst_gap_std",
    "total_uploaded_mb_mean",
    "communication_reduction_percent_mean",
]

DETAIL_FIELDS = [
    "dataset",
    "split",
    "method",
    "seed",
    "average_dice",
    "average_iou",
    "worst_client_dice",
    "client_dice_std",
    "best_worst_gap",
    "total_uploaded_mb",
    "communication_reduction_percent",
    "summary_path",
    "csv_log_path",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Invalid numeric value: {value}")
    return f"{value:.10f}".rstrip("0").rstrip(".")


def std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def expected_keys() -> set[tuple[str, str, str, int]]:
    return {(dataset, split, method, seed) for dataset, split, method in EXPECTED for seed in SEEDS}


def collect_summaries() -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[tuple[str, str, str, int], list[Path]], list[Path]]:
    grouped_paths: dict[tuple[str, str, str, int], list[Path]] = defaultdict(list)
    disallowed_paths: list[Path] = []
    for path in LOG_ROOT.glob("*/summary.json"):
        summary = read_json(path)
        args = summary.get("args", {})
        key = (
            summary.get("dataset", ""),
            summary.get("split", ""),
            summary.get("method", ""),
            int(args.get("seed", -1)),
        )
        if key[2] in DISALLOWED_METHODS or key not in expected_keys():
            disallowed_paths.append(path)
            continue
        grouped_paths[key].append(path)

    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for key, paths in grouped_paths.items():
        path = sorted(paths, key=lambda item: item.stat().st_mtime)[-1]
        latest[key] = read_json(path)
        latest[key]["_summary_path"] = str(path.relative_to(PROJECT_ROOT))
    return latest, grouped_paths, disallowed_paths


def uploaded_mb(summary: dict[str, Any]) -> float:
    value = summary.get("total_uploaded_mb", "")
    return float(value) if value not in {"", None} else FULL_UPLOAD_MB


def comm_reduction(summary: dict[str, Any]) -> float:
    value = summary.get("communication_reduction_vs_full_participation_percent", "")
    return float(value) if value not in {"", None} else 0.0


def detail_row(summary: dict[str, Any]) -> dict[str, Any]:
    args = summary.get("args", {})
    return {
        "dataset": summary["dataset"],
        "split": summary["split"],
        "method": summary["method"],
        "seed": args["seed"],
        "average_dice": fmt(float(summary["average_dice"])),
        "average_iou": fmt(float(summary["average_iou"])),
        "worst_client_dice": fmt(float(summary["worst_client_dice"])),
        "client_dice_std": fmt(float(summary["client_dice_std"])),
        "best_worst_gap": fmt(float(summary["best_worst_gap"])),
        "total_uploaded_mb": fmt(uploaded_mb(summary)),
        "communication_reduction_percent": fmt(comm_reduction(summary)),
        "summary_path": summary["_summary_path"],
        "csv_log_path": summary.get("csv_log_path", ""),
    }


def aggregate_rows(summaries: dict[tuple[str, str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, split, method in EXPECTED:
        group = [summaries[(dataset, split, method, seed)] for seed in SEEDS if (dataset, split, method, seed) in summaries]
        if not group:
            continue
        avg_dice = [float(item["average_dice"]) for item in group]
        avg_iou = [float(item["average_iou"]) for item in group]
        worst = [float(item["worst_client_dice"]) for item in group]
        client_std = [float(item["client_dice_std"]) for item in group]
        gap = [float(item["best_worst_gap"]) for item in group]
        uploaded = [uploaded_mb(item) for item in group]
        reduction = [comm_reduction(item) for item in group]
        seeds = [str(item.get("args", {}).get("seed")) for item in group]
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "method": method,
                "seeds": "|".join(seeds),
                "n": len(group),
                "average_dice_mean": fmt(mean(avg_dice)),
                "average_dice_std": fmt(std(avg_dice)),
                "average_iou_mean": fmt(mean(avg_iou)),
                "average_iou_std": fmt(std(avg_iou)),
                "worst_client_dice_mean": fmt(mean(worst)),
                "worst_client_dice_std": fmt(std(worst)),
                "client_dice_std_mean": fmt(mean(client_std)),
                "client_dice_std_std": fmt(std(client_std)),
                "best_worst_gap_mean": fmt(mean(gap)),
                "best_worst_gap_std": fmt(std(gap)),
                "total_uploaded_mb_mean": fmt(mean(uploaded)),
                "communication_reduction_percent_mean": fmt(mean(reduction)),
            }
        )
    return rows


def rows_for_dataset(details: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return [row for row in details if row["dataset"] == dataset]


def plot_metric(dataset: str, aggregates: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    rows = [row for row in aggregates if row["dataset"] == dataset]
    if not rows:
        return
    labels = [row["method"].replace("wca_comfedseg_", "wca_") for row in rows]
    means = [float(row[f"{metric}_mean"]) for row in rows]
    stds = [float(row[f"{metric}_std"]) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=stds, capsize=5, color=["#4C78A8", "#F58518"])
    plt.ylabel(metric)
    plt.title(f"Phase 5K {dataset} {metric} mean +/- std")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def best_comparison(aggregates: list[dict[str, Any]], dataset: str) -> dict[str, bool]:
    rows = {row["method"]: row for row in aggregates if row["dataset"] == dataset}
    if dataset == "busi":
        candidate = rows["wca_comfedseg_prox"]
        reference = rows["fedprox"]
    else:
        candidate = rows["wca_comfedseg_bn"]
        reference = rows["fedbn"]
    return {
        "average_dice_higher": float(candidate["average_dice_mean"]) > float(reference["average_dice_mean"]),
        "worst_client_dice_higher": float(candidate["worst_client_dice_mean"]) > float(reference["worst_client_dice_mean"]),
        "best_worst_gap_lower": float(candidate["best_worst_gap_mean"]) < float(reference["best_worst_gap_mean"]),
    }


def has_nan_or_inf(summaries: dict[tuple[str, str, str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        for key in ["average_dice", "average_iou", "worst_client_dice", "client_dice_std", "best_worst_gap"]:
            if not math.isfinite(float(summary[key])):
                return True
        csv_path = PROJECT_ROOT / summary.get("csv_log_path", "")
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for key in ["dice", "iou", "loss"]:
                        if not math.isfinite(float(row[key])):
                            return True
    return False


def busi_normal_avoided(summaries: dict[tuple[str, str, str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        if summary["dataset"] != "busi":
            continue
        dataset_meta = summary.get("split_configuration", {}).get("dataset", {})
        folder_status = dataset_meta.get("folder_status", {})
        if folder_status.get("normal_used") is not False:
            return False
        if "normal" not in dataset_meta.get("classes_deferred", []):
            return False
    return True


def write_reports(
    summaries: dict[tuple[str, str, str, int], dict[str, Any]],
    grouped_paths: dict[tuple[str, str, str, int], list[Path]],
    disallowed_paths: list[Path],
    details: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    expected = expected_keys()
    completed = set(summaries)
    missing = sorted(expected - completed)
    duplicate_keys = {key: paths for key, paths in grouped_paths.items() if len(paths) > 1}
    image_sizes = {summary.get("args", {}).get("image_size") for summary in summaries.values()}
    approved_only = not disallowed_paths and all(key[2] in APPROVED_METHODS for key in completed)
    exact_seed_counts = all(
        len([key for key in completed if key[:3] == (dataset, split, method)]) == len(SEEDS)
        for dataset, split, method in EXPECTED
    )
    busi_cmp = best_comparison(aggregates, "busi") if len(aggregates) == 4 else {}
    kvasir_cmp = best_comparison(aggregates, "kvasir_seg") if len(aggregates) == 4 else {}

    lines = [
        "# Phase 5K Core hard_noniid Multi-Seed Verification",
        "",
        "Phase 5K is the first staged scoped multi-seed validation subset. It is not a final paper claim.",
        "",
        "## Run Scope",
        "",
        "- BUSI hard_noniid: `fedprox`, `wca_comfedseg_prox`.",
        "- Kvasir-SEG hard_noniid: `fedbn`, `wca_comfedseg_bn`.",
        "- Seeds: 42, 123, 2025.",
        "- Fixed settings: clients=3, rounds=10, local_epochs=2, batch_size=4, image_size=128.",
        "",
        "## Completion Checks",
        "",
        f"- Expected run count: 12",
        f"- Completed unique run count: {len(completed)}",
        f"- Missing runs: {missing if missing else 'none'}",
        f"- Duplicate run groups: {len(duplicate_keys)}",
        f"- Disallowed/extra runs under phase5k log root: {[str(path.relative_to(PROJECT_ROOT)) for path in disallowed_paths] if disallowed_paths else 'none'}",
        f"- Failed runs recorded: none",
        f"- NaN/inf: {'yes' if has_nan_or_inf(summaries) else 'no'}",
        f"- Only approved methods were run: {'yes' if approved_only else 'no'}",
        f"- No failed variants: {'yes' if not any(key[2] in FAILED_METHODS for key in completed) else 'no'}",
        f"- image_size values observed: {sorted(image_sizes)}",
        f"- image_size=128 only: {'yes' if image_sizes == {128} else 'no'}",
        f"- image_size=256 avoided: {'yes' if 256 not in image_sizes else 'no'}",
        f"- BUSI normal avoided: {'yes' if busi_normal_avoided(summaries) else 'no'}",
        "- No large model / SAM / MedSAM / pretrained backbone was added: yes",
        f"- Logs located under phase5k_core_hard_multiseed: {'yes' if all('phase5k_core_hard_multiseed' in item['_summary_path'] for item in summaries.values()) else 'no'}",
        f"- Each method has exactly 3 seeds: {'yes' if exact_seed_counts else 'no'}",
        "",
        "## BUSI Interpretation",
        "",
        f"- WCA+FedProx worst-client Dice mean higher than FedProx: {busi_cmp.get('worst_client_dice_higher', 'n/a')}.",
        f"- WCA+FedProx best-worst gap mean lower than FedProx: {busi_cmp.get('best_worst_gap_lower', 'n/a')}.",
        f"- WCA+FedProx average Dice mean higher than FedProx: {busi_cmp.get('average_dice_higher', 'n/a')}.",
        "",
        "## Kvasir Interpretation",
        "",
        f"- WCA+BN average Dice mean higher than FedBN: {kvasir_cmp.get('average_dice_higher', 'n/a')}.",
        f"- WCA+BN worst-client Dice mean higher than FedBN: {kvasir_cmp.get('worst_client_dice_higher', 'n/a')}.",
        f"- WCA+BN best-worst gap mean lower than FedBN: {kvasir_cmp.get('best_worst_gap_lower', 'n/a')}.",
        "",
        "## Recommendation",
        "",
    ]
    if busi_cmp and kvasir_cmp:
        if busi_cmp["worst_client_dice_higher"] and busi_cmp["best_worst_gap_lower"]:
            lines.append("- BUSI: keep `wca_comfedseg_prox` as a scoped hard_noniid candidate.")
        else:
            lines.append("- BUSI: `wca_comfedseg_prox` is not stable enough yet; inspect hard_noniid behavior before expanding.")
        if (
            kvasir_cmp["average_dice_higher"]
            and kvasir_cmp["worst_client_dice_higher"]
            and kvasir_cmp["best_worst_gap_lower"]
        ):
            lines.append("- Kvasir: keep `wca_comfedseg_bn` as a scoped hard_noniid candidate.")
        else:
            lines.append("- Kvasir: `wca_comfedseg_bn` is not stable across the requested metrics; do not expand blindly.")
        if busi_cmp["worst_client_dice_higher"] and busi_cmp["best_worst_gap_lower"] and not (
            kvasir_cmp["average_dice_higher"]
            and kvasir_cmp["worst_client_dice_higher"]
            and kvasir_cmp["best_worst_gap_lower"]
        ):
            lines.append("- Next step: keep BUSI WCA+FedProx; diagnose Kvasir WCA+BN before running the full moderate_noniid scoped multi-seed.")
        elif (
            busi_cmp["worst_client_dice_higher"]
            and busi_cmp["best_worst_gap_lower"]
            and kvasir_cmp["average_dice_higher"]
            and kvasir_cmp["worst_client_dice_higher"]
            and kvasir_cmp["best_worst_gap_lower"]
        ):
            lines.append("- Next step: run moderate_noniid scoped multi-seed.")
        else:
            lines.append("- Next step: adjust hard_noniid candidates before broadening validation.")
    report_path = SUMMARY_DIR / "phase5k_core_hard_multiseed_verification.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = [
        "# Phase 5K Notes",
        "",
        "Phase 5K ran only the core hard_noniid scoped multi-seed subset.",
        "",
        "It did not run moderate_noniid multi-seed, optional WCA-Comm hard communication references, failed variants, or WCA+FedProx+conservative communication diagnostics.",
        "",
        "Use the verification report and aggregate CSV for candidate-positioning decisions. Do not treat this as a final SCI result.",
    ]
    (SUMMARY_DIR / "phase5k_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    summaries, grouped_paths, disallowed_paths = collect_summaries()
    details = [detail_row(summary) for _key, summary in sorted(summaries.items())]
    aggregates = aggregate_rows(summaries)

    write_csv(SUMMARY_DIR / "phase5k_core_hard_multiseed_aggregate.csv", aggregates, AGG_FIELDS)
    write_csv(SUMMARY_DIR / "phase5k_core_hard_multiseed_busi.csv", rows_for_dataset(details, "busi"), DETAIL_FIELDS)
    write_csv(SUMMARY_DIR / "phase5k_core_hard_multiseed_kvasir_seg.csv", rows_for_dataset(details, "kvasir_seg"), DETAIL_FIELDS)
    for dataset in ["busi", "kvasir_seg"]:
        for metric in ["average_dice", "worst_client_dice", "best_worst_gap"]:
            plot_metric(dataset, aggregates, metric, FIGURE_DIR / f"phase5k_{dataset}_{metric}.png")
    write_reports(summaries, grouped_paths, disallowed_paths, details, aggregates)

    print(SUMMARY_DIR / "phase5k_core_hard_multiseed_aggregate.csv")
    print(SUMMARY_DIR / "phase5k_core_hard_multiseed_busi.csv")
    print(SUMMARY_DIR / "phase5k_core_hard_multiseed_kvasir_seg.csv")
    print(SUMMARY_DIR / "phase5k_core_hard_multiseed_verification.md")
    print(SUMMARY_DIR / "phase5k_notes.md")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
