"""Aggregate Phase 5M moderate_noniid scoped multi-seed runs.

This script reads only results/logs/phase5m_moderate_multiseed. It does not
train models and does not mix Phase 5C single-seed reference logs.
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
LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5m_moderate_multiseed"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5m_moderate_multiseed"

SEEDS = [42, 123, 2025]
DATASETS = ["busi", "kvasir_seg"]
SPLIT = "moderate_noniid"
METHODS = ["fedavg", "fedprox", "fedbn", "wca_comfedseg_comm"]
EXPECTED = [(dataset, SPLIT, method) for dataset in DATASETS for method in METHODS]
APPROVED_METHODS = set(METHODS)
FAILED_METHODS = {"wca_comfedseg_smooth", "wca_comfedseg_pbn", "wca_comfedseg_rg"}
DISALLOWED_METHODS = FAILED_METHODS | {
    "wca_comfedseg_prox",
    "wca_comfedseg_bn",
    "wca_comfedseg_prox_comm_cons",
    "local",
    "centralized",
}
DEFAULT_FULL_UPLOAD_MB = 3.373832703

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
    "communication_log_path",
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


def fmt(value: float | int | str) -> str:
    if value == "":
        return ""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Invalid numeric value: {value}")
    return f"{numeric:.10f}".rstrip("0").rstrip(".")


def std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def expected_keys() -> set[tuple[str, str, str, int]]:
    return {(dataset, split, method, seed) for dataset, split, method in EXPECTED for seed in SEEDS}


def collect_summaries() -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]],
    dict[tuple[str, str, str, int], list[Path]],
    list[Path],
    list[Path],
]:
    grouped_paths: dict[tuple[str, str, str, int], list[Path]] = defaultdict(list)
    disallowed_paths: list[Path] = []
    incomplete_dirs: list[Path] = []
    if not LOG_ROOT.exists():
        return {}, grouped_paths, disallowed_paths, incomplete_dirs

    for run_dir in sorted(path for path in LOG_ROOT.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            incomplete_dirs.append(run_dir)
            continue
        summary = read_json(summary_path)
        args = summary.get("args", {})
        key = (
            summary.get("dataset", ""),
            summary.get("split", ""),
            summary.get("method", ""),
            int(args.get("seed", -1)),
        )
        if key[2] in DISALLOWED_METHODS or key not in expected_keys():
            disallowed_paths.append(summary_path)
            continue
        grouped_paths[key].append(summary_path)

    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for key, paths in grouped_paths.items():
        path = sorted(paths, key=lambda item: item.stat().st_mtime)[-1]
        latest[key] = read_json(path)
        latest[key]["_summary_path"] = str(path.relative_to(PROJECT_ROOT))
        latest[key]["_run_dir"] = str(path.parent.relative_to(PROJECT_ROOT))
    return latest, grouped_paths, disallowed_paths, incomplete_dirs


def full_upload_reference(summary: dict[str, Any] | None = None) -> float:
    if summary is None:
        return DEFAULT_FULL_UPLOAD_MB
    value = summary.get("full_participation_uploaded_mb", "")
    return float(value) if value not in {"", None} else DEFAULT_FULL_UPLOAD_MB


def uploaded_mb(summary: dict[str, Any]) -> float:
    value = summary.get("total_uploaded_mb", "")
    return float(value) if value not in {"", None} else full_upload_reference(summary)


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
        "average_dice": fmt(summary["average_dice"]),
        "average_iou": fmt(summary["average_iou"]),
        "worst_client_dice": fmt(summary["worst_client_dice"]),
        "client_dice_std": fmt(summary["client_dice_std"]),
        "best_worst_gap": fmt(summary["best_worst_gap"]),
        "total_uploaded_mb": fmt(uploaded_mb(summary)),
        "communication_reduction_percent": fmt(comm_reduction(summary)),
        "summary_path": summary["_summary_path"],
        "csv_log_path": summary.get("csv_log_path", ""),
        "communication_log_path": summary.get("communication_log_path", ""),
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


def plot_mean_std(dataset: str, aggregates: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    rows = [row for row in aggregates if row["dataset"] == dataset]
    labels = [row["method"].replace("wca_comfedseg_comm", "wca_comm") for row in rows]
    means = [float(row[f"{metric}_mean"]) for row in rows]
    stds = [float(row.get(f"{metric}_std", "0") or 0.0) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, means, yerr=stds, capsize=5, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Phase 5M {dataset} {metric} mean +/- std")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mean_only(dataset: str, aggregates: list[dict[str, Any]], metric: str, output_path: Path) -> None:
    rows = [row for row in aggregates if row["dataset"] == dataset]
    labels = [row["method"].replace("wca_comfedseg_comm", "wca_comm") for row in rows]
    values = [float(row[f"{metric}_mean"]) for row in rows]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Phase 5M {dataset} {metric}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_figures(aggregates: list[dict[str, Any]]) -> None:
    for dataset in DATASETS:
        plot_mean_std(dataset, aggregates, "average_dice", FIGURE_DIR / f"phase5m_{dataset}_average_dice.png")
        plot_mean_std(dataset, aggregates, "worst_client_dice", FIGURE_DIR / f"phase5m_{dataset}_worst_client_dice.png")
        plot_mean_std(dataset, aggregates, "best_worst_gap", FIGURE_DIR / f"phase5m_{dataset}_best_worst_gap.png")
        plot_mean_only(dataset, aggregates, "total_uploaded_mb", FIGURE_DIR / f"phase5m_{dataset}_uploaded_mb.png")
        plot_mean_only(
            dataset,
            aggregates,
            "communication_reduction_percent",
            FIGURE_DIR / f"phase5m_{dataset}_communication_reduction.png",
        )


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


def wca_comm_logs_ok(summaries: dict[tuple[str, str, str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        if summary["method"] != "wca_comfedseg_comm":
            continue
        log_path = summary.get("communication_log_path", "")
        if not log_path or not (PROJECT_ROOT / log_path).exists():
            return False
        if not math.isfinite(comm_reduction(summary)) or comm_reduction(summary) <= 0.0:
            return False
    return True


def method_rows(aggregates: list[dict[str, Any]], dataset: str) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in aggregates if row["dataset"] == dataset}


def best_method(rows: dict[str, dict[str, Any]], metric: str, higher_is_better: bool = True) -> str:
    return sorted(rows, key=lambda method: float(rows[method][f"{metric}_mean"]), reverse=higher_is_better)[0]


def competitive(value: float, best: float, tolerance: float = 0.02) -> bool:
    return value >= best - tolerance


def dataset_interpretation(aggregates: list[dict[str, Any]], dataset: str) -> list[str]:
    rows = method_rows(aggregates, dataset)
    wca = rows["wca_comfedseg_comm"]
    best_avg = best_method(rows, "average_dice")
    best_worst = best_method(rows, "worst_client_dice")
    best_gap = best_method(rows, "best_worst_gap", higher_is_better=False)
    wca_avg = float(wca["average_dice_mean"])
    wca_worst = float(wca["worst_client_dice_mean"])
    wca_gap = float(wca["best_worst_gap_mean"])
    return [
        f"- WCA-Comm communication reduction mean: {float(wca['communication_reduction_percent_mean']):.2f}%.",
        (
            f"- Average Dice: WCA-Comm={wca_avg:.4f}; best={best_avg} "
            f"({float(rows[best_avg]['average_dice_mean']):.4f}); competitive={competitive(wca_avg, float(rows[best_avg]['average_dice_mean']))}."
        ),
        (
            f"- Worst-client Dice: WCA-Comm={wca_worst:.4f}; best={best_worst} "
            f"({float(rows[best_worst]['worst_client_dice_mean']):.4f}); competitive={competitive(wca_worst, float(rows[best_worst]['worst_client_dice_mean']))}."
        ),
        (
            f"- Best-worst gap: WCA-Comm={wca_gap:.4f}; best={best_gap} "
            f"({float(rows[best_gap]['best_worst_gap_mean']):.4f}); competitive={wca_gap <= float(rows[best_gap]['best_worst_gap_mean']) + 0.02}."
        ),
    ]


def write_reports(
    summaries: dict[tuple[str, str, str, int], dict[str, Any]],
    grouped_paths: dict[tuple[str, str, str, int], list[Path]],
    disallowed_paths: list[Path],
    incomplete_dirs: list[Path],
    aggregates: list[dict[str, Any]],
) -> None:
    expected = expected_keys()
    completed = set(summaries)
    missing = sorted(expected - completed)
    duplicate_keys = {key: paths for key, paths in grouped_paths.items() if len(paths) > 1}
    image_sizes = {int(summary.get("args", {}).get("image_size")) for summary in summaries.values()}
    splits = {summary.get("split") for summary in summaries.values()}
    observed_methods = {summary.get("method") for summary in summaries.values()}
    approved_only = not disallowed_paths and observed_methods <= APPROVED_METHODS
    exact_seed_counts = all(
        len([key for key in completed if key[:3] == (dataset, split, method)]) == len(SEEDS)
        for dataset, split, method in EXPECTED
    )
    base_channels = {int(summary.get("args", {}).get("base_channels", 0)) for summary in summaries.values()}
    no_large_model = base_channels == {8}
    wca_logs_ok = wca_comm_logs_ok(summaries)
    no_nan_inf = not has_nan_or_inf(summaries)

    lines = [
        "# Phase 5M Moderate Non-IID Multi-Seed Verification",
        "",
        "Phase 5M is scoped real-dataset validation for moderate_noniid. It is not a final SCI claim.",
        "",
        "## Run Scope",
        "",
        "- Datasets: `busi`, `kvasir_seg`.",
        "- Split: `moderate_noniid` only.",
        "- Methods: `fedavg`, `fedprox`, `fedbn`, `wca_comfedseg_comm`.",
        "- Seeds: 42, 123, 2025.",
        "- Fixed settings: clients=3, rounds=10, local_epochs=2, batch_size=4, image_size=128.",
        "",
        "## Completion Checks",
        "",
        "- Expected run count: 24",
        f"- Completed unique run count: {len(completed)}",
        f"- Missing runs: {missing if missing else 'none'}",
        f"- Duplicate run groups: {len(duplicate_keys)}",
        f"- Failed or incomplete run directories: {[str(path.relative_to(PROJECT_ROOT)) for path in incomplete_dirs] if incomplete_dirs else 'none'}",
        f"- Disallowed/extra runs under phase5m log root: {[str(path.relative_to(PROJECT_ROOT)) for path in disallowed_paths] if disallowed_paths else 'none'}",
        f"- NaN/inf: {'no' if no_nan_inf else 'yes'}",
        f"- Only approved methods were run: {'yes' if approved_only else 'no'}",
        f"- No failed variants: {'yes' if not any(method in FAILED_METHODS for method in observed_methods) else 'no'}",
        f"- No hard_noniid: {'yes' if splits == {SPLIT} else 'no'}",
        f"- image_size values observed: {sorted(image_sizes)}",
        f"- image_size=128 only: {'yes' if image_sizes == {128} else 'no'}",
        f"- image_size=256 avoided: {'yes' if 256 not in image_sizes else 'no'}",
        f"- BUSI normal avoided: {'yes' if busi_normal_avoided(summaries) else 'no'}",
        f"- No large model / SAM / MedSAM / pretrained backbone was added: {'yes' if no_large_model else 'no'}",
        f"- Logs located under phase5m_moderate_multiseed: {'yes' if all('phase5m_moderate_multiseed' in item['_summary_path'] for item in summaries.values()) else 'no'}",
        f"- Each method has exactly 3 seeds per dataset: {'yes' if exact_seed_counts else 'no'}",
        f"- WCA-Comm communication logs exist: {'yes' if wca_logs_ok else 'no'}",
        f"- WCA-Comm communication reduction exists: {'yes' if wca_logs_ok else 'no'}",
        "",
        "## BUSI Moderate Non-IID Interpretation",
        "",
        *dataset_interpretation(aggregates, "busi"),
        "",
        "## Kvasir-SEG Moderate Non-IID Interpretation",
        "",
        *dataset_interpretation(aggregates, "kvasir_seg"),
        "",
        "## Cross-Dataset Decision",
        "",
    ]
    busi_rows = method_rows(aggregates, "busi")
    kvasir_rows = method_rows(aggregates, "kvasir_seg")
    wca_busi = busi_rows["wca_comfedseg_comm"]
    wca_kvasir = kvasir_rows["wca_comfedseg_comm"]
    wca_has_comm = (
        float(wca_busi["communication_reduction_percent_mean"]) > 0.0
        and float(wca_kvasir["communication_reduction_percent_mean"]) > 0.0
    )
    wca_busi_avg_comp = competitive(
        float(wca_busi["average_dice_mean"]),
        max(float(row["average_dice_mean"]) for row in busi_rows.values()),
    )
    wca_kvasir_avg_comp = competitive(
        float(wca_kvasir["average_dice_mean"]),
        max(float(row["average_dice_mean"]) for row in kvasir_rows.values()),
    )
    wca_busi_worst_comp = competitive(
        float(wca_busi["worst_client_dice_mean"]),
        max(float(row["worst_client_dice_mean"]) for row in busi_rows.values()),
    )
    wca_kvasir_worst_comp = competitive(
        float(wca_kvasir["worst_client_dice_mean"]),
        max(float(row["worst_client_dice_mean"]) for row in kvasir_rows.values()),
    )
    keep_candidate = wca_has_comm and wca_busi_avg_comp and wca_busi_worst_comp and wca_kvasir_avg_comp and wca_kvasir_worst_comp
    lines.extend(
        [
            f"- WCA-Comm stable enough to remain the moderate_noniid communication-efficient candidate: {keep_candidate}.",
            f"- Communication-efficiency trade-off acceptable: {wca_has_comm and (wca_busi_avg_comp or wca_kvasir_avg_comp)}.",
            f"- Proceed directly to final table/figure integration: {keep_candidate}.",
            f"- Additional diagnosis needed before final integration: {not keep_candidate}.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if keep_candidate:
        lines.append("- Keep `wca_comfedseg_comm` as the scoped moderate_noniid communication-efficient candidate and proceed to final table/figure integration.")
    else:
        lines.append("- Keep `wca_comfedseg_comm` as a communication-saving candidate, but present the Dice/fairness trade-off honestly and run targeted diagnosis before final claims.")
    lines.append("- Treat Phase 5M as scoped validation only, not as final SCI evidence.")
    (SUMMARY_DIR / "phase5m_moderate_multiseed_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    notes = [
        "# Phase 5M Notes",
        "",
        "Phase 5M ran the approved moderate_noniid scoped multi-seed matrix only.",
        "",
        "- No hard_noniid experiments were run.",
        "- No failed variants were run.",
        "- `wca_comfedseg_prox`, `wca_comfedseg_bn`, and `wca_comfedseg_prox_comm_cons` were not run.",
        "- `local` and `centralized` were not run.",
        "- image_size=256 was not used.",
        "- BUSI normal was not used.",
        "- No large model, SAM, MedSAM, or pretrained backbone was added.",
        "- Existing Phase 5A-5L result directories were not overwritten.",
        "",
        "Use the aggregate and detail CSV files for scoped candidate positioning. Do not treat these numbers as final SCI evidence.",
    ]
    (SUMMARY_DIR / "phase5m_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    summaries, grouped_paths, disallowed_paths, incomplete_dirs = collect_summaries()
    details = [detail_row(summary) for _key, summary in sorted(summaries.items())]
    aggregates = aggregate_rows(summaries)

    write_csv(SUMMARY_DIR / "phase5m_moderate_multiseed_aggregate.csv", aggregates, AGG_FIELDS)
    write_csv(SUMMARY_DIR / "phase5m_moderate_multiseed_busi.csv", rows_for_dataset(details, "busi"), DETAIL_FIELDS)
    write_csv(SUMMARY_DIR / "phase5m_moderate_multiseed_kvasir_seg.csv", rows_for_dataset(details, "kvasir_seg"), DETAIL_FIELDS)
    generate_figures(aggregates)
    write_reports(summaries, grouped_paths, disallowed_paths, incomplete_dirs, aggregates)

    print(SUMMARY_DIR / "phase5m_moderate_multiseed_aggregate.csv")
    print(SUMMARY_DIR / "phase5m_moderate_multiseed_busi.csv")
    print(SUMMARY_DIR / "phase5m_moderate_multiseed_kvasir_seg.csv")
    print(SUMMARY_DIR / "phase5m_moderate_multiseed_verification.md")
    print(SUMMARY_DIR / "phase5m_notes.md")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
