"""Compare Phase 5S BUS-UCLM WCA+BN scoped pilot logs.

Reads Phase 5S FedBN / WCA+BN runs and the existing Phase 5Q WCA-Comm
reference. This script does not train models.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5S_LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5s_bus_uclm_wca_bn_scoped_pilot"
PHASE5Q_LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5q_bus_uclm_moderate_pilot"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5s_bus_uclm_wca_bn_scoped_pilot"

DATASET = "bus_uclm"
SPLIT = "moderate_noniid"
SEED = 42
PHASE5S_METHODS = ["fedbn", "wca_comfedseg_bn"]
REFERENCE_METHODS = ["wca_comfedseg_comm"]
TABLE_METHODS = ["fedbn", "wca_comfedseg_bn", "wca_comfedseg_comm"]
FAILED_METHODS = {"wca_comfedseg_smooth", "wca_comfedseg_pbn", "wca_comfedseg_rg"}

TABLE_FIELDS = [
    "method",
    "source_phase",
    "dataset",
    "split",
    "seed",
    "rounds",
    "local_epochs",
    "batch_size",
    "image_size",
    "number_of_clients",
    "average_dice",
    "average_iou",
    "worst_client_id",
    "worst_client_dice",
    "best_client_dice",
    "client_dice_std",
    "best_worst_gap",
    "average_loss",
    "total_uploaded_mb",
    "communication_reduction_vs_full_participation_percent",
    "communication_log_path",
    "summary_path",
    "metrics_path",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def fmt(value: Any, digits: int = 10) -> str:
    if value == "" or value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric value: {value}")
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def find_latest_summary(log_root: Path, method: str) -> tuple[Path | None, list[Path]]:
    paths = sorted(
        log_root.glob(f"*_{method}_{DATASET}_{SPLIT}_seed{SEED}/summary.json"),
        key=lambda item: item.stat().st_mtime,
    )
    return (paths[-1] if paths else None), paths


def load_runs() -> tuple[dict[str, dict[str, Any]], dict[str, list[Path]], list[Path]]:
    runs: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Path]] = {}
    incomplete_dirs: list[Path] = []

    for run_dir in sorted(path for path in PHASE5S_LOG_ROOT.glob("*") if path.is_dir()):
        if not (run_dir / "summary.json").exists() or not (run_dir / "metrics.csv").exists():
            incomplete_dirs.append(run_dir)

    for method in PHASE5S_METHODS:
        latest, paths = find_latest_summary(PHASE5S_LOG_ROOT, method)
        grouped[method] = paths
        if latest is None:
            continue
        summary = read_json(latest)
        metrics_path = project_path(summary.get("csv_log_path", latest.parent / "metrics.csv"))
        runs[method] = {
            "summary": summary,
            "summary_path": latest,
            "metrics_path": metrics_path,
            "metrics": read_csv(metrics_path),
            "source_phase": "phase5s",
        }

    latest_ref, paths_ref = find_latest_summary(PHASE5Q_LOG_ROOT, "wca_comfedseg_comm")
    grouped["wca_comfedseg_comm"] = paths_ref
    if latest_ref is not None:
        summary = read_json(latest_ref)
        metrics_path = project_path(summary.get("csv_log_path", latest_ref.parent / "metrics.csv"))
        runs["wca_comfedseg_comm"] = {
            "summary": summary,
            "summary_path": latest_ref,
            "metrics_path": metrics_path,
            "metrics": read_csv(metrics_path),
            "source_phase": "phase5q_reference",
        }
    return runs, grouped, incomplete_dirs


def final_client_dice(metrics: list[dict[str, str]]) -> dict[int, float]:
    final_round = max(int(row["round"]) for row in metrics)
    return {int(row["client_id"]): float(row["dice"]) for row in metrics if int(row["round"]) == final_round}


def run_stats(metrics: list[dict[str, str]]) -> dict[str, Any]:
    dice = final_client_dice(metrics)
    values = list(dice.values())
    worst = min(dice, key=dice.get)
    best = max(dice, key=dice.get)
    final_rows = [row for row in metrics if int(row["round"]) == max(int(item["round"]) for item in metrics)]
    return {
        "client_dice": dice,
        "average_dice": mean(values),
        "average_iou": mean(float(row["iou"]) for row in final_rows),
        "average_loss": mean(float(row["loss"]) for row in final_rows),
        "worst_client_id": worst,
        "worst_client_dice": dice[worst],
        "best_client_id": best,
        "best_client_dice": dice[best],
        "client_dice_std": pstdev(values),
        "best_worst_gap": dice[best] - dice[worst],
    }


def table_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_reference = ""
    if "wca_comfedseg_comm" in runs:
        full_reference = runs["wca_comfedseg_comm"]["summary"].get("full_participation_uploaded_mb", "")
    for method in TABLE_METHODS:
        if method not in runs:
            continue
        summary = runs[method]["summary"]
        args = summary.get("args", {})
        stats = run_stats(runs[method]["metrics"])
        total_uploaded = summary.get("total_uploaded_mb", "")
        if method != "wca_comfedseg_comm" and full_reference != "":
            total_uploaded = full_reference
        rows.append(
            {
                "method": method,
                "source_phase": runs[method]["source_phase"],
                "dataset": summary.get("dataset", ""),
                "split": summary.get("split", ""),
                "seed": args.get("seed", ""),
                "rounds": args.get("rounds", ""),
                "local_epochs": args.get("local_epochs", ""),
                "batch_size": args.get("batch_size", ""),
                "image_size": args.get("image_size", ""),
                "number_of_clients": summary.get("number_of_clients", ""),
                "average_dice": fmt(stats["average_dice"]),
                "average_iou": fmt(stats["average_iou"]),
                "worst_client_id": stats["worst_client_id"],
                "worst_client_dice": fmt(stats["worst_client_dice"]),
                "best_client_dice": fmt(stats["best_client_dice"]),
                "client_dice_std": fmt(stats["client_dice_std"]),
                "best_worst_gap": fmt(stats["best_worst_gap"]),
                "average_loss": fmt(stats["average_loss"]),
                "total_uploaded_mb": fmt(total_uploaded),
                "communication_reduction_vs_full_participation_percent": fmt(
                    summary.get("communication_reduction_vs_full_participation_percent", 0.0)
                ),
                "communication_log_path": summary.get("communication_log_path", ""),
                "summary_path": rel(runs[method]["summary_path"]),
                "metrics_path": rel(runs[method]["metrics_path"]),
            }
        )
    return rows


def client_diagnosis_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dice_by_method = {method: final_client_dice(runs[method]["metrics"]) for method in TABLE_METHODS if method in runs}
    rows: list[dict[str, Any]] = []
    for client_id in [0, 1, 2]:
        fedbn = dice_by_method["fedbn"][client_id]
        wca_bn = dice_by_method["wca_comfedseg_bn"][client_id]
        wca_comm = dice_by_method["wca_comfedseg_comm"][client_id]
        rows.append(
            {
                "client_id": client_id,
                "fedbn_dice": fmt(fedbn),
                "wca_comfedseg_bn_dice": fmt(wca_bn),
                "wca_comfedseg_comm_dice": fmt(wca_comm),
                "wca_bn_minus_fedbn": fmt(wca_bn - fedbn),
                "wca_bn_minus_wca_comm": fmt(wca_bn - wca_comm),
                "wca_bn_best_for_client": wca_bn >= max(fedbn, wca_comm),
                "wca_bn_improves_c0_c1_issue": client_id in {0, 1} and wca_bn > wca_comm,
            }
        )
    return rows


def normal_deferred_ok(summary: dict[str, Any]) -> bool:
    dataset_meta = summary.get("split_configuration", {}).get("dataset", {})
    classes_used = set(dataset_meta.get("classes_used", []))
    classes_deferred = set(dataset_meta.get("classes_deferred", []))
    return "Normal" not in classes_used and "Normal" in classes_deferred


def no_empty_leakage(summary: dict[str, Any]) -> bool:
    split_config = summary.get("split_configuration", {})
    if split_config.get("leakage_check", {}).get("passed") is not True:
        return False
    return all(client.get("empty_client") is not True for client in split_config.get("clients", []))


def no_nan_inf(runs: dict[str, dict[str, Any]]) -> bool:
    for run in runs.values():
        for row in run["metrics"]:
            for key in ["dice", "iou", "loss"]:
                if not math.isfinite(float(row[key])):
                    return False
        stats = run_stats(run["metrics"])
        for key in ["average_dice", "worst_client_dice", "client_dice_std", "best_worst_gap"]:
            if not math.isfinite(float(stats[key])):
                return False
    return True


def phase5s_scope_ok(runs: dict[str, dict[str, Any]]) -> bool:
    for method in PHASE5S_METHODS:
        if method not in runs:
            return False
        summary = runs[method]["summary"]
        args = summary.get("args", {})
        if summary.get("method") != method or summary.get("dataset") != DATASET or summary.get("split") != SPLIT:
            return False
        if int(args.get("seed", -1)) != SEED or int(args.get("image_size", -1)) != 128:
            return False
        if int(args.get("rounds", -1)) != 10 or int(args.get("local_epochs", -1)) != 2:
            return False
        if int(args.get("clients", -1)) != 3 or int(args.get("batch_size", -1)) != 4:
            return False
    return True


def phase5s_extra_methods() -> set[str]:
    observed = set()
    if not PHASE5S_LOG_ROOT.exists():
        return observed
    for summary_path in PHASE5S_LOG_ROOT.glob("*/summary.json"):
        observed.add(str(read_json(summary_path).get("method", "")))
    return observed - set(PHASE5S_METHODS)


def font_pair() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18), ImageFont.truetype(
            "C:/Windows/Fonts/arialbd.ttf",
            22,
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font = font_pair()
    width, height = 920, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 80, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(values + [1e-8])
    if max_value <= 1.0:
        max_value = 1.0
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 26), title, fill="black", font=title_font)
    draw.text((16, margin_top + 8), ylabel, fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)
    colors = [(84, 162, 75), (178, 121, 162), (76, 120, 168)]
    gap = 32
    bar_width = int((plot_w - gap * (len(values) + 1)) / max(len(values), 1))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_left + gap + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_h = int((value / max_value) * (plot_h - 24))
        y0 = margin_top + plot_h - bar_h
        y1 = margin_top + plot_h
        draw.rectangle((x0, y0, x1, y1), fill=colors[idx % len(colors)])
        draw.text((x0, max(margin_top, y0 - 24)), f"{value:.4g}", fill="black", font=font)
        draw.text((x0, y1 + 14), label, fill="black", font=font)
    canvas.save(output_path)


def generate_figures(table: list[dict[str, Any]], diagnosis: list[dict[str, Any]]) -> None:
    labels = [row["method"].replace("wca_comfedseg_", "wca_").replace("wca_comm", "wca_comm") for row in table]
    bar_plot(
        labels,
        [float(row["average_dice"]) for row in table],
        "Phase 5S BUS-UCLM average Dice",
        "average Dice",
        FIGURE_DIR / "phase5s_average_dice.png",
    )
    bar_plot(
        labels,
        [float(row["worst_client_dice"]) for row in table],
        "Phase 5S BUS-UCLM worst-client Dice",
        "worst-client Dice",
        FIGURE_DIR / "phase5s_worst_client_dice.png",
    )
    bar_plot(
        labels,
        [float(row["best_worst_gap"]) for row in table],
        "Phase 5S BUS-UCLM best-worst gap",
        "best-worst gap",
        FIGURE_DIR / "phase5s_best_worst_gap.png",
    )
    client_labels: list[str] = []
    client_values: list[float] = []
    for row in diagnosis:
        cid = row["client_id"]
        client_labels.extend([f"C{cid} FedBN", f"C{cid} WCA+BN", f"C{cid} WCA-Comm"])
        client_values.extend(
            [
                float(row["fedbn_dice"]),
                float(row["wca_comfedseg_bn_dice"]),
                float(row["wca_comfedseg_comm_dice"]),
            ]
        )
    bar_plot(
        client_labels,
        client_values,
        "Phase 5S client-level Dice comparison",
        "client Dice",
        FIGURE_DIR / "phase5s_client_level_dice.png",
    )


def write_reports(
    runs: dict[str, dict[str, Any]],
    grouped: dict[str, list[Path]],
    incomplete_dirs: list[Path],
    table: list[dict[str, Any]],
    diagnosis: list[dict[str, Any]],
) -> None:
    row_by_method = {row["method"]: row for row in table}
    extra_methods = phase5s_extra_methods()
    phase5q_ref_ok = "wca_comfedseg_comm" in runs and runs["wca_comfedseg_comm"]["source_phase"] == "phase5q_reference"
    checks = {
        "phase5s_new_runs_only_fedbn_and_wca_bn": not extra_methods and {"fedbn", "wca_comfedseg_bn"} <= set(runs),
        "only_bus_uclm": all(runs[m]["summary"].get("dataset") == DATASET for m in runs),
        "only_moderate_noniid": all(runs[m]["summary"].get("split") == SPLIT for m in runs),
        "only_seed_42": all(int(runs[m]["summary"].get("args", {}).get("seed", -1)) == SEED for m in runs),
        "image_size_128_only": all(int(runs[m]["summary"].get("args", {}).get("image_size", -1)) == 128 for m in runs),
        "normal_deferred": all(normal_deferred_ok(runs[m]["summary"]) for m in runs),
        "no_failed_variants": not (set(runs) & FAILED_METHODS) and not (extra_methods & FAILED_METHODS),
        "no_hard_noniid": all(runs[m]["summary"].get("split") != "hard_noniid" for m in runs),
        "no_nan_or_inf": no_nan_inf(runs),
        "summary_and_metrics_present": not incomplete_dirs and all(runs[m]["summary_path"].exists() and runs[m]["metrics_path"].exists() for m in runs),
        "no_empty_client_or_leakage": all(no_empty_leakage(runs[m]["summary"]) for m in runs),
        "phase5q_wca_comm_reference_read": phase5q_ref_ok,
        "wca_comm_not_retrained_in_phase5s": "wca_comfedseg_comm" not in phase5s_extra_methods(),
        "phase5s_expected_settings": phase5s_scope_ok(runs),
    }
    technical_pass = all(checks.values())

    fedbn = row_by_method["fedbn"]
    wca_bn = row_by_method["wca_comfedseg_bn"]
    wca_comm = row_by_method["wca_comfedseg_comm"]
    wca_bn_avg_beats_fedbn = float(wca_bn["average_dice"]) > float(fedbn["average_dice"])
    wca_bn_worst_beats_fedbn = float(wca_bn["worst_client_dice"]) > float(fedbn["worst_client_dice"])
    wca_bn_gap_below_fedbn = float(wca_bn["best_worst_gap"]) < float(fedbn["best_worst_gap"])
    wca_bn_comm_reduction = float(wca_bn["communication_reduction_vs_full_participation_percent"] or 0.0)
    c01_improved = all(
        row["wca_bn_improves_c0_c1_issue"] in {True, "True"}
        for row in diagnosis
        if int(row["client_id"]) in {0, 1}
    )
    wca_bn_near_fedbn = (
        abs(float(wca_bn["average_dice"]) - float(fedbn["average_dice"])) <= 0.02
        and abs(float(wca_bn["worst_client_dice"]) - float(fedbn["worst_client_dice"])) <= 0.02
    )
    recommend_multiseed = technical_pass and wca_bn_avg_beats_fedbn and wca_bn_worst_beats_fedbn and wca_bn_gap_below_fedbn
    status = "passed" if technical_pass else "failed"
    if technical_pass and not recommend_multiseed:
        status = "needs adjustment" if wca_bn_near_fedbn else "passed"
    duplicate_groups = {method: len(paths) for method, paths in grouped.items() if len(paths) > 1}

    verification = [
        "# Phase 5S BUS-UCLM WCA+BN Scoped Pilot Verification",
        "",
        "Phase 5S is a scoped diagnostic pilot only. It is not final SCI evidence.",
        "",
        "## Checks",
        "",
    ]
    verification.extend(f"- {name}: {'passed' if ok else 'failed'}" for name, ok in checks.items())
    verification.extend(
        [
            f"- Phase 5S incomplete run directories: {[rel(path) for path in incomplete_dirs] if incomplete_dirs else 'none'}",
            f"- Phase 5S extra methods observed: {sorted(extra_methods) if extra_methods else 'none'}",
            f"- Phase 5S duplicate groups: {duplicate_groups if duplicate_groups else 'none'}",
            "",
            "## Results",
            "",
            f"- FedBN average Dice={float(fedbn['average_dice']):.4f}, worst-client Dice={float(fedbn['worst_client_dice']):.4f}, gap={float(fedbn['best_worst_gap']):.4f}.",
            f"- WCA+BN average Dice={float(wca_bn['average_dice']):.4f}, worst-client Dice={float(wca_bn['worst_client_dice']):.4f}, gap={float(wca_bn['best_worst_gap']):.4f}.",
            f"- WCA-Comm Phase 5Q reference average Dice={float(wca_comm['average_dice']):.4f}, worst-client Dice={float(wca_comm['worst_client_dice']):.4f}, gap={float(wca_comm['best_worst_gap']):.4f}, communication reduction={float(wca_comm['communication_reduction_vs_full_participation_percent']):.2f}%.",
            f"- WCA+BN exceeds FedBN average Dice: {wca_bn_avg_beats_fedbn}.",
            f"- WCA+BN exceeds FedBN worst-client Dice: {wca_bn_worst_beats_fedbn}.",
            f"- WCA+BN has lower gap than FedBN: {wca_bn_gap_below_fedbn}.",
            f"- WCA+BN improves WCA-Comm C0/C1 issue: {c01_improved}.",
            f"- WCA+BN communication reduction: {wca_bn_comm_reduction:.2f}%.",
            "",
            "## Decisions",
            "",
            f"- Phase 5S status: {status}.",
            f"- Recommend WCA+BN BUS-UCLM multi-seed: {recommend_multiseed}.",
            f"- Recommend BUS-UCLM as main-text positive result: {recommend_multiseed}.",
            f"- Continue using BUSI moderate as the only main success evidence: {not recommend_multiseed}.",
            f"- Need claim-boundary update: {not recommend_multiseed}.",
        ]
    )
    if recommend_multiseed:
        verification.append("- Interpretation: WCA+BN is strong enough for explicitly approved BUS-UCLM scoped multi-seed validation.")
    elif wca_bn_near_fedbn:
        verification.append("- Interpretation: WCA+BN is near FedBN but not clearly stronger; keep as diagnostic observation.")
    else:
        verification.append("- Interpretation: BUS-UCLM remains boundary/limitation evidence rather than WCA success evidence.")
    write_text(SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_scoped_pilot_verification.md", verification)

    notes = [
        "# Phase 5S Notes",
        "",
        "- New Phase 5S training runs: `fedbn`, `wca_comfedseg_bn`.",
        "- Phase 5Q reference reused: `wca_comfedseg_comm` summary, metrics, and communication log.",
        "- WCA-Comm was not retrained for Phase 5S.",
        "- FedAvg/FedProx were not newly run.",
        "- This is a scoped diagnostic pilot only, not final SCI evidence.",
        f"- Phase 5S status: {status}.",
        f"- Recommend WCA+BN BUS-UCLM multi-seed: {recommend_multiseed}.",
        f"- BUS-UCLM paper placement: {'main positive result candidate' if recommend_multiseed else 'diagnosis/boundary/appendix'}."
    ]
    write_text(SUMMARY_DIR / "phase5s_notes.md", notes)


def main() -> None:
    runs, grouped, incomplete_dirs = load_runs()
    table = table_rows(runs)
    diagnosis = client_diagnosis_rows(runs)
    write_csv(SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_scoped_pilot.csv", table, TABLE_FIELDS)
    write_csv(
        SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_client_diagnosis.csv",
        diagnosis,
        [
            "client_id",
            "fedbn_dice",
            "wca_comfedseg_bn_dice",
            "wca_comfedseg_comm_dice",
            "wca_bn_minus_fedbn",
            "wca_bn_minus_wca_comm",
            "wca_bn_best_for_client",
            "wca_bn_improves_c0_c1_issue",
        ],
    )
    generate_figures(table, diagnosis)
    write_reports(runs, grouped, incomplete_dirs, table, diagnosis)

    print(SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_scoped_pilot.csv")
    print(SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_scoped_pilot_verification.md")
    print(SUMMARY_DIR / "phase5s_bus_uclm_wca_bn_client_diagnosis.csv")
    print(SUMMARY_DIR / "phase5s_notes.md")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
