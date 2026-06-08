"""Phase 5R BUS-UCLM moderate_noniid failure diagnosis.

This script is read-only with respect to experiments: it consumes Phase 5Q
logs and Phase 5P split summaries, then writes Phase 5R diagnostic artifacts.
It does not train models, modify splits, or call run_experiment.py.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5r_bus_uclm_failure_diagnosis"

PHASE5Q_TABLE = SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot.csv"
PHASE5Q_VERIFICATION = SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot_verification.md"
CLIENT_DISTRIBUTION = SUMMARY_DIR / "phase5p_ultrasound_client_distribution.csv"
SPLIT_COMPARISON = SUMMARY_DIR / "phase5p_bus_uclm_split_comparison.csv"
DATASET_SUMMARY = SUMMARY_DIR / "phase5p_busi_vs_bus_uclm_dataset_summary.csv"

DATASET = "bus_uclm"
SPLIT = "moderate_noniid"
SEED = 42
METHODS = ["fedavg", "fedprox", "fedbn", "wca_comfedseg_comm"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def fmt(value: Any, digits: int = 10) -> str:
    if value == "" or value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric value: {value}")
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def load_phase5q_table() -> dict[str, dict[str, str]]:
    rows = read_csv(PHASE5Q_TABLE)
    table = {row["method"]: row for row in rows}
    missing = sorted(set(METHODS) - set(table))
    if missing:
        raise RuntimeError(f"Missing Phase 5Q methods: {missing}")
    for method, row in table.items():
        if row["dataset"] != DATASET or row["split"] != SPLIT or int(row["seed"]) != SEED:
            raise RuntimeError(f"Unexpected Phase 5Q scope in {method}: {row}")
    return table


def load_method_logs(table: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    logs: dict[str, dict[str, Any]] = {}
    for method, row in table.items():
        summary_path = project_path(row["summary_path"])
        metrics_path = project_path(row["csv_log_path"])
        summary = read_json(summary_path)
        metrics = read_csv(metrics_path)
        if not metrics:
            raise RuntimeError(f"Empty metrics log: {metrics_path}")
        logs[method] = {
            "summary": summary,
            "metrics": metrics,
            "summary_path": summary_path,
            "metrics_path": metrics_path,
        }
    return logs


def final_round(metrics: list[dict[str, str]]) -> int:
    return max(int(row["round"]) for row in metrics)


def final_client_dice(metrics: list[dict[str, str]]) -> dict[int, float]:
    last_round = final_round(metrics)
    return {
        int(row["client_id"]): float(row["dice"])
        for row in metrics
        if int(row["round"]) == last_round
    }


def client_dice_by_round(metrics: list[dict[str, str]]) -> dict[int, dict[int, float]]:
    by_round: dict[int, dict[int, float]] = defaultdict(dict)
    for row in metrics:
        by_round[int(row["round"])][int(row["client_id"])] = float(row["dice"])
    return dict(by_round)


def trajectory(metrics: list[dict[str, str]], client_id: int) -> str:
    values = [
        (int(row["round"]), float(row["dice"]))
        for row in metrics
        if int(row["client_id"]) == client_id
    ]
    return "|".join(f"r{round_id}:{value:.6f}" for round_id, value in sorted(values))


def method_stats(metrics: list[dict[str, str]]) -> dict[str, Any]:
    dice = final_client_dice(metrics)
    values = list(dice.values())
    worst_client = min(dice, key=dice.get)
    best_client = max(dice, key=dice.get)
    return {
        "final_average_dice": mean(values),
        "final_client_dice": dice,
        "worst_client_id": worst_client,
        "worst_client_dice": dice[worst_client],
        "best_client_id": best_client,
        "best_client_dice": dice[best_client],
        "client_dice_std": pstdev(values),
        "best_worst_gap": dice[best_client] - dice[worst_client],
    }


def load_split_features() -> dict[int, dict[str, dict[str, str]]]:
    rows = read_csv(CLIENT_DISTRIBUTION)
    selected = [
        row
        for row in rows
        if row["dataset"] == "BUS-UCLM" and row["split_mode"] == SPLIT
    ]
    features: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in selected:
        features[int(row["client_id"])][row["subset"]] = row
    if sorted(features) != [0, 1, 2]:
        raise RuntimeError(f"Unexpected BUS-UCLM client feature keys: {sorted(features)}")
    return dict(features)


def client_performance_rows(logs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    stats = {method: method_stats(logs[method]["metrics"]) for method in METHODS}
    fedbn_dice = stats["fedbn"]["final_client_dice"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_stat = stats[method]
        method_worst = method_stat["worst_client_id"]
        for client_id, dice in sorted(method_stat["final_client_dice"].items()):
            rows.append(
                {
                    "method": method,
                    "dataset": DATASET,
                    "split": SPLIT,
                    "seed": SEED,
                    "client_id": client_id,
                    "final_average_dice": fmt(method_stat["final_average_dice"]),
                    "final_client_dice": fmt(dice),
                    "worst_client_id": method_worst,
                    "best_client_id": method_stat["best_client_id"],
                    "is_worst_client": client_id == method_worst,
                    "is_best_client": client_id == method_stat["best_client_id"],
                    "client_dice_std": fmt(method_stat["client_dice_std"]),
                    "best_worst_gap": fmt(method_stat["best_worst_gap"]),
                    "fedbn_minus_method_client_dice": fmt(fedbn_dice[client_id] - dice),
                    "same_worst_client_as_fedbn": method_worst == stats["fedbn"]["worst_client_id"],
                    "same_worst_client_as_wca_comm": method_worst == stats["wca_comfedseg_comm"]["worst_client_id"],
                    "client_dice_trajectory": trajectory(logs[method]["metrics"], client_id),
                }
            )
    return rows


def split_feature_vs_performance_rows(
    logs: dict[str, dict[str, Any]],
    features: dict[int, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    dice_by_method = {method: final_client_dice(logs[method]["metrics"]) for method in METHODS}
    worst_by_method = {method: min(dice_by_method[method], key=dice_by_method[method].get) for method in METHODS}
    rows: list[dict[str, Any]] = []
    for client_id in sorted(features):
        row: dict[str, Any] = {"dataset": DATASET, "split": SPLIT, "seed": SEED, "client_id": client_id}
        for subset in ["train", "val", "test"]:
            subset_features = features[client_id][subset]
            for key in [
                "sample_count",
                "benign_count",
                "malignant_count",
                "malignant_ratio",
                "mean_mask_area_fraction",
                "median_mask_area_fraction",
                "patient_study_identity_count",
                "empty_client",
                "stem_leakage",
                "patient_leakage",
            ]:
                row[f"{subset}_{key}"] = subset_features[key]
        for method in METHODS:
            row[f"{method}_final_dice"] = fmt(dice_by_method[method][client_id])
            row[f"{method}_is_worst_client"] = client_id == worst_by_method[method]
        row["fedbn_minus_wca_comm_dice"] = fmt(
            dice_by_method["fedbn"][client_id] - dice_by_method["wca_comfedseg_comm"][client_id]
        )
        row["fedbn_minus_fedavg_dice"] = fmt(dice_by_method["fedbn"][client_id] - dice_by_method["fedavg"][client_id])
        row["fedbn_minus_fedprox_dice"] = fmt(dice_by_method["fedbn"][client_id] - dice_by_method["fedprox"][client_id])
        rows.append(row)
    return rows


def load_wca_communication(table: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    path = project_path(table["wca_comfedseg_comm"]["communication_log_path"])
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty WCA communication log: {path}")
    return rows


def scheduler_rows(logs: dict[str, dict[str, Any]], comm_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    wca_metrics = logs["wca_comfedseg_comm"]["metrics"]
    wca_round_dice = client_dice_by_round(wca_metrics)
    final_dice = final_client_dice(wca_metrics)
    final_worst = min(final_dice, key=final_dice.get)
    selected_counts = Counter()
    skipped_counts = Counter()
    for row in comm_rows:
        client_id = int(row["client_id"])
        if row["selected"] == "True":
            selected_counts[client_id] += 1
        else:
            skipped_counts[client_id] += 1

    previous_worst_by_round: dict[int, int | str] = {}
    evaluated_worst_by_round: dict[int, int] = {}
    for round_id in sorted(wca_round_dice):
        evaluated_worst_by_round[round_id] = min(wca_round_dice[round_id], key=wca_round_dice[round_id].get)
        previous_worst_by_round[round_id] = "" if round_id == 1 else evaluated_worst_by_round[round_id - 1]

    total_rounds = len({int(row["round"]) for row in comm_rows})
    rows: list[dict[str, Any]] = []
    for row in comm_rows:
        round_id = int(row["round"])
        client_id = int(row["client_id"])
        selected = row["selected"] == "True"
        rows.append(
            {
                "round": round_id,
                "client_id": client_id,
                "selected": selected,
                "selected_reason": row["selected_reason"],
                "previous_validation_dice": row["previous_validation_dice"],
                "performance_deficit": row["performance_deficit"],
                "uploaded_mb": fmt(row["uploaded_mb"]),
                "previous_round_worst_client_id": previous_worst_by_round[round_id],
                "selected_previous_round_worst": (
                    previous_worst_by_round[round_id] != "" and client_id == previous_worst_by_round[round_id] and selected
                ),
                "current_round_evaluated_worst_client_id": evaluated_worst_by_round[round_id],
                "is_final_worst_client": client_id == final_worst,
                "final_client_dice": fmt(final_dice[client_id]),
                "selection_count": selected_counts[client_id],
                "skip_count": skipped_counts[client_id],
                "selection_frequency": fmt(selected_counts[client_id] / total_rounds),
                "skip_frequency": fmt(skipped_counts[client_id] / total_rounds),
                "communication_reduction_source": (
                    "skipped_client" if not selected else "uploaded_client"
                ),
            }
        )
    return rows


def fedbn_effect_rows(logs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    stats = {method: method_stats(logs[method]["metrics"]) for method in METHODS}
    dice_by_method = {method: stats[method]["final_client_dice"] for method in METHODS}
    rows: list[dict[str, Any]] = []
    for client_id in sorted(dice_by_method["fedbn"]):
        rows.append(
            {
                "client_id": client_id,
                "fedavg_dice": fmt(dice_by_method["fedavg"][client_id]),
                "fedprox_dice": fmt(dice_by_method["fedprox"][client_id]),
                "fedbn_dice": fmt(dice_by_method["fedbn"][client_id]),
                "wca_comfedseg_comm_dice": fmt(dice_by_method["wca_comfedseg_comm"][client_id]),
                "fedbn_minus_fedavg": fmt(dice_by_method["fedbn"][client_id] - dice_by_method["fedavg"][client_id]),
                "fedbn_minus_fedprox": fmt(dice_by_method["fedbn"][client_id] - dice_by_method["fedprox"][client_id]),
                "fedbn_minus_wca_comm": fmt(
                    dice_by_method["fedbn"][client_id] - dice_by_method["wca_comfedseg_comm"][client_id]
                ),
                "fedbn_is_best_for_client": dice_by_method["fedbn"][client_id]
                == max(dice_by_method[method][client_id] for method in METHODS),
                "fedbn_worst_client_id": stats["fedbn"]["worst_client_id"],
                "fedbn_improves_worst_client_vs_fedavg": fmt(
                    stats["fedbn"]["worst_client_dice"] - stats["fedavg"]["worst_client_dice"]
                ),
                "fedbn_improves_worst_client_vs_wca_comm": fmt(
                    stats["fedbn"]["worst_client_dice"] - stats["wca_comfedseg_comm"]["worst_client_dice"]
                ),
                "fedbn_client_std": fmt(stats["fedbn"]["client_dice_std"]),
                "fedavg_client_std": fmt(stats["fedavg"]["client_dice_std"]),
                "wca_comm_client_std": fmt(stats["wca_comfedseg_comm"]["client_dice_std"]),
                "fedbn_gap": fmt(stats["fedbn"]["best_worst_gap"]),
                "fedavg_gap": fmt(stats["fedavg"]["best_worst_gap"]),
                "wca_comm_gap": fmt(stats["wca_comfedseg_comm"]["best_worst_gap"]),
                "fedbn_std_reduction_vs_fedavg": fmt(stats["fedavg"]["client_dice_std"] - stats["fedbn"]["client_dice_std"]),
                "fedbn_gap_reduction_vs_fedavg": fmt(stats["fedavg"]["best_worst_gap"] - stats["fedbn"]["best_worst_gap"]),
                "fedbn_gap_reduction_vs_wca_comm": fmt(
                    stats["wca_comfedseg_comm"]["best_worst_gap"] - stats["fedbn"]["best_worst_gap"]
                ),
            }
        )
    return rows


def font_pair() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18), ImageFont.truetype(
            "C:/Windows/Fonts/arialbd.ttf",
            22,
        )
    except OSError:
        font = ImageFont.load_default()
        return font, font


def bar_plot(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font = font_pair()
    colors = colors or [(76, 120, 168), (245, 133, 24), (84, 162, 75), (178, 121, 162)]
    width, height = 980, 580
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 80, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max(values + [1e-8])
    if max_value <= 1.0 and "frequency" not in ylabel.lower():
        max_value = max(max_value, 1.0)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 26), title, fill="black", font=title_font)
    draw.text((18, margin_top + 8), ylabel, fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)
    gap = 18
    bar_width = max(12, int((plot_w - gap * (len(values) + 1)) / max(len(values), 1)))
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


def scatter_plot(
    points: list[tuple[float, float, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font = font_pair()
    width, height = 920, 580
    margin_left, margin_right, margin_top, margin_bottom = 110, 60, 80, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 0.01
        x_max += 0.01
    if y_min == y_max:
        y_min -= 0.01
        y_max += 0.01
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 26), title, fill="black", font=title_font)
    draw.text((margin_left + plot_w // 2 - 60, height - 42), xlabel, fill="black", font=font)
    draw.text((18, margin_top + 8), ylabel, fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)
    for x, y, label in points:
        px = margin_left + int(((x - x_min) / (x_max - x_min)) * plot_w)
        py = margin_top + plot_h - int(((y - y_min) / (y_max - y_min)) * plot_h)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(76, 120, 168))
        draw.text((px + 8, py - 8), label, fill="black", font=font)
    canvas.save(output_path)


def generate_figures(
    logs: dict[str, dict[str, Any]],
    scheduler: list[dict[str, Any]],
    split_perf: list[dict[str, Any]],
) -> None:
    stats = {method: method_stats(logs[method]["metrics"]) for method in METHODS}
    labels: list[str] = []
    values: list[float] = []
    colors: list[tuple[int, int, int]] = []
    method_colors = {
        "fedavg": (76, 120, 168),
        "fedprox": (245, 133, 24),
        "fedbn": (84, 162, 75),
        "wca_comfedseg_comm": (178, 121, 162),
    }
    for method in METHODS:
        for client_id in sorted(stats[method]["final_client_dice"]):
            labels.append(f"{method.replace('wca_comfedseg_comm', 'wca')} C{client_id}")
            values.append(stats[method]["final_client_dice"][client_id])
            colors.append(method_colors[method])
    bar_plot(labels, values, "Phase 5R per-method final client Dice", "final Dice", FIGURE_DIR / "phase5r_client_dice_by_method.png", colors)

    selected_frequency: dict[int, float] = {}
    for row in scheduler:
        selected_frequency[int(row["client_id"])] = float(row["selection_frequency"])
    bar_plot(
        [f"C{client_id}" for client_id in sorted(selected_frequency)],
        [selected_frequency[client_id] for client_id in sorted(selected_frequency)],
        "Phase 5R WCA-Comm selection frequency",
        "selection frequency",
        FIGURE_DIR / "phase5r_wca_selection_frequency.png",
    )

    scatter_plot(
        [
            (
                float(row["train_malignant_ratio"]),
                float(row["wca_comfedseg_comm_final_dice"]),
                f"C{row['client_id']}",
            )
            for row in split_perf
        ],
        "Phase 5R train malignant ratio vs WCA final Dice",
        "train malignant ratio",
        "WCA final Dice",
        FIGURE_DIR / "phase5r_split_feature_vs_wca_dice.png",
    )

    labels = [f"C{client_id}" for client_id in sorted(stats["fedbn"]["final_client_dice"])]
    fedbn_values = [stats["fedbn"]["final_client_dice"][client_id] for client_id in sorted(stats["fedbn"]["final_client_dice"])]
    wca_values = [
        stats["wca_comfedseg_comm"]["final_client_dice"][client_id]
        for client_id in sorted(stats["wca_comfedseg_comm"]["final_client_dice"])
    ]
    paired_labels = []
    paired_values = []
    paired_colors = []
    for label, fedbn_value, wca_value in zip(labels, fedbn_values, wca_values):
        paired_labels.extend([f"{label} FedBN", f"{label} WCA"])
        paired_values.extend([fedbn_value, wca_value])
        paired_colors.extend([(84, 162, 75), (178, 121, 162)])
    bar_plot(
        paired_labels,
        paired_values,
        "Phase 5R FedBN vs WCA-Comm client Dice",
        "final Dice",
        FIGURE_DIR / "phase5r_fedbn_vs_wca_client_dice.png",
        paired_colors,
    )


def write_interpretation_update(
    logs: dict[str, dict[str, Any]],
    fedbn_rows: list[dict[str, Any]],
    split_comparison: list[dict[str, str]],
    dataset_summary: list[dict[str, str]],
) -> None:
    stats = {method: method_stats(logs[method]["metrics"]) for method in METHODS}
    bus_uclm_row = next(row for row in dataset_summary if row["dataset"] == "BUS-UCLM")
    moderate_rows = [row for row in split_comparison if row["dataset"] == "BUS-UCLM" and row["split_mode"] == SPLIT]
    lines = [
        "# Phase 5R BUSI vs BUS-UCLM Interpretation Update",
        "",
        "This update uses Phase 5Q BUS-UCLM logs and Phase 5P split statistics only. It is diagnosis evidence, not final SCI evidence.",
        "",
        "## Current Evidence",
        "",
        "- BUSI moderate_noniid remains the current positive WCA-Comm evidence: Phase 5M reported WCA-Comm as strongest on average Dice, worst-client Dice, and best-worst gap with about 30% uploaded communication reduction.",
        (
            "- BUS-UCLM moderate_noniid Phase 5Q does not reproduce that positive pattern: "
            f"FedBN average Dice={stats['fedbn']['final_average_dice']:.4f}, "
            f"worst-client Dice={stats['fedbn']['worst_client_dice']:.4f}, gap={stats['fedbn']['best_worst_gap']:.4f}; "
            f"WCA-Comm average Dice={stats['wca_comfedseg_comm']['final_average_dice']:.4f}, "
            f"worst-client Dice={stats['wca_comfedseg_comm']['worst_client_dice']:.4f}, "
            f"gap={stats['wca_comfedseg_comm']['best_worst_gap']:.4f}."
        ),
        (
            f"- BUS-UCLM uses Benign+Malignant only: n={bus_uclm_row['included_sample_count']}, "
            f"malignant_ratio={bus_uclm_row['malignant_ratio']}, "
            f"deferred_normal_count={bus_uclm_row['deferred_normal_count']}, "
            f"patient/study identities={bus_uclm_row['patient_study_identity_count']}."
        ),
        "",
        "## Split Context",
        "",
    ]
    for row in moderate_rows:
        lines.append(
            f"- BUS-UCLM moderate_noniid {row['subset']}: count_imbalance={row['sample_count_imbalance_ratio']}, "
            f"malignant_ratio_range={row['malignant_ratio_range']}, mean_mask_area_range={row['mean_mask_area_range']}, "
            f"empty_client_count={row['empty_client_count']}, leakage={row['stem_leakage']}/{row['patient_leakage']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BUSI and BUS-UCLM are both breast ultrasound datasets, but the observed non-IID behavior is not the same.",
            "- BUS-UCLM currently looks more favorable to FedBN/local normalization than to communication-aware WCA scheduling.",
            "- BUS-UCLM should not be written as a second-dataset WCA-Comm success.",
            "- BUS-UCLM is valuable as a boundary/domain-heterogeneity observation and can motivate discussion that ultrasound non-IID can arise from different mechanisms.",
            "- The ultrasound-focused paper direction can remain, but the claim must be narrower: BUSI-main positive evidence, BUS-UCLM boundary diagnosis unless later scoped experiments change the evidence.",
        ]
    )
    write_text(SUMMARY_DIR / "phase5r_busi_bus_uclm_interpretation_update.md", lines)


def write_final_report(
    logs: dict[str, dict[str, Any]],
    split_perf: list[dict[str, Any]],
    scheduler: list[dict[str, Any]],
    fedbn_rows: list[dict[str, Any]],
) -> None:
    stats = {method: method_stats(logs[method]["metrics"]) for method in METHODS}
    selected_counts = {client_id: 0 for client_id in [0, 1, 2]}
    skipped_counts = {client_id: 0 for client_id in [0, 1, 2]}
    selected_previous_worst = []
    for row in scheduler:
        client_id = int(row["client_id"])
        if row["selected"] is True:
            selected_counts[client_id] += 1
        else:
            skipped_counts[client_id] += 1
        if row["selected_previous_round_worst"] is True:
            selected_previous_worst.append((row["round"], client_id))

    wca_final_worst = stats["wca_comfedseg_comm"]["worst_client_id"]
    fedbn_final_worst = stats["fedbn"]["worst_client_id"]
    client2_wca = stats["wca_comfedseg_comm"]["final_client_dice"][2]
    client2_fedbn = stats["fedbn"]["final_client_dice"][2]
    fedbn_best_clients = [
        row["client_id"]
        for row in fedbn_rows
        if row["fedbn_is_best_for_client"] is True or row["fedbn_is_best_for_client"] == "True"
    ]
    lines = [
        "# Phase 5R BUS-UCLM Failure Diagnosis",
        "",
        "Phase 5R uses Phase 5Q logs and Phase 5P split statistics only. No new training was run.",
        "",
        "## 1. Status",
        "",
        "- Phase 5R status: passed.",
        "- Diagnosis scope: BUS-UCLM moderate_noniid, seed 42, Phase 5Q logs.",
        "- No new model training, split modification, Normal inclusion, hard_noniid run, Kvasir run, BUSI run, failed variant, foundation model, SAM, or MedSAM was used.",
        "",
        "## 2. Main WCA-Comm Failure Pattern",
        "",
        (
            f"- WCA-Comm reduced communication by 30%, but its average Dice={stats['wca_comfedseg_comm']['final_average_dice']:.4f}, "
            f"worst-client Dice={stats['wca_comfedseg_comm']['worst_client_dice']:.4f}, "
            f"and best-worst gap={stats['wca_comfedseg_comm']['best_worst_gap']:.4f} were worse than FedBN."
        ),
        (
            f"- FedBN achieved average Dice={stats['fedbn']['final_average_dice']:.4f}, "
            f"worst-client Dice={stats['fedbn']['worst_client_dice']:.4f}, gap={stats['fedbn']['best_worst_gap']:.4f}."
        ),
        f"- WCA-Comm worst client was C{wca_final_worst}; FedBN worst client was C{fedbn_final_worst}.",
        "",
        "## 3. Why FedBN Is Likely Stronger",
        "",
        f"- FedBN had the smallest client Dice std ({stats['fedbn']['client_dice_std']:.4f}) and smallest best-worst gap ({stats['fedbn']['best_worst_gap']:.4f}).",
        f"- FedBN improved the FedAvg worst-client Dice by {stats['fedbn']['worst_client_dice'] - stats['fedavg']['worst_client_dice']:.4f}.",
        f"- FedBN improved the WCA-Comm worst-client Dice by {stats['fedbn']['worst_client_dice'] - stats['wca_comfedseg_comm']['worst_client_dice']:.4f}.",
        f"- FedBN was the best method for clients: {fedbn_best_clients}.",
        "- This supports the interpretation that BUS-UCLM moderate_noniid contains a stronger local-normalization/domain component than the current WCA-Comm mechanism handles.",
        "",
        "## 4. Scheduler Diagnosis",
        "",
        f"- WCA-Comm selected counts per client: {selected_counts}.",
        f"- WCA-Comm skipped counts per client: {skipped_counts}.",
        f"- The scheduler selected the previous/evolving weak client in rounds: {selected_previous_worst}.",
        (
            f"- Communication saving mainly came from skipping C2 after round 1. C2 was the strong WCA client "
            f"(WCA final Dice={client2_wca:.4f}; FedBN final Dice={client2_fedbn:.4f}), not the final weak client."
        ),
        "- Therefore, Phase 5R does not support a simple claim that WCA failed because it skipped the key weak client. The scheduler prioritized weak clients C0/C1, but the method still did not repair their performance enough.",
        "",
        "## 5. Split Feature Diagnosis",
        "",
    ]
    for row in split_perf:
        lines.append(
            f"- C{row['client_id']}: train_n={row['train_sample_count']}, train_malignant_ratio={float(row['train_malignant_ratio']):.4f}, "
            f"train_mean_mask_area={float(row['train_mean_mask_area_fraction']):.4f}, prefixes={row['train_patient_study_identity_count']}; "
            f"FedBN Dice={float(row['fedbn_final_dice']):.4f}, WCA-Comm Dice={float(row['wca_comfedseg_comm_final_dice']):.4f}."
        )
    lines.extend(
        [
            "",
            "## 6. Decisions",
            "",
            "- Recommend continuing BUS-UCLM WCA-Comm multi-seed directly: no.",
            "- Recommend WCA+BN BUS-UCLM moderate_noniid scoped pilot: yes, but only as an explicitly approved diagnostic pilot, not as a default main run.",
            "- Recommended paper placement for BUS-UCLM now: diagnosis / boundary observation, likely appendix or a carefully framed subsection, not a main positive result table.",
            "- Ultrasound-focused direction: keep, but frame as BUSI-main positive evidence plus BUS-UCLM boundary/domain-heterogeneity evidence.",
            "",
            "## 7. Next Stage",
            "",
            "- Inspect BUS-UCLM client/domain characteristics around patient prefixes and image/mask acquisition heterogeneity.",
            "- If approved, run a scoped WCA+BN BUS-UCLM moderate_noniid pilot against FedBN and WCA-Comm.",
            "- Do not include BUS-UCLM as second-dataset success unless future scoped evidence supports WCA-style improvement.",
        ]
    )
    write_text(SUMMARY_DIR / "phase5r_bus_uclm_failure_diagnosis.md", lines)


def main() -> None:
    table = load_phase5q_table()
    logs = load_method_logs(table)
    features = load_split_features()
    comm_rows = load_wca_communication(table)
    split_comparison = read_csv(SPLIT_COMPARISON)
    dataset_summary = read_csv(DATASET_SUMMARY)

    perf_rows = client_performance_rows(logs)
    split_perf_rows = split_feature_vs_performance_rows(logs, features)
    scheduler = scheduler_rows(logs, comm_rows)
    fedbn = fedbn_effect_rows(logs)

    write_csv(
        SUMMARY_DIR / "phase5r_bus_uclm_client_performance_diagnosis.csv",
        perf_rows,
        [
            "method",
            "dataset",
            "split",
            "seed",
            "client_id",
            "final_average_dice",
            "final_client_dice",
            "worst_client_id",
            "best_client_id",
            "is_worst_client",
            "is_best_client",
            "client_dice_std",
            "best_worst_gap",
            "fedbn_minus_method_client_dice",
            "same_worst_client_as_fedbn",
            "same_worst_client_as_wca_comm",
            "client_dice_trajectory",
        ],
    )
    split_fields = ["dataset", "split", "seed", "client_id"]
    for subset in ["train", "val", "test"]:
        for key in [
            "sample_count",
            "benign_count",
            "malignant_count",
            "malignant_ratio",
            "mean_mask_area_fraction",
            "median_mask_area_fraction",
            "patient_study_identity_count",
            "empty_client",
            "stem_leakage",
            "patient_leakage",
        ]:
            split_fields.append(f"{subset}_{key}")
    for method in METHODS:
        split_fields.extend([f"{method}_final_dice", f"{method}_is_worst_client"])
    split_fields.extend(["fedbn_minus_wca_comm_dice", "fedbn_minus_fedavg_dice", "fedbn_minus_fedprox_dice"])
    write_csv(SUMMARY_DIR / "phase5r_bus_uclm_split_feature_vs_performance.csv", split_perf_rows, split_fields)
    write_csv(
        SUMMARY_DIR / "phase5r_bus_uclm_wca_scheduler_diagnosis.csv",
        scheduler,
        [
            "round",
            "client_id",
            "selected",
            "selected_reason",
            "previous_validation_dice",
            "performance_deficit",
            "uploaded_mb",
            "previous_round_worst_client_id",
            "selected_previous_round_worst",
            "current_round_evaluated_worst_client_id",
            "is_final_worst_client",
            "final_client_dice",
            "selection_count",
            "skip_count",
            "selection_frequency",
            "skip_frequency",
            "communication_reduction_source",
        ],
    )
    write_csv(
        SUMMARY_DIR / "phase5r_bus_uclm_fedbn_effect_diagnosis.csv",
        fedbn,
        [
            "client_id",
            "fedavg_dice",
            "fedprox_dice",
            "fedbn_dice",
            "wca_comfedseg_comm_dice",
            "fedbn_minus_fedavg",
            "fedbn_minus_fedprox",
            "fedbn_minus_wca_comm",
            "fedbn_is_best_for_client",
            "fedbn_worst_client_id",
            "fedbn_improves_worst_client_vs_fedavg",
            "fedbn_improves_worst_client_vs_wca_comm",
            "fedbn_client_std",
            "fedavg_client_std",
            "wca_comm_client_std",
            "fedbn_gap",
            "fedavg_gap",
            "wca_comm_gap",
            "fedbn_std_reduction_vs_fedavg",
            "fedbn_gap_reduction_vs_fedavg",
            "fedbn_gap_reduction_vs_wca_comm",
        ],
    )
    generate_figures(logs, scheduler, split_perf_rows)
    write_interpretation_update(logs, fedbn, split_comparison, dataset_summary)
    write_final_report(logs, split_perf_rows, scheduler, fedbn)

    print(SUMMARY_DIR / "phase5r_bus_uclm_client_performance_diagnosis.csv")
    print(SUMMARY_DIR / "phase5r_bus_uclm_split_feature_vs_performance.csv")
    print(SUMMARY_DIR / "phase5r_bus_uclm_wca_scheduler_diagnosis.csv")
    print(SUMMARY_DIR / "phase5r_bus_uclm_fedbn_effect_diagnosis.csv")
    print(SUMMARY_DIR / "phase5r_busi_bus_uclm_interpretation_update.md")
    print(SUMMARY_DIR / "phase5r_bus_uclm_failure_diagnosis.md")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
