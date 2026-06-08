"""Compare Phase 5Q BUS-UCLM moderate_noniid pilot runs.

This script reads only saved Phase 5Q logs. It does not train models and does
not mix BUSI, Kvasir, IID, hard_noniid, or historical failed variants.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.method_registry import load_method_registry


LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5q_bus_uclm_moderate_pilot"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5q_bus_uclm_moderate_pilot"

DATASET = "bus_uclm"
SPLIT = "moderate_noniid"
SEED = 42
METHODS = ["fedavg", "fedprox", "fedbn", "wca_comfedseg_comm"]
APPROVED_METHODS = set(METHODS)
FAILED_METHODS = {"wca_comfedseg_smooth", "wca_comfedseg_pbn", "wca_comfedseg_rg"}
DISALLOWED_METHODS = FAILED_METHODS | {
    "busi",
    "kvasir_seg",
    "local",
    "centralized",
    "wca_comfedseg",
    "wca_comfedseg_prox",
    "wca_comfedseg_bn",
    "wca_comfedseg_comm_conservative",
    "wca_comfedseg_prox_comm_cons",
}

TABLE_FIELDS = [
    "method",
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
    "csv_log_path",
    "communication_log_path",
    "summary_path",
    "method_role",
    "include_in_main_table",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric value: {value}")
    return f"{number:.10f}".rstrip("0").rstrip(".")


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def resolve_logged_path(value: str, fallback: Path) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return fallback


def expected_keys() -> set[tuple[str, int]]:
    return {(method, SEED) for method in METHODS}


def collect_summaries() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], list[Path]], list[Path], list[Path]]:
    grouped: dict[tuple[str, int], list[Path]] = {}
    disallowed_paths: list[Path] = []
    incomplete_dirs: list[Path] = []
    if not LOG_ROOT.exists():
        return {}, grouped, disallowed_paths, incomplete_dirs

    for run_dir in sorted(path for path in LOG_ROOT.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            incomplete_dirs.append(run_dir)
            continue
        summary = read_json(summary_path)
        args = summary.get("args", {})
        key = (str(summary.get("method", "")), int(args.get("seed", -1)))
        scope_ok = (
            summary.get("dataset") == DATASET
            and summary.get("split") == SPLIT
            and key in expected_keys()
            and key[0] in APPROVED_METHODS
        )
        if not scope_ok or key[0] in DISALLOWED_METHODS:
            disallowed_paths.append(summary_path)
            continue
        grouped.setdefault(key, []).append(summary_path)

    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for key, paths in grouped.items():
        selected = sorted(paths, key=lambda item: item.stat().st_mtime)[-1]
        summary = read_json(selected)
        summary["_summary_path"] = rel(selected)
        summary["_run_dir"] = rel(selected.parent)
        latest[key] = summary
    return latest, grouped, disallowed_paths, incomplete_dirs


def full_upload_reference(summaries: dict[tuple[str, int], dict[str, Any]]) -> float | None:
    wca = summaries.get(("wca_comfedseg_comm", SEED))
    if not wca:
        return None
    value = wca.get("full_participation_uploaded_mb", "")
    return float(value) if value not in {"", None} else None


def uploaded_mb(summary: dict[str, Any], full_reference: float | None) -> float | str:
    value = summary.get("total_uploaded_mb", "")
    if value not in {"", None}:
        return float(value)
    return full_reference if full_reference is not None else ""


def communication_reduction(summary: dict[str, Any]) -> float:
    value = summary.get("communication_reduction_vs_full_participation_percent", "")
    return float(value) if value not in {"", None} else 0.0


def table_rows(summaries: dict[tuple[str, int], dict[str, Any]], registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    full_reference = full_upload_reference(summaries)
    rows = []
    for method in METHODS:
        summary = summaries.get((method, SEED))
        if summary is None:
            continue
        args = summary.get("args", {})
        rows.append(
            {
                "method": method,
                "dataset": summary.get("dataset", ""),
                "split": summary.get("split", ""),
                "seed": args.get("seed", ""),
                "rounds": args.get("rounds", summary.get("final_round", "")),
                "local_epochs": args.get("local_epochs", ""),
                "batch_size": args.get("batch_size", ""),
                "image_size": args.get("image_size", ""),
                "number_of_clients": summary.get("number_of_clients", ""),
                "average_dice": fmt(summary.get("average_dice", "")),
                "average_iou": fmt(summary.get("average_iou", "")),
                "worst_client_id": summary.get("worst_client_id", ""),
                "worst_client_dice": fmt(summary.get("worst_client_dice", "")),
                "best_client_dice": fmt(summary.get("best_client_dice", "")),
                "client_dice_std": fmt(summary.get("client_dice_std", "")),
                "best_worst_gap": fmt(summary.get("best_worst_gap", "")),
                "average_loss": fmt(summary.get("average_loss", "")),
                "total_uploaded_mb": fmt(uploaded_mb(summary, full_reference)),
                "communication_reduction_vs_full_participation_percent": fmt(communication_reduction(summary)),
                "csv_log_path": summary.get("csv_log_path", ""),
                "communication_log_path": summary.get("communication_log_path", ""),
                "summary_path": summary.get("_summary_path", ""),
                "method_role": registry.get(method, {}).get("role", ""),
                "include_in_main_table": registry.get(method, {}).get("include_in_main_table", ""),
            }
        )
    return rows


def numeric_summary_values(summary: dict[str, Any]) -> list[float]:
    keys = ["average_dice", "average_iou", "worst_client_dice", "client_dice_std", "best_worst_gap", "average_loss"]
    return [float(summary[key]) for key in keys if summary.get(key, "") not in {"", None}]


def has_nan_or_inf(summaries: dict[tuple[str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        for value in numeric_summary_values(summary):
            if not math.isfinite(value):
                return True
        metrics_path = resolve_logged_path(summary.get("csv_log_path", ""), PROJECT_ROOT / summary["_run_dir"] / "metrics.csv")
        if not metrics_path.exists():
            return True
        for row in read_csv(metrics_path):
            for key in ["dice", "iou", "loss"]:
                if not math.isfinite(float(row[key])):
                    return True
    return False


def metrics_logs_ok(summaries: dict[tuple[str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        metrics_path = resolve_logged_path(summary.get("csv_log_path", ""), PROJECT_ROOT / summary["_run_dir"] / "metrics.csv")
        if not metrics_path.exists() or not read_csv(metrics_path):
            return False
    return True


def wca_communication_log_ok(summaries: dict[tuple[str, int], dict[str, Any]]) -> bool:
    summary = summaries.get(("wca_comfedseg_comm", SEED))
    if summary is None:
        return False
    log_path = resolve_logged_path(summary.get("communication_log_path", ""), Path(""))
    if not summary.get("communication_log_path") or not log_path.exists():
        return False
    rows = read_csv(log_path)
    return bool(rows) and communication_reduction(summary) > 0.0


def normal_deferred_ok(summary: dict[str, Any]) -> bool:
    dataset_meta = summary.get("split_configuration", {}).get("dataset", {})
    classes_used = set(dataset_meta.get("classes_used", []))
    classes_deferred = set(dataset_meta.get("classes_deferred", []))
    included_counts = dataset_meta.get("included_counts", {})
    return (
        "Normal" not in classes_used
        and "Normal" in classes_deferred
        and "Normal" not in included_counts
        and int(dataset_meta.get("deferred_total", 0)) >= 1
    )


def no_empty_clients_or_leakage(summaries: dict[tuple[str, int], dict[str, Any]]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for key, summary in summaries.items():
        split_config = summary.get("split_configuration", {})
        leakage = split_config.get("leakage_check", {})
        if leakage.get("passed") is not True:
            problems.append(f"{key}: leakage_check did not pass")
        for client in split_config.get("clients", []):
            if client.get("empty_client") is True:
                problems.append(f"{key}: empty client in {client}")
    return not problems, problems


def only_expected_settings(summaries: dict[tuple[str, int], dict[str, Any]]) -> bool:
    for summary in summaries.values():
        args = summary.get("args", {})
        if summary.get("dataset") != DATASET or summary.get("split") != SPLIT:
            return False
        if int(args.get("seed", -1)) != SEED:
            return False
        if int(args.get("image_size", -1)) != 128:
            return False
        if int(args.get("clients", -1)) != 3:
            return False
        if int(args.get("rounds", -1)) != 10:
            return False
        if int(args.get("local_epochs", -1)) != 2:
            return False
        if int(args.get("batch_size", -1)) != 4:
            return False
    return True


def best_method(rows: list[dict[str, Any]], metric: str, higher_is_better: bool = True) -> str:
    return sorted(rows, key=lambda row: float(row[metric]), reverse=higher_is_better)[0]["method"]


def row_for(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    return next(row for row in rows if row["method"] == method)


def wca_advantage(rows: list[dict[str, Any]], metric: str, higher_is_better: bool = True) -> bool:
    wca_value = float(row_for(rows, "wca_comfedseg_comm")[metric])
    baseline_values = [float(row[metric]) for row in rows if row["method"] != "wca_comfedseg_comm"]
    return wca_value > max(baseline_values) if higher_is_better else wca_value < min(baseline_values)


def generate_figures(rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("average_dice", "Average Dice", "phase5q_bus_uclm_average_dice.png", True),
        ("worst_client_dice", "Worst-client Dice", "phase5q_bus_uclm_worst_client_dice.png", True),
        ("best_worst_gap", "Best-worst Client Dice Gap", "phase5q_bus_uclm_best_worst_gap.png", False),
        ("total_uploaded_mb", "Uploaded MB", "phase5q_bus_uclm_uploaded_mb.png", False),
        (
            "communication_reduction_vs_full_participation_percent",
            "Communication Reduction (%)",
            "phase5q_bus_uclm_communication_reduction.png",
            True,
        ),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    labels = [row["method"].replace("wca_comfedseg_comm", "wca_comm") for row in rows]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    for metric, title, filename, _higher in specs:
        values = [float(row[metric] or 0.0) for row in rows]
        width, height = 920, 560
        margin_left, margin_right, margin_top, margin_bottom = 90, 40, 80, 110
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        max_value = max(values) if values else 1.0
        max_value = max(max_value, 1e-8)
        if metric in {"average_dice", "worst_client_dice"}:
            max_value = max(max_value, 1.0)

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 26), f"Phase 5Q BUS-UCLM {title}", fill="black", font=title_font)
        draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
        draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)
        draw.text((16, margin_top + 8), title, fill="black", font=font)

        bar_gap = 36
        bar_width = int((plot_w - bar_gap * (len(values) + 1)) / max(len(values), 1))
        for idx, (label, value) in enumerate(zip(labels, values)):
            x0 = margin_left + bar_gap + idx * (bar_width + bar_gap)
            x1 = x0 + bar_width
            bar_h = int((value / max_value) * (plot_h - 24))
            y0 = margin_top + plot_h - bar_h
            y1 = margin_top + plot_h
            draw.rectangle((x0, y0, x1, y1), fill=colors[idx % len(colors)])
            draw.text((x0, max(margin_top, y0 - 24)), f"{value:.4g}", fill="black", font=font)
            draw.text((x0, y1 + 14), label, fill="black", font=font)
        image.save(FIGURE_DIR / filename)


def status_from_checks(checks: dict[str, bool], rows: list[dict[str, Any]]) -> tuple[str, bool, bool, bool]:
    technical_pass = all(checks.values())
    if not technical_pass:
        return "failed", False, False, True

    wca_comm_reduced = float(row_for(rows, "wca_comfedseg_comm")["communication_reduction_vs_full_participation_percent"]) > 0
    wca_avg_best = best_method(rows, "average_dice") == "wca_comfedseg_comm"
    wca_worst_best = best_method(rows, "worst_client_dice") == "wca_comfedseg_comm"
    wca_gap_best = best_method(rows, "best_worst_gap", higher_is_better=False) == "wca_comfedseg_comm"
    wca_strong = wca_comm_reduced and wca_worst_best and wca_gap_best
    needs_diagnosis = not (wca_avg_best and wca_worst_best and wca_gap_best)
    recommend_multiseed = wca_strong
    return "passed", recommend_multiseed, wca_strong, needs_diagnosis


def write_reports(
    summaries: dict[tuple[str, int], dict[str, Any]],
    grouped: dict[tuple[str, int], list[Path]],
    disallowed_paths: list[Path],
    incomplete_dirs: list[Path],
    rows: list[dict[str, Any]],
) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    completed = set(summaries)
    missing = sorted(expected_keys() - completed)
    duplicate_groups = {key: paths for key, paths in grouped.items() if len(paths) > 1}
    empty_leakage_ok, empty_leakage_problems = no_empty_clients_or_leakage(summaries)
    methods_seen = {summary.get("method") for summary in summaries.values()}
    datasets_seen = {summary.get("dataset") for summary in summaries.values()}
    splits_seen = {summary.get("split") for summary in summaries.values()}
    seeds_seen = {int(summary.get("args", {}).get("seed", -1)) for summary in summaries.values()}
    image_sizes_seen = {int(summary.get("args", {}).get("image_size", -1)) for summary in summaries.values()}

    checks = {
        "completed_4_of_4": completed == expected_keys(),
        "only_bus_uclm": datasets_seen == {DATASET},
        "only_moderate_noniid": splits_seen == {SPLIT},
        "only_seed_42": seeds_seen == {SEED},
        "image_size_128_only": image_sizes_seen == {128},
        "normal_deferred": bool(summaries) and all(normal_deferred_ok(summary) for summary in summaries.values()),
        "no_failed_variants": not (methods_seen & FAILED_METHODS),
        "approved_methods_only": methods_seen <= APPROVED_METHODS and not disallowed_paths,
        "no_nan_or_inf": not has_nan_or_inf(summaries),
        "summary_and_metrics_present": len(incomplete_dirs) == 0 and metrics_logs_ok(summaries),
        "wca_communication_log_present": wca_communication_log_ok(summaries),
        "wca_communication_reduction_present": (
            ("wca_comfedseg_comm", SEED) in summaries
            and communication_reduction(summaries[("wca_comfedseg_comm", SEED)]) > 0
        ),
        "no_empty_client_or_leakage": empty_leakage_ok,
        "no_training_failure_or_interruption": len(incomplete_dirs) == 0 and not missing,
        "expected_settings_only": only_expected_settings(summaries),
    }
    phase_status, recommend_multiseed, wca_strong, needs_diagnosis = status_from_checks(checks, rows)

    best_avg = best_method(rows, "average_dice") if rows else ""
    best_worst = best_method(rows, "worst_client_dice") if rows else ""
    best_gap = best_method(rows, "best_worst_gap", higher_is_better=False) if rows else ""
    wca_row = row_for(rows, "wca_comfedseg_comm") if any(row["method"] == "wca_comfedseg_comm" for row in rows) else {}
    wca_comm_reduced = bool(wca_row) and float(wca_row["communication_reduction_vs_full_participation_percent"]) > 0.0
    wca_worst_advantage = wca_advantage(rows, "worst_client_dice") if rows and len(rows) == len(METHODS) else False
    wca_gap_advantage = wca_advantage(rows, "best_worst_gap", higher_is_better=False) if rows and len(rows) == len(METHODS) else False
    wca_avg_advantage = wca_advantage(rows, "average_dice") if rows and len(rows) == len(METHODS) else False

    verification_lines = [
        "# Phase 5Q BUS-UCLM Moderate Non-IID Pilot Verification",
        "",
        "Phase 5Q is a BUS-UCLM moderate_noniid pilot only. It is not final SCI evidence.",
        "",
        "## Scope",
        "",
        "- Dataset: `bus_uclm` only.",
        "- Split: `moderate_noniid` only.",
        "- Methods: `fedavg`, `fedprox`, `fedbn`, `wca_comfedseg_comm`.",
        "- Seed: 42 only.",
        "- Fixed settings: clients=3, rounds=10, local_epochs=2, batch_size=4, image_size=128.",
        "- Normal cases remain deferred.",
        "",
        "## Verification Checks",
        "",
    ]
    verification_lines.extend(f"- {name}: {'passed' if ok else 'failed'}" for name, ok in checks.items())
    verification_lines.extend(
        [
            f"- Missing runs: {missing if missing else 'none'}",
            f"- Duplicate run groups: {len(duplicate_groups)}",
            f"- Incomplete run directories: {[rel(path) for path in incomplete_dirs] if incomplete_dirs else 'none'}",
            f"- Disallowed/extra summaries: {[rel(path) for path in disallowed_paths] if disallowed_paths else 'none'}",
            f"- Empty/leakage problems: {empty_leakage_problems if empty_leakage_problems else 'none'}",
            "",
            "## Pilot Results",
            "",
            f"- Best average Dice method: `{best_avg}`.",
            f"- Best worst-client Dice method: `{best_worst}`.",
            f"- Smallest best-worst gap method: `{best_gap}`.",
            f"- WCA-Comm communication reduction: {'yes' if wca_comm_reduced else 'no'}"
            + (f" ({wca_row.get('communication_reduction_vs_full_participation_percent', '')}%)." if wca_row else "."),
            f"- WCA-Comm average Dice advantage over all baselines: {wca_avg_advantage}.",
            f"- WCA-Comm worst-client Dice advantage over all baselines: {wca_worst_advantage}.",
            f"- WCA-Comm best-worst gap advantage over all baselines: {wca_gap_advantage}.",
            "",
            "## BUSI Moderate Non-IID Relationship",
            "",
        ]
    )
    if wca_strong:
        verification_lines.extend(
            [
                "- WCA-Comm is also strong in this BUS-UCLM pilot, so the ultrasound-focused line is strengthened.",
                "- Recommendation: proceed to explicitly approved BUS-UCLM moderate_noniid multi-seed validation.",
            ]
        )
    elif wca_comm_reduced and not (wca_worst_advantage and wca_gap_advantage):
        verification_lines.extend(
            [
                "- WCA-Comm reduces communication, but Dice/fairness superiority is not strong in this pilot.",
                "- Recommendation: treat BUS-UCLM as an important boundary and diagnose before multi-seed validation.",
            ]
        )
    else:
        verification_lines.extend(
            [
                "- WCA-Comm does not show the required communication/fairness signal in this pilot.",
                "- Recommendation: mark as needs diagnosis or failed rather than promoting it as success evidence.",
            ]
        )
    verification_lines.extend(
        [
            "",
            "## Phase 5Q Decision",
            "",
            f"- Phase 5Q status: {phase_status}.",
            f"- Recommend BUS-UCLM moderate_noniid multi-seed: {recommend_multiseed}.",
            f"- Continue ultrasound-focused paper direction: {phase_status == 'passed'}.",
            f"- Need BUS-UCLM split / WCA scheduler / FedBN-effect diagnosis: {needs_diagnosis}.",
            "- Pilot evidence: run completion, real BUS-UCLM moderate_noniid metrics, communication log behavior.",
            "- Not final claim: any SCI-level multi-seed or two-ultrasound-dataset conclusion.",
        ]
    )
    (SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot_verification.md").write_text(
        "\n".join(verification_lines) + "\n",
        encoding="utf-8",
    )

    notes_lines = [
        "# Phase 5Q Notes",
        "",
        "Phase 5Q is a single-seed BUS-UCLM moderate_noniid pilot.",
        "",
        "- This phase does not run BUSI, Kvasir, IID, hard_noniid, Normal cases, or failed variants.",
        "- This phase does not modify model structure, loss, optimizer, server/client training logic, or pretrained backbone usage.",
        "- All generated files use the `phase5q` prefix.",
        "- Results can guide whether BUS-UCLM deserves multi-seed validation.",
        "- Results cannot be used as final SCI claims.",
        "",
        f"Phase 5Q status: {phase_status}.",
        f"Recommend multi-seed: {recommend_multiseed}.",
        f"Need diagnosis before multi-seed: {needs_diagnosis}.",
    ]
    (SUMMARY_DIR / "phase5q_notes.md").write_text("\n".join(notes_lines) + "\n", encoding="utf-8")


def main() -> None:
    registry = load_method_registry()
    for method in METHODS:
        entry = registry.get(method, {})
        if entry.get("status") != "active" or entry.get("run_on_real_dataset_default") is not True:
            raise RuntimeError(f"Method is not approved for this pilot: {method}")
    if FAILED_METHODS & APPROVED_METHODS:
        raise RuntimeError(f"Failed variants are included unexpectedly: {FAILED_METHODS & APPROVED_METHODS}")

    summaries, grouped, disallowed_paths, incomplete_dirs = collect_summaries()
    rows = table_rows(summaries, registry)
    write_csv(SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot.csv", rows, TABLE_FIELDS)
    if rows:
        generate_figures(rows)
    write_reports(summaries, grouped, disallowed_paths, incomplete_dirs, rows)

    print(SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot.csv")
    print(SUMMARY_DIR / "phase5q_bus_uclm_moderate_pilot_verification.md")
    print(SUMMARY_DIR / "phase5q_notes.md")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
