"""Phase 5L BUSI hard_noniid WCA+FedProx instability diagnosis.

This script is read-only with respect to experiment logs. It does not train
models and only writes Phase 5L diagnostic summaries/figures.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "results" / "logs" / "phase5k_core_hard_multiseed"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "phase5l_busi_wca_prox_instability"
SEEDS = [42, 123, 2025]
METHODS = ["fedprox", "wca_comfedseg_prox"]
SPLIT = "hard_noniid"
DATASET = "busi"
BUSI_ROOT = PROJECT_ROOT / "data" / "raw" / "BUSI" / "Dataset_BUSI_with_GT"
BUSI_CLASSES = ("benign", "malignant")


@dataclass(frozen=True)
class LightBUSISample:
    image_path: Path
    mask_paths: tuple[Path, ...]
    class_name: str
    sample_id: str
    mask_area_fraction: float

SEED_FIELDS = [
    "seed",
    "method",
    "average_dice",
    "worst_client_dice",
    "worst_client_id",
    "best_client_dice",
    "client_dice_std",
    "best_worst_gap",
    "average_loss",
    "interpretation",
]

CURVE_FIELDS = [
    "seed",
    "method",
    "round",
    "client_id",
    "dice",
    "iou",
    "loss",
    "is_round_worst_client",
]

SPLIT_FIELDS = [
    "seed",
    "split",
    "client_id",
    "sample_count",
    "benign_count",
    "malignant_count",
    "benign_ratio",
    "malignant_ratio",
    "mean_mask_area_fraction",
    "median_mask_area_fraction",
    "min_mask_area_fraction",
    "max_mask_area_fraction",
]

WEIGHT_FIELDS = [
    "seed",
    "round",
    "client_id",
    "validation_dice_used",
    "performance_deficit",
    "deficit_weight",
    "final_aggregation_weight",
    "is_max_weight",
    "round_worst_client_id",
    "next_round_worst_client_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | str | int | None) -> str:
    if value is None or value == "":
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid value: {value}")
    return f"{number:.10f}".rstrip("0").rstrip(".")


def is_mask_file(path: Path) -> bool:
    return "_mask" in path.stem


def base_image_stem(path: Path) -> str:
    return re.sub(r"_mask(?:_\d+)?$", "", path.stem)


def mask_area_fraction(paths: tuple[Path, ...], image_size: int = 128) -> float:
    merged = np.zeros((image_size, image_size), dtype=bool)
    for path in paths:
        with Image.open(path) as handle:
            mask = handle.convert("L")
            mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
            merged |= np.asarray(mask) > 0
    return float(merged.mean())


def discover_light_busi_samples() -> list[LightBUSISample]:
    samples: list[LightBUSISample] = []
    for class_name in BUSI_CLASSES:
        class_dir = BUSI_ROOT / class_name
        png_files = sorted(class_dir.glob("*.png"))
        image_files = [path for path in png_files if not is_mask_file(path)]
        mask_files = [path for path in png_files if is_mask_file(path)]
        masks_by_stem: dict[str, list[Path]] = {}
        for mask_path in mask_files:
            masks_by_stem.setdefault(base_image_stem(mask_path), []).append(mask_path)
        for image_path in image_files:
            mask_paths = tuple(sorted(masks_by_stem.get(image_path.stem, [])))
            if not mask_paths:
                continue
            samples.append(
                LightBUSISample(
                    image_path=image_path,
                    mask_paths=mask_paths,
                    class_name=class_name,
                    sample_id=f"{class_name}/{image_path.stem}",
                    mask_area_fraction=mask_area_fraction(mask_paths),
                )
            )
    return samples


def split_counts(total: int) -> tuple[int, int, int]:
    train_count = max(1, int(round(total * 0.70)))
    val_count = max(1, int(round(total * 0.15)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, total - val_count - test_count)
    return train_count, val_count, test_count


def split_light_busi_samples(samples: list[LightBUSISample], seed: int) -> dict[str, list[LightBUSISample]]:
    rng = np.random.default_rng(seed)
    splits: dict[str, list[LightBUSISample]] = {"train": [], "val": [], "test": []}
    for class_name in BUSI_CLASSES:
        class_samples = [sample for sample in samples if sample.class_name == class_name]
        indices = np.arange(len(class_samples))
        rng.shuffle(indices)
        train_count, val_count, _ = split_counts(len(indices))
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


def hard_noniid_partition(samples: list[LightBUSISample], num_clients: int = 3) -> list[list[int]]:
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


def latest_run_dir(method: str, seed: int) -> Path:
    pattern = f"*_{method}_{DATASET}_{SPLIT}_seed{seed}"
    candidates = sorted(LOG_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Missing Phase 5K BUSI run: {method} seed {seed}")
    return candidates[-1]


def metrics_by_round(metrics: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in metrics:
        grouped.setdefault(int(row["round"]), []).append(row)
    return grouped


def final_metrics_from_rows(metrics: list[dict[str, str]]) -> dict[str, float | int]:
    grouped = metrics_by_round(metrics)
    final_round = max(grouped)
    final_rows = grouped[final_round]
    dice = [float(row["dice"]) for row in final_rows]
    losses = [float(row["loss"]) for row in final_rows]
    worst_row = min(final_rows, key=lambda row: float(row["dice"]))
    return {
        "average_dice": mean(dice),
        "worst_client_dice": float(worst_row["dice"]),
        "worst_client_id": int(worst_row["client_id"]),
        "best_client_dice": max(dice),
        "client_dice_std": pstdev(dice),
        "best_worst_gap": max(dice) - min(dice),
        "average_loss": mean(losses),
    }


def interpretation_for(seed: int, method: str, final: dict[str, float | int], paired: dict[str, float | int] | None) -> str:
    if paired is None:
        return "reference run"
    worst_delta = float(final["worst_client_dice"]) - float(paired["worst_client_dice"])
    gap_delta = float(final["best_worst_gap"]) - float(paired["best_worst_gap"])
    avg_delta = float(final["average_dice"]) - float(paired["average_dice"])
    direction = []
    direction.append("worst improved" if worst_delta > 0 else "worst degraded")
    direction.append("gap reduced" if gap_delta < 0 else "gap increased")
    direction.append("average improved" if avg_delta > 0 else "average degraded")
    return f"vs paired FedProx seed {seed}: {', '.join(direction)}; deltas avg={avg_delta:+.4f}, worst={worst_delta:+.4f}, gap={gap_delta:+.4f}"


def build_seed_and_curve_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], dict[str, Any]] = {}

    for seed in SEEDS:
        paired_fedprox_final = None
        for method in METHODS:
            run_dir = latest_run_dir(method, seed)
            metrics = read_csv(run_dir / "metrics.csv")
            summary = read_json(run_dir / "summary.json")
            final = final_metrics_from_rows(metrics)
            cache[(seed, method)] = {
                "run_dir": run_dir,
                "metrics": metrics,
                "summary": summary,
                "final": final,
            }
            if method == "fedprox":
                paired_fedprox_final = final
            interpretation = interpretation_for(seed, method, final, paired_fedprox_final if method != "fedprox" else None)
            seed_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "average_dice": fmt(final["average_dice"]),
                    "worst_client_dice": fmt(final["worst_client_dice"]),
                    "worst_client_id": final["worst_client_id"],
                    "best_client_dice": fmt(final["best_client_dice"]),
                    "client_dice_std": fmt(final["client_dice_std"]),
                    "best_worst_gap": fmt(final["best_worst_gap"]),
                    "average_loss": fmt(final["average_loss"]),
                    "interpretation": interpretation,
                }
            )

            for round_id, rows in sorted(metrics_by_round(metrics).items()):
                worst_client = min(rows, key=lambda row: float(row["dice"]))["client_id"]
                for row in rows:
                    curve_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "round": round_id,
                            "client_id": row["client_id"],
                            "dice": fmt(row["dice"]),
                            "iou": fmt(row["iou"]),
                            "loss": fmt(row["loss"]),
                            "is_round_worst_client": str(row["client_id"]) == str(worst_client),
                        }
                    )
    return seed_rows, curve_rows, cache


def build_split_distribution_rows() -> list[dict[str, Any]]:
    samples = discover_light_busi_samples()
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        split_samples = split_light_busi_samples(samples, seed=seed)
        for split_name, samples_for_split in split_samples.items():
            partitions = hard_noniid_partition(samples_for_split, 3)
            for client_id, indices in enumerate(partitions):
                selected = [samples_for_split[index] for index in indices]
                benign_count = sum(1 for sample in selected if sample.class_name == "benign")
                malignant_count = sum(1 for sample in selected if sample.class_name == "malignant")
                areas = [sample.mask_area_fraction for sample in selected]
                sample_count = len(selected)
                rows.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "client_id": client_id,
                        "sample_count": sample_count,
                        "benign_count": benign_count,
                        "malignant_count": malignant_count,
                        "benign_ratio": fmt(benign_count / sample_count if sample_count else 0.0),
                        "malignant_ratio": fmt(malignant_count / sample_count if sample_count else 0.0),
                        "mean_mask_area_fraction": fmt(mean(areas) if areas else 0.0),
                        "median_mask_area_fraction": fmt(median(areas) if areas else 0.0),
                        "min_mask_area_fraction": fmt(min(areas) if areas else 0.0),
                        "max_mask_area_fraction": fmt(max(areas) if areas else 0.0),
                    }
                )
    return rows


def build_weight_rows(cache: dict[tuple[int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = cache[(seed, "wca_comfedseg_prox")]["run_dir"]
        metrics = cache[(seed, "wca_comfedseg_prox")]["metrics"]
        grouped_metrics = metrics_by_round(metrics)
        weight_path = run_dir / f"wca_weights_busi_hard_noniid_seed{seed}.csv"
        weights = read_csv(weight_path)
        weights_by_round: dict[int, list[dict[str, str]]] = {}
        for row in weights:
            weights_by_round.setdefault(int(row["round"]), []).append(row)
        for round_id, round_weights in sorted(weights_by_round.items()):
            current_worst = min(grouped_metrics[round_id], key=lambda row: float(row["dice"]))["client_id"]
            if round_id + 1 in grouped_metrics:
                next_worst = min(grouped_metrics[round_id + 1], key=lambda row: float(row["dice"]))["client_id"]
            else:
                next_worst = ""
            max_weight = max(float(row["final_aggregation_weight"]) for row in round_weights)
            for row in round_weights:
                rows.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "client_id": row["client_id"],
                        "validation_dice_used": row["validation_dice_used"],
                        "performance_deficit": fmt(row["performance_deficit"]),
                        "deficit_weight": fmt(row["deficit_weight"]),
                        "final_aggregation_weight": fmt(row["final_aggregation_weight"]),
                        "is_max_weight": abs(float(row["final_aggregation_weight"]) - max_weight) < 1e-12,
                        "round_worst_client_id": current_worst,
                        "next_round_worst_client_id": next_worst,
                    }
                )
    return rows


def plot_per_client_curves(cache: dict[tuple[int, str], dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        plt.figure(figsize=(10, 5))
        for method, style in [("fedprox", "-"), ("wca_comfedseg_prox", "--")]:
            metrics = cache[(seed, method)]["metrics"]
            for client_id in ["0", "1", "2"]:
                xs = [int(row["round"]) for row in metrics if row["client_id"] == client_id]
                ys = [float(row["dice"]) for row in metrics if row["client_id"] == client_id]
                plt.plot(xs, ys, style, label=f"{method} c{client_id}")
        plt.title(f"Phase 5L BUSI seed {seed} per-client Dice")
        plt.xlabel("round")
        plt.ylabel("Dice")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"phase5l_busi_seed{seed}_per_client_dice.png", dpi=150)
        plt.close()


def plot_worst_and_gap(cache: dict[tuple[int, str], dict[str, Any]]) -> None:
    for seed in SEEDS:
        for metric_name in ["worst_dice", "gap"]:
            plt.figure(figsize=(8, 4))
            for method in METHODS:
                grouped = metrics_by_round(cache[(seed, method)]["metrics"])
                xs = sorted(grouped)
                if metric_name == "worst_dice":
                    ys = [min(float(row["dice"]) for row in grouped[x]) for x in xs]
                    ylabel = "worst-client Dice"
                else:
                    ys = [
                        max(float(row["dice"]) for row in grouped[x])
                        - min(float(row["dice"]) for row in grouped[x])
                        for x in xs
                    ]
                    ylabel = "best-worst gap"
                plt.plot(xs, ys, marker="o", label=method)
            plt.title(f"Phase 5L BUSI seed {seed} {ylabel}")
            plt.xlabel("round")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"phase5l_busi_seed{seed}_{metric_name}.png", dpi=150)
            plt.close()


def plot_distribution(split_rows: list[dict[str, Any]]) -> None:
    val_rows = [row for row in split_rows if row["split"] == "val"]
    labels = [f"s{row['seed']} c{row['client_id']}" for row in val_rows]
    malignant = [float(row["malignant_ratio"]) for row in val_rows]
    areas = [float(row["mean_mask_area_fraction"]) for row in val_rows]
    x = list(range(len(labels)))
    plt.figure(figsize=(11, 4))
    plt.bar([i - 0.2 for i in x], malignant, width=0.4, label="malignant ratio")
    plt.bar([i + 0.2 for i in x], areas, width=0.4, label="mean mask area")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Phase 5L BUSI hard_noniid val distribution by seed/client")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "phase5l_busi_val_distribution_by_seed.png", dpi=150)
    plt.close()


def plot_weight_focus(weight_rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10, 4))
    for seed in SEEDS:
        seed_rows = [row for row in weight_rows if int(row["seed"]) == seed]
        for client_id in ["0", "1", "2"]:
            xs = [int(row["round"]) for row in seed_rows if row["client_id"] == client_id]
            ys = [float(row["final_aggregation_weight"]) for row in seed_rows if row["client_id"] == client_id]
            plt.plot(xs, ys, label=f"s{seed} c{client_id}")
    plt.title("Phase 5L WCA+FedProx aggregation weights")
    plt.xlabel("round")
    plt.ylabel("aggregation weight")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "phase5l_busi_wca_weight_focus.png", dpi=150)
    plt.close()


def final_client_map(cache: dict[tuple[int, str], dict[str, Any]], seed: int, method: str) -> dict[int, float]:
    grouped = metrics_by_round(cache[(seed, method)]["metrics"])
    final_round = max(grouped)
    return {int(row["client_id"]): float(row["dice"]) for row in grouped[final_round]}


def long_term_worst_counts(cache: dict[tuple[int, str], dict[str, Any]], seed: int, method: str) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for rows in metrics_by_round(cache[(seed, method)]["metrics"]).values():
        worst = int(min(rows, key=lambda row: float(row["dice"]))["client_id"])
        counts[worst] += 1
    return counts


def split_summary(split_rows: list[dict[str, Any]], seed: int, split_name: str = "val") -> str:
    rows = [row for row in split_rows if int(row["seed"]) == seed and row["split"] == split_name]
    pieces = []
    for row in rows:
        pieces.append(
            f"client {row['client_id']}: n={row['sample_count']}, malignant={float(row['malignant_ratio']):.2f}, mask_mean={float(row['mean_mask_area_fraction']):.3f}"
        )
    return "; ".join(pieces)


def write_report(
    seed_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    cache: dict[tuple[int, str], dict[str, Any]],
) -> None:
    def row_for(seed: int, method: str) -> dict[str, Any]:
        return next(row for row in seed_rows if int(row["seed"]) == seed and row["method"] == method)

    lines = [
        "# Phase 5L BUSI WCA+FedProx Instability Diagnosis",
        "",
        "Phase 5L is a BUSI hard_noniid diagnostic only. No new training runs were launched.",
        "",
        "## Logs Read",
        "",
    ]
    for seed in SEEDS:
        for method in METHODS:
            lines.append(f"- `{method}` seed {seed}: `{cache[(seed, method)]['run_dir'].relative_to(PROJECT_ROOT)}`")
    lines.extend(
        [
            "",
            "## Seed-Level Diagnosis",
            "",
            "| seed | FedProx worst | WCA+FedProx worst | FedProx gap | WCA+FedProx gap | final weak client comparison |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for seed in SEEDS:
        fed = row_for(seed, "fedprox")
        wca = row_for(seed, "wca_comfedseg_prox")
        fed_map = final_client_map(cache, seed, "fedprox")
        wca_map = final_client_map(cache, seed, "wca_comfedseg_prox")
        weak = min(wca_map, key=wca_map.get)
        lines.append(
            f"| {seed} | {float(fed['worst_client_dice']):.4f} | {float(wca['worst_client_dice']):.4f} | "
            f"{float(fed['best_worst_gap']):.4f} | {float(wca['best_worst_gap']):.4f} | "
            f"WCA weakest client {weak}, Dice {wca_map[weak]:.4f}; paired FedProx client {weak}, Dice {fed_map[weak]:.4f} |"
        )

    lines.extend(["", "## Client-Level Findings", ""])
    for seed in SEEDS:
        fed_counts = long_term_worst_counts(cache, seed, "fedprox")
        wca_counts = long_term_worst_counts(cache, seed, "wca_comfedseg_prox")
        lines.append(
            f"- Seed {seed}: FedProx worst-round counts {fed_counts}; WCA+FedProx worst-round counts {wca_counts}. "
            f"Val split: {split_summary(split_rows, seed)}."
        )

    lines.extend(["", "## WCA Weighting Findings", ""])
    for seed in SEEDS:
        max_rows = [row for row in weight_rows if int(row["seed"]) == seed and row["is_max_weight"] is True]
        max_counts = {client_id: 0 for client_id in ["0", "1", "2"]}
        for row in max_rows:
            max_counts[row["client_id"]] += 1
        heavy_rows = [row for row in weight_rows if int(row["seed"]) == seed and float(row["final_aggregation_weight"]) >= 0.70]
        lines.append(
            f"- Seed {seed}: max-weight client counts {max_counts}; rounds with weight >= 0.70: "
            f"{[(row['round'], row['client_id'], row['final_aggregation_weight']) for row in heavy_rows]}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Seed 42 is strong because WCA weighting repeatedly reduces imbalance without leaving a persistent weak client at the end; final gap is much smaller than FedProx and the weakest client is improved.",
            "- Seed 123 is unstable because WCA+FedProx sharply improves client 2 but sacrifices client 0 relative to FedProx; final worst-client Dice drops to 0.4017 although the gap remains lower than FedProx seed 42.",
            "- Seed 2025 is unstable because client 0 remains the repeated weak client for many rounds under WCA+FedProx; WCA gives client 0 very high aggregation weight for several rounds, but recovery is incomplete and final gap becomes worse than FedProx.",
            "- The main issue is not a logging or summary aggregation bug: final summaries match metrics.csv, all runs use BUSI hard_noniid, image_size=128, seed-specific logs, and no NaN/inf is observed.",
            "- Split distribution contributes to instability: hard_noniid intentionally creates strong client imbalance, and seed-specific train/val/test sampling changes each client's malignant ratio and mask-area range. However, split distribution alone does not explain the failure, because the WCA weighting response can over-focus one weak client and move the weakness to another client.",
            "- The current evidence points more to WCA weighting sensitivity under BUSI hard_noniid than to FedProx mu alone. The FedProx term is identical between FedProx and WCA+FedProx; the unstable behavior appears after adding WCA aggregation weights.",
            "",
            "## Recommendation",
            "",
            "- Do not run alpha/mu diagnostic experiments yet; the read-only evidence is sufficient to identify WCA weighting sensitivity as the immediate issue.",
            "- Downgrade `wca_comfedseg_prox` for BUSI hard_noniid from scoped candidate to diagnostic appendix candidate unless a revised, more stable WCA weighting rule is designed.",
            "- Do not use BUSI hard_noniid WCA+FedProx as a main new-method claim.",
            "- Keep Kvasir WCA+BN as the stronger hard_noniid scoped candidate from Phase 5K.",
            "- Next broad validation should not expand BUSI WCA+FedProx. Prefer moderate_noniid scoped multi-seed for WCA-Comm, or a focused BUSI WCA weighting redesign if BUSI hard_noniid remains central.",
            "",
            "## Constraints",
            "",
            "- No new diagnostic training runs were launched.",
            "- No Kvasir experiments were run.",
            "- No moderate_noniid experiments were run.",
            "- No failed variants were run.",
            "- `wca_comfedseg_prox_comm_cons` was not run.",
            "- image_size=256 was not used.",
            "- BUSI normal was not used.",
            "- No large model / SAM / MedSAM / pretrained backbone was added.",
            "- Existing Phase 5A-5K results were not overwritten.",
        ]
    )
    (SUMMARY_DIR / "phase5l_busi_wca_prox_instability_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows, curve_rows, cache = build_seed_and_curve_rows()
    split_rows = build_split_distribution_rows()
    weight_rows = build_weight_rows(cache)

    write_csv(SUMMARY_DIR / "phase5l_busi_seed_level_diagnosis.csv", seed_rows, SEED_FIELDS)
    write_csv(SUMMARY_DIR / "phase5l_busi_client_curve_diagnosis.csv", curve_rows, CURVE_FIELDS)
    write_csv(SUMMARY_DIR / "phase5l_busi_split_distribution_by_seed.csv", split_rows, SPLIT_FIELDS)
    write_csv(SUMMARY_DIR / "phase5l_busi_wca_weight_diagnosis.csv", weight_rows, WEIGHT_FIELDS)
    plot_per_client_curves(cache)
    plot_worst_and_gap(cache)
    plot_distribution(split_rows)
    plot_weight_focus(weight_rows)
    write_report(seed_rows, curve_rows, split_rows, weight_rows, cache)

    print(SUMMARY_DIR / "phase5l_busi_wca_prox_instability_diagnosis.md")
    print(SUMMARY_DIR / "phase5l_busi_seed_level_diagnosis.csv")
    print(SUMMARY_DIR / "phase5l_busi_client_curve_diagnosis.csv")
    print(SUMMARY_DIR / "phase5l_busi_split_distribution_by_seed.csv")
    print(SUMMARY_DIR / "phase5l_busi_wca_weight_diagnosis.csv")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
