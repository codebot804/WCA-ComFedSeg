"""Command-line entry point for WCA-ComFedSeg experiments.

The runner supports synthetic checks and real-dataset federated segmentation
experiments for BUSI, BUS-UCLM, and Kvasir-SEG. It records client-level metrics,
WCA aggregation weights, communication logs, and JSON summaries for
reproducible analysis.
"""

import argparse
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from datasets.busi import build_busi_client_loaders
from datasets.bus_uclm import build_bus_uclm_client_loaders
from datasets.kvasir_seg import build_kvasir_client_loaders
from datasets.synthetic_segmentation import build_synthetic_client_loaders
from federated.client import FederatedClient
from federated.server import FederatedServer
from methods.centralized import run_centralized_training
from methods.local import run_local_training
from models.unet import build_unet2d
from utils.logging_utils import make_run_dir, save_csv, save_json
from utils.visualization import save_phase1_prediction_figures, save_phase2_split_distribution_figure


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WCA-ComFedSeg experiment runner.")
    parser.add_argument(
        "--method",
        default="fedavg",
        choices=[
            "local",
            "centralized",
            "fedavg",
            "fedprox",
            "fedbn",
            "wca_comfedseg",
            "wca_comfedseg_prox",
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
            "wca_comfedseg_bn",
            "wca_comfedseg_smooth",
            "wca_comfedseg_pbn",
            "wca_comfedseg_rg",
        ],
        help="Baseline method to run.",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "busi", "bus_uclm", "kvasir_seg"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--split",
        default="iid",
        choices=["iid", "moderate_noniid", "hard_noniid", "extreme_noniid"],
        help="Client split mode.",
    )
    parser.add_argument("--clients", type=int, default=3, help="Number of simulated clients.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of communication rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Client optimizer learning rate.")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size.")
    parser.add_argument("--train-samples", type=int, default=24, help="Training samples per client.")
    parser.add_argument("--val-samples", type=int, default=8, help="Validation samples per client.")
    parser.add_argument("--test-samples", type=int, default=8, help="Test samples per client.")
    parser.add_argument("--base-channels", type=int, default=8, help="Base channels for the small U-Net.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, or auto.")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"], help="Client split evaluated each round.")
    parser.add_argument("--results-dir", default="results/logs", help="Directory for CSV and JSON logs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment config.")
    parser.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx proximal coefficient.")
    parser.add_argument("--debug-fedbn", action="store_true", help="Print FedBN aggregated/local BatchNorm key summary.")
    parser.add_argument("--debug-pbn", action="store_true", help="Print WCA-PBN BatchNorm key partition summary.")
    parser.add_argument("--wca-alpha", type=float, default=0.5, help="WCA aggregation deficit weight coefficient.")
    parser.add_argument("--wca-beta", type=float, default=0.5, help="WCA-Smooth temporal smoothing coefficient.")
    parser.add_argument("--wca-max-weight", type=float, default=0.5, help="WCA-Smooth maximum raw client weight before renormalization.")
    parser.add_argument("--rg-min-gate", type=float, default=0.2, help="Reliability-Gated WCA minimum deficit gate.")
    parser.add_argument("--client-fraction", type=float, default=0.67, help="Fraction of clients selected by the communication scheduler.")
    parser.add_argument("--min-selected-clients", type=int, default=2, help="Minimum selected clients per communication round.")
    parser.add_argument("--comm-scheduler", default="adaptive", choices=["adaptive"], help="Communication scheduler strategy.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def build_clients(
    args: argparse.Namespace,
    device: torch.device,
    model_fn,
) -> tuple[list[FederatedClient], dict]:
    if args.dataset == "synthetic":
        loaders, split_metadata = build_synthetic_client_loaders(
            num_clients=args.clients,
            batch_size=args.batch_size,
            split_mode=args.split,
            image_size=args.image_size,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            seed=args.seed,
        )
    elif args.dataset == "busi":
        loaders, split_metadata = build_busi_client_loaders(
            num_clients=args.clients,
            batch_size=args.batch_size,
            split_mode=args.split,
            image_size=args.image_size,
            seed=args.seed,
        )
    elif args.dataset == "bus_uclm":
        loaders, split_metadata = build_bus_uclm_client_loaders(
            num_clients=args.clients,
            batch_size=args.batch_size,
            split_mode=args.split,
            image_size=args.image_size,
            seed=args.seed,
        )
    elif args.dataset == "kvasir_seg":
        loaders, split_metadata = build_kvasir_client_loaders(
            num_clients=args.clients,
            batch_size=args.batch_size,
            split_mode=args.split,
            image_size=args.image_size,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    clients = [
        FederatedClient(
            client_id=client_id,
            train_loader=client_loaders["train"],
            val_loader=client_loaders["val"],
            test_loader=client_loaders["test"],
            model_fn=model_fn,
            device=device,
            lr=args.lr,
        )
        for client_id, client_loaders in loaders.items()
    ]
    return clients, split_metadata


def add_run_context(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(
            {
                "dataset": args.dataset,
                "clients": args.clients,
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "fedprox_mu": args.fedprox_mu if args.method in {
                    "fedprox",
                    "wca_comfedseg_prox",
                    "wca_comfedseg_prox_comm_cons",
                } else "",
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def build_client_level_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    csv_path: str,
    prediction_figure_dir: str,
    prediction_figure_count: int,
    split_metadata: dict,
    split_figure_path: str,
    wca_weight_log_path: str = "",
    wca_smooth_weight_log_path: str = "",
    wca_pbn_weight_log_path: str = "",
    wca_rg_weight_log_path: str = "",
    communication_log_path: str = "",
    server: FederatedServer | None = None,
) -> dict[str, Any]:
    final_round = max(row["round"] for row in rows)
    final_rows = [row for row in rows if row["round"] == final_round]
    final_dice = [float(row["dice"]) for row in final_rows]
    final_iou = [float(row["iou"]) for row in final_rows]
    final_losses = [float(row["loss"]) for row in final_rows]
    worst_row = min(final_rows, key=lambda row: float(row["dice"]))
    best_client_dice = max(final_dice)
    if args.dataset != "synthetic":
        phase_name = "phase_5_real_dataset_run"
    elif args.method in {
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
    }:
        phase_name = "phase_4b_communication_efficient_wca"
    elif args.method == "wca_comfedseg_rg":
        phase_name = "phase_4f_reliability_gated_wca"
    elif args.method == "wca_comfedseg_pbn":
        phase_name = "phase_4e_wca_personalized_batchnorm"
    elif args.method == "wca_comfedseg_smooth":
        phase_name = "phase_4d_wca_smooth_aggregation"
    elif args.method == "wca_comfedseg":
        phase_name = "phase_4a_worst_client_aware_aggregation"
    else:
        phase_name = "phase_3_synthetic_baseline_validation"

    summary = {
        "phase": phase_name,
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "number_of_clients": args.clients,
        "args": vars(args),
        "csv_log_path": csv_path,
        "summary_note": (
            "Synthetic pipeline check summary."
            if args.dataset == "synthetic"
            else "Real-dataset experiment summary."
        ),
        "prediction_figure_dir": prediction_figure_dir,
        "prediction_figure_count": prediction_figure_count,
        "split_figure_path": split_figure_path,
        "wca_weight_log_path": wca_weight_log_path,
        "wca_smooth_weight_log_path": wca_smooth_weight_log_path,
        "wca_pbn_weight_log_path": wca_pbn_weight_log_path,
        "wca_rg_weight_log_path": wca_rg_weight_log_path,
        "communication_log_path": communication_log_path,
        "wca_alpha": args.wca_alpha if args.method in {
            "wca_comfedseg",
            "wca_comfedseg_prox",
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
            "wca_comfedseg_bn",
            "wca_comfedseg_smooth",
            "wca_comfedseg_pbn",
            "wca_comfedseg_rg",
        } else "",
        "rg_min_gate": args.rg_min_gate if args.method == "wca_comfedseg_rg" else "",
        "wca_beta": args.wca_beta if args.method == "wca_comfedseg_smooth" else "",
        "wca_max_weight": args.wca_max_weight if args.method == "wca_comfedseg_smooth" else "",
        "client_fraction": args.client_fraction if args.method in {
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
        } else "",
        "min_selected_clients": args.min_selected_clients if args.method in {
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
        } else "",
        "split_configuration": split_metadata,
        "final_round": final_round,
        "average_dice": mean(final_dice),
        "average_iou": mean(final_iou),
        "average_loss": mean(final_losses),
        "worst_client_id": int(worst_row["client_id"]),
        "worst_client_dice": float(worst_row["dice"]),
        "best_client_dice": best_client_dice,
        "client_dice_std": pstdev(final_dice) if len(final_dice) > 1 else 0.0,
        "best_worst_gap": best_client_dice - float(worst_row["dice"]),
        "final_client_metrics": final_rows,
    }
    if args.method in {
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
    } and server is not None:
        summary.update(
            {
                "total_uploaded_parameters": server.total_uploaded_parameters,
                "total_uploaded_mb": server.total_uploaded_mb,
                "full_participation_uploaded_parameters": server.full_participation_uploaded_parameters,
                "full_participation_uploaded_mb": server.full_participation_uploaded_mb,
                "communication_reduction_vs_full_participation_percent": (
                    server.communication_reduction_vs_full_participation_percent
                ),
            }
        )
    if args.method in {"wca_comfedseg_pbn", "wca_comfedseg_bn"} and server is not None:
        summary.update(
            {
                "local_bn_diagnostic_enabled": True,
                "bn_keys_kept_local_count": len(server.local_bn_keys),
                "non_bn_keys_aggregated_count": len(server.non_bn_keys),
                "bn_diagnostic_note": (
                    "Phase 5H clean WCA+BN diagnostic; distinct from the old failed "
                    "wca_comfedseg_pbn variant by method name, output naming, and interpretation."
                    if args.method == "wca_comfedseg_bn"
                    else "Historical WCA-PBN diagnostic variant."
                ),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    norm = "batch" if args.method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"} else "group"
    model_fn = lambda: build_unet2d(base_channels=args.base_channels, norm=norm)

    clients, split_metadata = build_clients(args, device, model_fn)
    if args.method == "local":
        rows, trained_model = run_local_training(
            clients=clients,
            model_fn=model_fn,
            device=device,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            split_name=args.split,
            eval_split=args.eval_split,
            lr=args.lr,
        )
    elif args.method == "centralized":
        rows, trained_model = run_centralized_training(
            clients=clients,
            model_fn=model_fn,
            device=device,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            split_name=args.split,
            eval_split=args.eval_split,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        server = FederatedServer(
            model_fn=model_fn,
            clients=clients,
            device=device,
            method=args.method,
            fedprox_mu=args.fedprox_mu,
            debug_fedbn=args.debug_fedbn,
            debug_pbn=args.debug_pbn,
            wca_alpha=args.wca_alpha,
            wca_beta=args.wca_beta,
            wca_max_weight=args.wca_max_weight,
            rg_min_gate=args.rg_min_gate,
            client_fraction=args.client_fraction,
            min_selected_clients=args.min_selected_clients,
            comm_scheduler=args.comm_scheduler,
        )
        rows = server.run(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            eval_split=args.eval_split,
            split_name=args.split,
            dataset_name=args.dataset,
        )
        trained_model = server.global_model

    rows = add_run_context(rows, args)
    results_dir_name = Path(args.results_dir).name
    is_phase_prefixed_run = results_dir_name.startswith("phase5")
    prediction_figure_dir = (
        f"results/figures/{results_dir_name}_predictions/{args.dataset}"
        if args.dataset != "synthetic" and is_phase_prefixed_run
        else f"results/figures/phase5a_predictions/{args.dataset}"
        if args.dataset != "synthetic"
        else "results/figures/phase1_predictions"
    )
    prediction_figures = save_phase1_prediction_figures(
        model=trained_model,
        clients=clients,
        output_dir=prediction_figure_dir,
        split=args.eval_split,
        examples_per_client=2,
    )
    if args.dataset != "synthetic":
        split_figure_dir = (
            f"results/figures/{results_dir_name}_real_splits/{args.dataset}"
            if is_phase_prefixed_run
            else f"results/figures/phase5a_real_splits/{args.dataset}"
        )
    else:
        split_figure_dir = (
            "results/figures/phase4c_hard_noniid"
            if args.split == "hard_noniid"
            else "results/figures/phase2_splits"
        )
    split_figure_path = save_phase2_split_distribution_figure(
        split_metadata=split_metadata,
        output_dir=split_figure_dir,
    )

    run_dir = make_run_dir(args.results_dir, args.method, f"{args.dataset}_{args.split}", args.seed)
    csv_path = run_dir / "metrics.csv"
    wca_weight_path = run_dir / f"wca_weights_{args.dataset}_{args.split}_seed{args.seed}.csv"
    wca_smooth_weight_path = run_dir / f"wca_smooth_weights_{args.dataset}_{args.split}_seed{args.seed}.csv"
    wca_pbn_weight_path = run_dir / f"wca_pbn_weights_{args.dataset}_{args.split}_seed{args.seed}.csv"
    wca_rg_weight_path = run_dir / f"wca_rg_weights_{args.dataset}_{args.split}_seed{args.seed}.csv"
    communication_path = run_dir / f"communication_{args.method}_{args.dataset}_{args.split}_seed{args.seed}.csv"
    summary_path = run_dir / "summary.json"
    args_path = run_dir / "args.json"
    split_config_path = run_dir / "split_config.json"
    client_summary_name = (
        f"{results_dir_name.split('_', 1)[0]}_{run_dir.name}_{args.split}_summary.json"
        if is_phase_prefixed_run
        else f"{run_dir.name}_{args.split}_summary.json"
    )
    client_summary_path = Path("results/summaries") / client_summary_name

    fieldnames = [
        "round",
        "client_id",
        "method",
        "split",
        "dice",
        "iou",
        "loss",
        "dataset",
        "clients",
        "local_epochs",
        "batch_size",
        "seed",
        "fedprox_mu",
    ]
    summary = build_client_level_summary(
        args,
        rows,
        str(csv_path),
        prediction_figure_dir,
        len(prediction_figures),
        split_metadata,
        str(split_figure_path),
        str(wca_weight_path) if args.method in {
            "wca_comfedseg",
            "wca_comfedseg_prox",
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
            "wca_comfedseg_bn",
        } else "",
        str(wca_smooth_weight_path) if args.method == "wca_comfedseg_smooth" else "",
        str(wca_pbn_weight_path) if args.method == "wca_comfedseg_pbn" else "",
        str(wca_rg_weight_path) if args.method == "wca_comfedseg_rg" else "",
        str(communication_path) if args.method in {
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
        } else "",
        server if args.method in {
            "wca_comfedseg_comm",
            "wca_comfedseg_comm_conservative",
            "wca_comfedseg_prox_comm_cons",
            "wca_comfedseg_pbn",
            "wca_comfedseg_bn",
            "wca_comfedseg_rg",
        } else None,
    )
    save_csv(csv_path, rows, fieldnames)
    if args.method in {"wca_comfedseg", "wca_comfedseg_prox"}:
        save_csv(
            wca_weight_path,
            server.wca_weight_rows,
            [
                "round",
                "client_id",
                "data_size",
                "data_weight",
                "validation_dice_used",
                "performance_deficit",
                "deficit_weight",
                "final_aggregation_weight",
                "wca_alpha",
            ],
        )
    if args.method in {
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
    }:
        save_csv(
            wca_weight_path,
            server.wca_weight_rows,
            [
                "round",
                "client_id",
                "selected",
                "data_size",
                "data_weight",
                "validation_dice_used",
                "performance_deficit",
                "deficit_weight",
                "final_aggregation_weight",
                "wca_alpha",
            ],
        )
        save_csv(
            communication_path,
            server.communication_rows,
            [
                "round",
                "client_id",
                "method",
                "dataset",
                "split",
                "selected",
                "selected_reason",
                "previous_validation_dice",
                "performance_deficit",
                "data_size",
                "uploaded_parameters",
                "uploaded_mb",
                "cumulative_uploaded_parameters",
                "cumulative_uploaded_mb",
                "client_fraction",
                "min_selected_clients",
                "wca_alpha",
            ],
        )
    if args.method == "wca_comfedseg_smooth":
        save_csv(
            wca_smooth_weight_path,
            server.wca_smooth_weight_rows,
            [
                "round",
                "client_id",
                "data_size",
                "data_weight",
                "validation_dice_used",
                "performance_deficit",
                "deficit_weight",
                "raw_wca_weight",
                "capped_weight",
                "previous_weight",
                "smoothed_weight",
                "final_aggregation_weight",
                "wca_alpha",
                "wca_beta",
                "wca_max_weight",
            ],
        )
    if args.method in {"wca_comfedseg_pbn", "wca_comfedseg_bn"}:
        save_csv(
            wca_weight_path if args.method == "wca_comfedseg_bn" else wca_pbn_weight_path,
            server.wca_pbn_weight_rows,
            [
                "round",
                "client_id",
                "data_size",
                "data_weight",
                "validation_dice_used",
                "performance_deficit",
                "deficit_weight",
                "final_aggregation_weight",
                "wca_alpha",
                "bn_parameters_kept_local",
                "non_bn_parameters_aggregated",
                "bn_diagnostic_variant",
            ],
        )
    if args.method == "wca_comfedseg_rg":
        save_csv(
            wca_rg_weight_path,
            server.wca_rg_weight_rows,
            [
                "round",
                "client_id",
                "data_size",
                "data_weight",
                "previous_validation_dice",
                "current_validation_dice",
                "validation_dice_improvement",
                "positive_improvement",
                "reliability_score",
                "reliability_gate",
                "performance_deficit",
                "reliable_deficit",
                "reliable_deficit_weight",
                "final_aggregation_weight",
                "wca_alpha",
                "rg_min_gate",
            ],
        )
    save_json(args_path, vars(args))
    save_json(split_config_path, split_metadata)
    save_json(summary_path, summary)
    save_json(client_summary_path, summary)

    if args.dataset != "synthetic":
        run_label = "Phase 5"
    elif args.method in {
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
    }:
        run_label = "Phase 4B"
    elif args.method == "wca_comfedseg_pbn":
        run_label = "Phase 4E"
    elif args.method == "wca_comfedseg_rg":
        run_label = "Phase 4F"
    elif args.method == "wca_comfedseg_smooth":
        run_label = "Phase 4D"
    elif args.method in {"wca_comfedseg", "wca_comfedseg_prox", "wca_comfedseg_bn"}:
        run_label = "Phase 4A"
    else:
        run_label = "Phase 3"
    print(f"{run_label} run completed on {device}.")
    print(f"CSV log saved to: {csv_path}")
    if args.method in {
        "wca_comfedseg",
        "wca_comfedseg_prox",
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
        "wca_comfedseg_bn",
    }:
        print(f"WCA weight log saved to: {wca_weight_path}")
    if args.method == "wca_comfedseg_smooth":
        print(f"WCA-Smooth weight log saved to: {wca_smooth_weight_path}")
    if args.method == "wca_comfedseg_pbn":
        print(f"WCA-PBN weight log saved to: {wca_pbn_weight_path}")
    if args.method == "wca_comfedseg_rg":
        print(f"WCA-RG weight log saved to: {wca_rg_weight_path}")
    if args.method in {
        "wca_comfedseg_comm",
        "wca_comfedseg_comm_conservative",
        "wca_comfedseg_prox_comm_cons",
    }:
        print(f"Communication log saved to: {communication_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Client-level summary saved to: {client_summary_path}")
    print(f"Split distribution figure saved to: {split_figure_path}")
    print(f"Prediction figures saved to: {prediction_figure_dir} ({len(prediction_figures)} files)")


if __name__ == "__main__":
    main()
