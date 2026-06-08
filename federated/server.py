"""Federated server orchestration for synthetic FL baselines and WCA variants."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from federated.client import FederatedClient
from federated.communication import (
    count_uploaded_parameters,
    estimate_uploaded_mb,
    select_clients_for_conservative_communication,
    select_clients_for_communication,
)
from federated.fedbn import (
    batchnorm_state_keys,
    merge_global_with_local_batchnorm,
    partition_fedbn_state_keys,
    validate_fedbn_key_partition,
)
from federated.fedavg import fedavg_aggregate
from methods.wca_comfedseg import compute_wca_aggregation_weights, wca_aggregate
from methods.wca_reliability_gated import (
    aggregate_model_states_with_rg_weights,
    compute_reliability_gated_wca_weights,
)
from methods.wca_smooth import compute_wca_smooth_aggregation_weights


class FederatedServer:
    """Runs a minimal all-client FedAvg training loop."""

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        clients: list[FederatedClient],
        device: torch.device,
        method: str = "fedavg",
        fedprox_mu: float = 0.0,
        debug_fedbn: bool = False,
        debug_pbn: bool = False,
        wca_alpha: float = 0.5,
        wca_beta: float = 0.5,
        wca_max_weight: float = 0.5,
        rg_min_gate: float = 0.2,
        client_fraction: float = 0.67,
        min_selected_clients: int = 2,
        comm_scheduler: str = "adaptive",
    ) -> None:
        self.model_fn = model_fn
        self.clients = clients
        self.device = device
        self.method = method
        self.fedprox_mu = fedprox_mu
        self.debug_fedbn = debug_fedbn
        self.debug_pbn = debug_pbn
        self.wca_alpha = wca_alpha
        self.wca_beta = wca_beta
        self.wca_max_weight = wca_max_weight
        self.rg_min_gate = rg_min_gate
        self.client_fraction = client_fraction
        self.min_selected_clients = min_selected_clients
        self.comm_scheduler = comm_scheduler
        self.global_model = model_fn().to(device)
        self.bn_keys = (
            batchnorm_state_keys(self.global_model)
            if method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"}
            else set()
        )
        self.non_bn_keys, self.local_bn_keys = (
            partition_fedbn_state_keys(self.global_model)
            if method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"}
            else (list(self.global_model.state_dict().keys()), [])
        )
        self.wca_weight_rows: list[dict[str, float | int | str]] = []
        self.wca_smooth_weight_rows: list[dict[str, float | int | str]] = []
        self.wca_pbn_weight_rows: list[dict[str, float | int | str | bool]] = []
        self.wca_rg_weight_rows: list[dict[str, float | int | str]] = []
        self.communication_rows: list[dict[str, float | int | str | bool]] = []
        self.total_uploaded_parameters = 0
        self.total_uploaded_mb = 0.0
        self.full_participation_uploaded_parameters = 0
        self.full_participation_uploaded_mb = 0.0
        self.communication_reduction_vs_full_participation_percent = 0.0
        if self.method == "fedbn" and self.debug_fedbn:
            summary = validate_fedbn_key_partition(self.global_model)
            print("FedBN key partition sanity check:")
            print(f"  has_batchnorm: {summary['has_batchnorm']}")
            print(f"  aggregated_key_count: {summary['aggregated_key_count']}")
            print(f"  local_batchnorm_key_count: {summary['local_batchnorm_key_count']}")
            print(f"  aggregated_key_examples: {summary['aggregated_key_examples']}")
            print(f"  local_batchnorm_key_examples: {summary['local_batchnorm_key_examples']}")
        if self.method == "wca_comfedseg_pbn" and self.debug_pbn:
            summary = validate_fedbn_key_partition(self.global_model)
            print("WCA-PBN BatchNorm partition sanity check:")
            print(f"  has_batchnorm: {summary['has_batchnorm']}")
            print(f"  non_bn_keys_aggregated_count: {summary['aggregated_key_count']}")
            print(f"  bn_keys_kept_local_count: {summary['local_batchnorm_key_count']}")
            print(f"  non_bn_key_examples: {summary['aggregated_key_examples']}")
            print(f"  bn_key_examples: {summary['local_batchnorm_key_examples']}")
            print("  confirmation: BN keys are kept local and excluded from aggregation.")

    def run(
        self,
        rounds: int,
        local_epochs: int,
        eval_split: str = "val",
        split_name: str = "iid",
        dataset_name: str = "synthetic",
    ) -> list[dict[str, float | int | str]]:
        if self.method not in {
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
        }:
            raise NotImplementedError("Unsupported federated method.")

        rows: list[dict[str, float | int | str]] = []
        previous_validation_dice: dict[int, float] | None = None
        previous_rg_validation_dice: dict[int, float] | None = None
        previous_smooth_weights: list[float] | None = None
        conservative_skipped_streaks = {client.client_id: 0 for client in self.clients}
        full_upload_parameters = count_uploaded_parameters(
            {key: value.detach().cpu() for key, value in self.global_model.state_dict().items()}
        )
        self.full_participation_uploaded_parameters = full_upload_parameters * len(self.clients) * rounds
        self.full_participation_uploaded_mb = estimate_uploaded_mb(self.full_participation_uploaded_parameters)

        for round_idx in range(1, rounds + 1):
            global_state = {key: value.detach().cpu() for key, value in self.global_model.state_dict().items()}
            client_states = []
            client_sample_counts = []
            current_rg_validation_dice: dict[int, float] = {}
            selected_clients: list[FederatedClient]

            if self.method == "wca_comfedseg_comm":
                selections = select_clients_for_communication(
                    clients=self.clients,
                    previous_validation_dice=previous_validation_dice,
                    round_idx=round_idx,
                    client_fraction=self.client_fraction,
                    min_selected_clients=self.min_selected_clients,
                    scheduler=self.comm_scheduler,
                )
                selection_by_client_id = {selection.client_id: selection for selection in selections}
                selected_clients = [
                    client
                    for client in self.clients
                    if selection_by_client_id[client.client_id].selected
                ]
            elif self.method in {"wca_comfedseg_comm_conservative", "wca_comfedseg_prox_comm_cons"}:
                selections = select_clients_for_conservative_communication(
                    clients=self.clients,
                    previous_validation_dice=previous_validation_dice,
                    round_idx=round_idx,
                    client_fraction=self.client_fraction,
                    min_selected_clients=self.min_selected_clients,
                    skipped_streaks=conservative_skipped_streaks,
                )
                selection_by_client_id = {selection.client_id: selection for selection in selections}
                selected_clients = [
                    client
                    for client in self.clients
                    if selection_by_client_id[client.client_id].selected
                ]
            else:
                selected_clients = self.clients

            round_uploaded_parameters = 0
            round_uploaded_mb = 0.0
            for client in selected_clients:
                initial_state = None
                proximal_mu = 0.0
                if self.method in {"fedprox", "wca_comfedseg_prox", "wca_comfedseg_prox_comm_cons"}:
                    proximal_mu = self.fedprox_mu
                if self.method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"}:
                    initial_state = merge_global_with_local_batchnorm(
                        global_state=global_state,
                        local_state=client.local_model_state,
                        bn_keys=self.bn_keys,
                    )

                state, num_samples, _ = client.train_from_global(
                    global_state,
                    local_epochs=local_epochs,
                    proximal_mu=proximal_mu,
                    initial_state=initial_state,
                )
                client_states.append(state)
                client_sample_counts.append(num_samples)
                if self.method == "wca_comfedseg_rg":
                    current_metrics = client.evaluate_state(state, split=eval_split)
                    current_rg_validation_dice[client.client_id] = float(current_metrics["dice"])
                if self.method in {
                    "wca_comfedseg_comm",
                    "wca_comfedseg_comm_conservative",
                    "wca_comfedseg_prox_comm_cons",
                }:
                    uploaded_parameters = count_uploaded_parameters(state)
                    round_uploaded_parameters += uploaded_parameters
                    round_uploaded_mb += estimate_uploaded_mb(uploaded_parameters)

            if self.method == "wca_comfedseg_rg":
                previous_dice_for_weights = None
                if previous_rg_validation_dice is not None:
                    previous_dice_for_weights = [
                        previous_rg_validation_dice.get(client.client_id)
                        for client in selected_clients
                    ]
                current_dice_for_weights = [
                    current_rg_validation_dice.get(client.client_id)
                    for client in selected_clients
                ]
                aggregation_weights, weight_rows = compute_reliability_gated_wca_weights(
                    client_num_samples=client_sample_counts,
                    previous_validation_dice=previous_dice_for_weights,
                    current_validation_dice=current_dice_for_weights,
                    alpha=self.wca_alpha,
                    min_gate=self.rg_min_gate,
                )
                for client, weight_row in zip(selected_clients, weight_rows):
                    logged_row = dict(weight_row)
                    logged_row["round"] = round_idx
                    logged_row["client_id"] = client.client_id
                    self.wca_rg_weight_rows.append(logged_row)
                aggregated_state = aggregate_model_states_with_rg_weights(client_states, aggregation_weights)
            elif self.method in {"wca_comfedseg_pbn", "wca_comfedseg_bn"}:
                dice_for_weights = None
                if previous_validation_dice is not None:
                    dice_for_weights = [
                        previous_validation_dice.get(client.client_id)
                        for client in selected_clients
                    ]
                aggregation_weights, weight_rows = compute_wca_aggregation_weights(
                    client_num_samples=client_sample_counts,
                    client_validation_dice=dice_for_weights,
                    alpha=self.wca_alpha,
                )
                for client, weight_row in zip(selected_clients, weight_rows):
                    logged_row = dict(weight_row)
                    logged_row["round"] = round_idx
                    logged_row["client_id"] = client.client_id
                    logged_row["bn_parameters_kept_local"] = len(self.local_bn_keys)
                    logged_row["non_bn_parameters_aggregated"] = len(self.non_bn_keys)
                    logged_row["bn_diagnostic_variant"] = self.method
                    self.wca_pbn_weight_rows.append(logged_row)
                aggregated_state = wca_aggregate(client_states, aggregation_weights, exclude_keys=self.bn_keys)
            elif self.method == "wca_comfedseg_smooth":
                dice_for_weights = None
                if previous_validation_dice is not None:
                    dice_for_weights = [
                        previous_validation_dice.get(client.client_id)
                        for client in selected_clients
                    ]
                aggregation_weights, weight_rows = compute_wca_smooth_aggregation_weights(
                    client_num_samples=client_sample_counts,
                    client_validation_dice=dice_for_weights,
                    previous_weights=previous_smooth_weights,
                    alpha=self.wca_alpha,
                    beta=self.wca_beta,
                    max_weight=self.wca_max_weight,
                )
                previous_smooth_weights = list(aggregation_weights)
                for client, weight_row in zip(selected_clients, weight_rows):
                    logged_row = dict(weight_row)
                    logged_row["round"] = round_idx
                    logged_row["client_id"] = client.client_id
                    self.wca_smooth_weight_rows.append(logged_row)
                aggregated_state = wca_aggregate(client_states, aggregation_weights)
            elif self.method in {
                "wca_comfedseg",
                "wca_comfedseg_prox",
                "wca_comfedseg_comm",
                "wca_comfedseg_comm_conservative",
                "wca_comfedseg_prox_comm_cons",
            }:
                dice_for_weights = None
                if previous_validation_dice is not None:
                    dice_for_weights = [
                        previous_validation_dice.get(client.client_id)
                        for client in selected_clients
                    ]
                aggregation_weights, weight_rows = compute_wca_aggregation_weights(
                    client_num_samples=client_sample_counts,
                    client_validation_dice=dice_for_weights,
                    alpha=self.wca_alpha,
                )
                if self.method in {
                    "wca_comfedseg_comm",
                    "wca_comfedseg_comm_conservative",
                    "wca_comfedseg_prox_comm_cons",
                }:
                    weight_by_client_id = {}
                    for client, weight_row in zip(selected_clients, weight_rows):
                        logged_row = dict(weight_row)
                        logged_row["round"] = round_idx
                        logged_row["client_id"] = client.client_id
                        logged_row["selected"] = True
                        weight_by_client_id[client.client_id] = logged_row
                    for client in self.clients:
                        if client.client_id in weight_by_client_id:
                            self.wca_weight_rows.append(weight_by_client_id[client.client_id])
                        else:
                            selection = selection_by_client_id[client.client_id]
                            self.wca_weight_rows.append(
                                {
                                    "round": round_idx,
                                    "client_id": client.client_id,
                                    "selected": False,
                                    "data_size": client.num_train_samples,
                                    "data_weight": 0.0,
                                    "validation_dice_used": (
                                        ""
                                        if selection.previous_validation_dice is None
                                        else selection.previous_validation_dice
                                    ),
                                    "performance_deficit": selection.performance_deficit,
                                    "deficit_weight": 0.0,
                                    "final_aggregation_weight": 0.0,
                                    "wca_alpha": self.wca_alpha,
                                }
                            )
                else:
                    for client, weight_row in zip(selected_clients, weight_rows):
                        logged_row = dict(weight_row)
                        logged_row["round"] = round_idx
                        logged_row["client_id"] = client.client_id
                        self.wca_weight_rows.append(logged_row)
                aggregated_state = wca_aggregate(client_states, aggregation_weights)
            else:
                aggregated_state = fedavg_aggregate(
                    client_states,
                    client_sample_counts,
                    exclude_keys=self.bn_keys if self.method == "fedbn" else None,
                )
            if self.method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"}:
                # BatchNorm keys are deliberately excluded from aggregation.
                # The server keeps placeholder global BN values only so the
                # state_dict remains complete; client-specific BN state is
                # restored by merge_global_with_local_batchnorm below.
                for key in self.bn_keys:
                    aggregated_state[key] = global_state[key]
            self.global_model.load_state_dict(aggregated_state)
            updated_global_state = {
                key: value.detach().cpu()
                for key, value in self.global_model.state_dict().items()
            }

            if self.method in {
                "wca_comfedseg_comm",
                "wca_comfedseg_comm_conservative",
                "wca_comfedseg_prox_comm_cons",
            }:
                self.total_uploaded_parameters += round_uploaded_parameters
                self.total_uploaded_mb += round_uploaded_mb
                for client in self.clients:
                    selection = selection_by_client_id[client.client_id]
                    uploaded_parameters = full_upload_parameters if selection.selected else 0
                    uploaded_mb = estimate_uploaded_mb(uploaded_parameters)
                    self.communication_rows.append(
                        {
                            "round": round_idx,
                            "client_id": client.client_id,
                            "method": self.method,
                            "dataset": dataset_name,
                            "split": split_name,
                            "selected": selection.selected,
                            "selected_reason": selection.selected_reason,
                            "previous_validation_dice": (
                                ""
                                if selection.previous_validation_dice is None
                                else selection.previous_validation_dice
                            ),
                            "performance_deficit": selection.performance_deficit,
                            "data_size": client.num_train_samples,
                            "uploaded_parameters": uploaded_parameters,
                            "uploaded_mb": uploaded_mb,
                            "cumulative_uploaded_parameters": self.total_uploaded_parameters,
                            "cumulative_uploaded_mb": self.total_uploaded_mb,
                            "client_fraction": self.client_fraction,
                            "min_selected_clients": self.min_selected_clients,
                            "wca_alpha": self.wca_alpha,
                        }
                    )
                    if self.method in {"wca_comfedseg_comm_conservative", "wca_comfedseg_prox_comm_cons"}:
                        if selection.selected:
                            conservative_skipped_streaks[client.client_id] = 0
                        else:
                            conservative_skipped_streaks[client.client_id] += 1

            for client in self.clients:
                if self.method in {"fedbn", "wca_comfedseg_pbn", "wca_comfedseg_bn"}:
                    eval_state = merge_global_with_local_batchnorm(
                        global_state=updated_global_state,
                        local_state=client.local_model_state,
                        bn_keys=self.bn_keys,
                    )
                    metrics = client.evaluate_state(eval_state, split=eval_split)
                else:
                    metrics = client.evaluate(self.global_model, split=eval_split)
                rows.append(
                    {
                        "round": round_idx,
                        "client_id": client.client_id,
                        "method": self.method,
                        "split": split_name,
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "loss": metrics["loss"],
                    }
                )
                if self.method in {
                    "wca_comfedseg",
                    "wca_comfedseg_prox",
                    "wca_comfedseg_comm",
                    "wca_comfedseg_comm_conservative",
                    "wca_comfedseg_prox_comm_cons",
                    "wca_comfedseg_bn",
                    "wca_comfedseg_smooth",
                    "wca_comfedseg_pbn",
                }:
                    if previous_validation_dice is None:
                        previous_validation_dice = {}
                    previous_validation_dice[client.client_id] = float(metrics["dice"])
            if self.method == "wca_comfedseg_rg":
                previous_rg_validation_dice = dict(current_rg_validation_dice)

        if self.full_participation_uploaded_parameters > 0:
            saved = self.full_participation_uploaded_parameters - self.total_uploaded_parameters
            self.communication_reduction_vs_full_participation_percent = (
                100.0 * float(saved) / float(self.full_participation_uploaded_parameters)
            )
        return rows
