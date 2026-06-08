"""Sanity checks for the Phase 4B communication scheduler."""

from dataclasses import dataclass

from federated.communication import count_uploaded_parameters, select_clients_for_communication


@dataclass
class DummyClient:
    client_id: int
    num_train_samples: int


def test_round_one_selects_all_clients() -> None:
    clients = [DummyClient(0, 10), DummyClient(1, 20), DummyClient(2, 30)]

    selections = select_clients_for_communication(
        clients=clients,
        previous_validation_dice=None,
        round_idx=1,
        client_fraction=0.67,
        min_selected_clients=2,
    )

    assert [selection.selected for selection in selections] == [True, True, True]
    assert {selection.selected_reason for selection in selections} == {"all_clients_round1"}


def test_later_round_always_selects_worst_client() -> None:
    clients = [DummyClient(0, 10), DummyClient(1, 20), DummyClient(2, 30)]

    selections = select_clients_for_communication(
        clients=clients,
        previous_validation_dice={0: 0.9, 1: 0.2, 2: 0.8},
        round_idx=2,
        client_fraction=0.67,
        min_selected_clients=2,
    )

    selected = {selection.client_id for selection in selections if selection.selected}
    assert len(selected) == 2
    assert 1 in selected
    assert next(selection for selection in selections if selection.client_id == 1).selected_reason == "worst_client"


def test_count_uploaded_parameters_uses_all_state_tensors() -> None:
    import torch

    state = {"a": torch.zeros(2, 3), "b": torch.zeros(4)}

    assert count_uploaded_parameters(state) == 10
