"""Utilities for reading training budget registry settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_BUDGET_PATH = Path("configs/training_budget_registry.yaml")
SMOKE_TEST_FINAL_WARNING = (
    "The 5-round / 1-local-epoch setting is intended for smoke tests only, "
    "not for final evaluation."
)


def load_training_budget_registry(path: str | Path = DEFAULT_BUDGET_PATH) -> dict[str, dict[str, Any]]:
    """Load the training budget registry YAML."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Training budget registry must be a mapping: {registry_path}")
    return data


def get_synthetic_final_budget(path: str | Path = DEFAULT_BUDGET_PATH) -> dict[str, Any]:
    """Return the synthetic confirmation budget."""

    registry = load_training_budget_registry(path)
    return dict(registry["synthetic_final_confirmation"])


def get_smoke_test_budget(path: str | Path = DEFAULT_BUDGET_PATH) -> dict[str, Any]:
    """Return the synthetic smoke-test budget."""

    registry = load_training_budget_registry(path)
    return dict(registry["synthetic_smoke_test"])


def get_tiny_budget_final_warning(rounds: int, local_epochs: int, final_setting: bool = True) -> str:
    """Return warning text when tiny smoke-test settings are used as final settings."""

    if final_setting and int(rounds) == 5 and int(local_epochs) == 1:
        return SMOKE_TEST_FINAL_WARNING
    return ""
