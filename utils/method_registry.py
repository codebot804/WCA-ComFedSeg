"""Utilities for reading the project method registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/method_registry.yaml")


def load_method_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    """Load the method registry YAML."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Method registry must be a mapping: {registry_path}")
    return data


def get_methods_for_real_dataset_main(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Return methods approved for default real-dataset main runs."""

    registry = load_method_registry(path)
    return [
        method_name
        for method_name, metadata in registry.items()
        if metadata.get("run_on_real_dataset_default") is True
        and metadata.get("status") == "active"
    ]


def get_methods_for_ablation(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Return methods marked for ablation or appendix-only use."""

    registry = load_method_registry(path)
    return [
        method_name
        for method_name, metadata in registry.items()
        if metadata.get("include_in_ablation") is True
    ]


def is_method_allowed_for_real_main(
    method_name: str,
    path: str | Path = DEFAULT_REGISTRY_PATH,
    allow_failed_override: bool = False,
) -> bool:
    """Return whether a method may be used in future real-dataset main scripts.

    Failed variants are blocked unless an explicit override is passed by a
    future script. This utility does not enforce behavior globally.
    """

    registry = load_method_registry(path)
    if method_name not in registry:
        return False
    metadata = registry[method_name]
    if metadata.get("run_on_real_dataset_default") is True and metadata.get("status") == "active":
        return True
    if allow_failed_override and metadata.get("status") == "inactive":
        return True
    return False
