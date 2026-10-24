"""Load the single frozen baseline registry used by future experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "configs/decision_aware/baseline_registry_v1.json"


def load_baseline_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("status") != "frozen":
        raise ValueError("baseline registry must be frozen")
    if not isinstance(registry.get("stages"), dict):
        raise ValueError("baseline registry is missing stages")
    return registry


def get_baseline_stage(name: str) -> dict[str, Any]:
    registry = load_baseline_registry()
    try:
        return registry["stages"][name]
    except KeyError as exc:
        available = ", ".join(sorted(registry["stages"]))
        raise KeyError(f"unknown baseline stage {name!r}; available: {available}") from exc


def verify_registered_sources() -> list[str]:
    """Return validation errors without rewriting or recomputing any baseline."""
    registry = load_baseline_registry()
    errors: list[str] = []
    checked: set[tuple[str, str]] = set()

    def visit(stage_name: str, value: Any) -> None:
        if isinstance(value, dict):
            if "source" in value:
                check_source(stage_name, value)
            for child in value.values():
                visit(stage_name, child)
        elif isinstance(value, list):
            for child in value:
                visit(stage_name, child)

    def check_source(stage_name: str, value: dict[str, Any]) -> None:
        source = str(value["source"])
        expected = str(value.get("source_sha256", "")).lower()
        key = (source, expected)
        if key in checked:
            return
        checked.add(key)
        source_path = REPO_ROOT / source
        if not source_path.is_file():
            errors.append(f"{stage_name}: missing source {source}")
            return
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(
                f"{stage_name}: SHA256 mismatch for {source}: {actual} != {expected}"
            )

    for stage_name, stage in registry["stages"].items():
        visit(stage_name, stage)
    return errors
