from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.decision_aware.compare_joint_system_baselines import (
    _load_bundle_source_checkpoints,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, task: str, seed: int) -> None:
    torch.save(
        {
            "model_state": {"marker": torch.tensor([seed])},
            "config": {"seed": seed, "task": task},
            "norm_stats": {},
            "fusion_mode": f"{task}_{seed}",
        },
        path,
    )


def test_joint_bundle_uses_its_own_recorded_source_checkpoints(tmp_path: Path):
    paths = {}
    declared = {}
    for seed, task in enumerate(("da", "rt_at_da", "rt"), start=1):
        path = tmp_path / f"{task}.pt"
        _write_source(path, task, seed)
        paths[task] = path
        declared[task] = {"path": str(path), "sha256": _sha256(path)}

    bundle = {
        "source_checkpoints": declared,
        # 这些故意是错路径；应优先使用bundle明确记录的source。
        "experiment_config": {
            "joint_da_checkpoint": "wrong-da.pt",
            "joint_rt_at_da_checkpoint": "wrong-rt-da.pt",
            "joint_rt_checkpoint": "wrong-rt.pt",
        },
    }
    resolved, checkpoints, audit = _load_bundle_source_checkpoints(
        bundle, tmp_path / "joint.pt"
    )

    assert resolved == {task: path.resolve() for task, path in paths.items()}
    assert {task: value["config"]["seed"] for task, value in checkpoints.items()} == {
        "da": 1,
        "rt_at_da": 2,
        "rt": 3,
    }
    assert all(item["declared_by_bundle"] for item in audit.values())


def test_joint_bundle_rejects_source_hash_mismatch(tmp_path: Path):
    paths = {}
    for seed, task in enumerate(("da", "rt_at_da", "rt"), start=1):
        path = tmp_path / f"{task}.pt"
        _write_source(path, task, seed)
        paths[task] = path
    bundle = {
        "source_checkpoints": {
            task: {
                "path": str(path),
                "sha256": "0" * 64 if task == "da" else _sha256(path),
            }
            for task, path in paths.items()
        }
    }

    with pytest.raises(ValueError, match="SHA256不匹配"):
        _load_bundle_source_checkpoints(bundle, tmp_path / "joint.pt")
