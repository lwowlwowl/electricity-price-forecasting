#!/usr/bin/env python
"""Apply a frozen robust ensemble calibrator to complete profit archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.robust_ensemble import (  # noqa: E402
    EconomicEnsembleCalibrator,
    EnsembleCalibratorConfig,
)


TASKS = ("da", "rt_at_da", "rt")


def _display_path(path: Path) -> str:
    """Keep metadata portable without requiring every checkpoint to live in ROOT."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def apply(
    directory: Path,
    checkpoint_path: Path,
    output_family: str,
    *,
    phase: str = "inner",
    batch_size: int = 1024,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("source_validation_improved", False)
        or checkpoint.get("deployment_approved", False)
    ):
        raise ValueError("calibrator is not approved by its frozen pre-test rule")
    members = tuple(checkpoint["members"])
    config = EnsembleCalibratorConfig(**checkpoint["calibrator_config"])
    calibrator = EconomicEnsembleCalibrator(config)
    calibrator.load_state_dict(checkpoint["model_state"], strict=True)
    calibrator.eval()
    records = {
        task: [_load(directory / f"{family}_{task}_{phase}_seed0.npz") for family in members]
        for task in TASKS
    }
    for task in TASKS:
        reference = records[task][0]
        for record in records[task][1:]:
            if not np.array_equal(reference["nodes"], record["nodes"]):
                raise ValueError(f"{task}: node mismatch")
            if not np.array_equal(reference["dates"], record["dates"]):
                raise ValueError(f"{task}: date mismatch")
            for key in ("target_da", "target_rt"):
                if key in reference and not np.allclose(reference[key], record[key], rtol=0, atol=5e-4):
                    raise ValueError(f"{task}: {key} mismatch")

    da_leading_shape = records["da"][0]["prediction"].shape[:2]
    rt_leading_shape = records["rt"][0]["prediction"].shape[:2]
    da_members = np.stack([item["prediction"] for item in records["da"]], axis=2)
    rt_da_members = np.stack([item["prediction"] for item in records["rt_at_da"]], axis=2)
    rt_members = np.stack([item["prediction"] for item in records["rt"]], axis=2)
    da_flat = da_members.reshape(-1, len(members), da_members.shape[-1])
    rt_da_flat = rt_da_members.reshape(-1, len(members), rt_da_members.shape[-1])
    rt_flat = rt_members.reshape(-1, len(members), rt_members.shape[-2], rt_members.shape[-1])
    da_outputs = [[], []]
    with torch.no_grad():
        for start in range(0, len(da_flat), batch_size):
            stop = min(len(da_flat), start + batch_size)
            values = calibrator.calibrate_da_pair(
                torch.from_numpy(da_flat[start:stop]).float(),
                torch.from_numpy(rt_da_flat[start:stop]).float(),
            )
            for index, value in enumerate(values):
                da_outputs[index].append(value.numpy())
        rt_outputs = []
        for start in range(0, len(rt_flat), batch_size):
            stop = min(len(rt_flat), start + batch_size)
            rt_outputs.append(calibrator.calibrate_rt(
                torch.from_numpy(rt_flat[start:stop]).float()
            ).numpy())
    predictions = {
        "da": np.concatenate(da_outputs[0]).reshape(*da_leading_shape, da_members.shape[-1]),
        "rt_at_da": np.concatenate(da_outputs[1]).reshape(*da_leading_shape, rt_da_members.shape[-1]),
        "rt": np.concatenate(rt_outputs).reshape(
            *rt_leading_shape, rt_members.shape[-2], rt_members.shape[-1]
        ),
    }
    for task in TASKS:
        reference = records[task][0]
        arrays = {key: value for key, value in reference.items() if key not in {"prediction", "metadata"}}
        arrays["prediction"] = predictions[task].astype(np.float32)
        arrays["metadata"] = json.dumps({
            "family": output_family,
            "task": task,
            "phase": phase,
            "protocol": "source_market_robust_decision_aware_ensemble",
            "members": list(members),
            "calibrator": _display_path(checkpoint_path),
            "ercot_training_labels_used": bool(
                checkpoint.get("ercot_training_labels_used", False)
            ),
            "training_stage": checkpoint.get("training_stage", "inner_train"),
            "2025_used": False,
        }, ensure_ascii=False, sort_keys=True)
        np.savez_compressed(directory / f"{output_family}_{task}_{phase}_seed0.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="data/results/global_implicit_qp_profit_predictions_v1")
    parser.add_argument("--checkpoint", default="data/checkpoints/global_robust_ensemble_v1/calibrator.pt")
    parser.add_argument("--output-family", default="implicitqp_robust_ensemble_v1")
    parser.add_argument("--phase", choices=("inner", "outer", "final"), default="inner")
    args = parser.parse_args()
    apply(ROOT / args.prediction_dir, ROOT / args.checkpoint, args.output_family, phase=args.phase)
    print(json.dumps({"event": "robust_ensemble_archives_ready", "family": args.output_family}))


if __name__ == "__main__":
    main()
