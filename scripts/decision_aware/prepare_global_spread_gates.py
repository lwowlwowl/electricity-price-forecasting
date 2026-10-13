#!/usr/bin/env python
"""生成原始lag168与稳定Global之间的无真值门控候选。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _write(source: dict, prediction: np.ndarray, path: Path, family: str, task: str) -> None:
    arrays = dict(source)
    arrays["prediction"] = prediction.astype(np.float32)
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata.update({
        "family": family,
        "task": task,
        "construction": "truth-free seasonal/stable spread gate",
    })
    arrays["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _seasonal_prediction(root: Path, template: dict, task: str, phase: str) -> np.ndarray:
    predictions = []
    target_key = "target_da" if task == "da" else "target_rt"
    for index, node in enumerate(template["nodes"]):
        path = (
            root / "data" / "results" / "global_baselines_v1" / f"{task}_{phase}"
            / f"seasonal_{node}_seed0.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            prediction = archive["prediction"].astype(np.float32)
            target = archive["target"].astype(np.float32)
        expected = template[target_key][index]
        if target.shape != expected.shape or not np.allclose(
            target, expected, rtol=0.0, atol=1.0e-5
        ):
            raise ValueError(f"seasonal真值不一致: {task}/{node}")
        predictions.append(prediction)
    return np.stack(predictions)


def build_gate_predictions(
    seasonal_da: np.ndarray,
    seasonal_rt: np.ndarray,
    stable_da: np.ndarray,
    stable_rt: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    seasonal_spread = seasonal_da - seasonal_rt
    stable_spread = stable_da - stable_rt
    same_direction = np.signbit(seasonal_spread) == np.signbit(stable_spread)
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for weight in (0.50, 0.75, 0.90):
        family = f"gate_blend{int(weight * 100)}"
        candidates[family] = (
            weight * seasonal_da + (1.0 - weight) * stable_da,
            weight * seasonal_rt + (1.0 - weight) * stable_rt,
        )

    candidates["gate_agree"] = (
        np.where(same_direction, seasonal_da, stable_da),
        np.where(same_direction, seasonal_rt, stable_rt),
    )
    candidates["gate_agreeblend"] = (
        np.where(same_direction, seasonal_da, 0.5 * (seasonal_da + stable_da)),
        np.where(same_direction, seasonal_rt, 0.5 * (seasonal_rt + stable_rt)),
    )

    for threshold in (50.0, 100.0, 200.0):
        use_seasonal = np.abs(seasonal_spread) <= threshold
        family = f"gate_extreme{int(threshold)}"
        candidates[family] = (
            np.where(use_seasonal, seasonal_da, stable_da),
            np.where(use_seasonal, seasonal_rt, stable_rt),
        )
        midpoint = 0.5 * (seasonal_da + seasonal_rt)
        clipped = np.clip(seasonal_spread, -threshold, threshold)
        candidates[f"gate_clip{int(threshold)}"] = (
            midpoint + 0.5 * clipped,
            midpoint - 0.5 * clipped,
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir", default="data/results/global_lag168_predictions_v1"
    )
    parser.add_argument(
        "--output", default="data/results/global_spread_gate_manifest_v1.json"
    )
    parser.add_argument("--phase", default="inner", choices=("inner",))
    args = parser.parse_args()
    directory = ROOT / args.prediction_dir
    stable = {
        task: _load(directory / f"stable_{task}_{args.phase}_seed0.npz")
        for task in ("da", "rt_at_da", "rt")
    }
    seasonal_da = _seasonal_prediction(ROOT, stable["da"], "da", args.phase)
    seasonal_rt = _seasonal_prediction(
        ROOT, stable["rt_at_da"], "rt_at_da", args.phase
    )
    candidates = build_gate_predictions(
        seasonal_da,
        seasonal_rt,
        stable["da"]["prediction"],
        stable["rt_at_da"]["prediction"],
    )
    families = {}
    for family, (prediction_da, prediction_rt) in candidates.items():
        _write(
            stable["da"], prediction_da,
            directory / f"{family}_da_{args.phase}_seed0.npz", family, "da",
        )
        _write(
            stable["rt_at_da"], prediction_rt,
            directory / f"{family}_rt_at_da_{args.phase}_seed0.npz",
            family, "rt_at_da",
        )
        _write(
            stable["rt"], stable["rt"]["prediction"],
            directory / f"{family}_rt_{args.phase}_seed0.npz", family, "rt",
        )
        families[family] = {
            "da": "seasonal/stable gate archive",
            "rt_at_da": "seasonal/stable gate archive",
            "rt": "frozen stable Global RT archive",
        }
    report = {
        "experiment": "global_spread_gate_candidates_v1",
        "phase": args.phase,
        "truth_used_for_gate": False,
        "families": families,
        "prediction_dir": args.prediction_dir,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "spread_gate_archives_saved",
        "families": sorted(families),
        "output": output.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
