#!/usr/bin/env python
"""Build DA/RT-at-DA spread-calibration candidates from frozen predictions.

The DA forecast remains unchanged.  Only the RT-at-DA forecast is adjusted so
that the predicted spread satisfies

    DA - RT_at_DA_calibrated = alpha * (DA - RT_at_DA_original).

This is a post-hoc calibration experiment: it neither retrains a forecasting
model nor reads realized prices while producing a decision-time forecast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _save(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _family(alpha: float) -> str:
    return f"spread{int(round(alpha * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-family", default="universal")
    parser.add_argument("--phase", choices=("inner", "outer", "final"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphas", default="0.75,1.00,1.25,1.50,1.75,2.00")
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    suffix = f"{args.phase}_seed{args.seed}.npz"
    source = {
        task: _load(input_dir / f"{args.source_family}_{task}_{suffix}")
        for task in ("da", "rt_at_da", "rt")
    }
    da_prediction = source["da"]["prediction"].astype(np.float64)
    rt_at_da_prediction = source["rt_at_da"]["prediction"].astype(np.float64)
    if da_prediction.shape != rt_at_da_prediction.shape:
        raise ValueError("DA与RT-at-DA预测形状不一致")

    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas or any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("alphas必须为非空正数列表")
    if len({_family(alpha) for alpha in alphas}) != len(alphas):
        raise ValueError("alphas在两位百分比命名下发生冲突")

    manifest = {
        "experiment": "frozen_da_rt_at_da_spread_calibration_v1",
        "source_family": args.source_family,
        "phase": args.phase,
        "seed": args.seed,
        "formula": "rt_at_da_calibrated = da - alpha * (da - rt_at_da)",
        "selection_rule": "select_on_2023_only_then_confirm_unchanged_on_2024",
        "candidates": [],
    }
    for alpha in alphas:
        family = _family(alpha)
        calibrated = da_prediction - alpha * (da_prediction - rt_at_da_prediction)
        for task in ("da", "rt_at_da", "rt"):
            values = dict(source[task])
            if task == "rt_at_da":
                values["prediction"] = calibrated.astype(source[task]["prediction"].dtype)
            metadata = {
                "family": family,
                "source_family": args.source_family,
                "task": task,
                "phase": args.phase,
                "seed": args.seed,
                "spread_alpha": alpha,
                "realized_target_used_for_calibration": False,
            }
            values["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
            _save(output_dir / f"{family}_{task}_{suffix}", values)
        manifest["candidates"].append({"family": family, "alpha": alpha})

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"spread_calibration_{args.phase}_seed{args.seed}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "spread_calibration_candidates_saved",
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "families": [item["family"] for item in manifest["candidates"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
