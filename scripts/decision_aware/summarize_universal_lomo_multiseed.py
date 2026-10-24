#!/usr/bin/env python
"""汇总当前无ID LOMO的seed 0/1/2预测稳定性，不按盈利选择seed。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("da", "rt_at_da", "rt")


def _seed_path(result_root: Path, holdout: str, task: str, seed: int) -> Path:
    if seed == 0:
        return (
            ROOT / "data" / "results" / f"universal_lomo_{task}_v1"
            / f"{holdout}_{task}_v1.npz"
        )
    return result_root / f"{holdout}_{task}_inner_seed{seed}.npz"


def _macro_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape:
        raise ValueError("预测与真值形状不一致")
    node_axes = tuple(range(1, target.ndim))
    return float(np.mean(np.mean(np.abs(prediction - target), axis=node_axes)))


def _report_path(path: Path) -> str:
    """Keep repository artifacts portable without rejecting test/temp paths."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def summarize(
    result_root: Path,
    holdouts: tuple[str, ...],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    ensemble_output_root: Path | None = None,
) -> dict:
    report: dict = {
        "protocol": "universal_lomo_fixed_multiseed",
        "seed_selection_by_profit": False,
        "seeds": list(seeds),
        "results": {},
    }
    for holdout in holdouts:
        for task in tasks:
            predictions: list[np.ndarray] = []
            target_reference = None
            node_reference = None
            date_reference = None
            target_da_reference = None
            seed_mae: dict[str, float] = {}
            paths: dict[str, str] = {}
            for seed in seeds:
                path = _seed_path(result_root, holdout, task, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as archive:
                    prediction = archive["prediction"].astype(np.float32)
                    target = archive["target"].astype(np.float32)
                    nodes = archive["nodes"]
                    dates = archive["dates"]
                    target_da = (
                        archive["target_da"].astype(np.float32)
                        if "target_da" in archive.files else None
                    )
                if target_reference is None:
                    target_reference = target
                    node_reference = nodes
                    date_reference = dates
                    target_da_reference = target_da
                else:
                    if not np.array_equal(target_reference, target):
                        raise ValueError(f"{holdout}/{task}不同seed真值不一致")
                    if not np.array_equal(node_reference, nodes):
                        raise ValueError(f"{holdout}/{task}不同seed节点不一致")
                    if not np.array_equal(date_reference, dates):
                        raise ValueError(f"{holdout}/{task}不同seed日期不一致")
                    if (target_da_reference is None) != (target_da is None):
                        raise ValueError(f"{holdout}/{task}不同seed的DA结算真值字段不一致")
                    if target_da is not None and not np.array_equal(
                        target_da_reference, target_da
                    ):
                        raise ValueError(f"{holdout}/{task}不同seed的DA结算真值不一致")
                predictions.append(prediction)
                seed_mae[str(seed)] = _macro_mae(target, prediction)
                paths[str(seed)] = _report_path(path)
            values = np.asarray(list(seed_mae.values()), dtype=np.float64)
            ensemble = np.mean(np.stack(predictions), axis=0)
            result = {
                "seed_node_macro_mae": seed_mae,
                "mean_mae": float(np.mean(values)),
                "sample_std_mae": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "worst_seed_mae": float(np.max(values)),
                "best_seed_mae": float(np.min(values)),
                "equal_weight_ensemble_mae": _macro_mae(target_reference, ensemble),
                "paths": paths,
            }
            if ensemble_output_root is not None:
                ensemble_output_root.mkdir(parents=True, exist_ok=True)
                ensemble_path = (
                    ensemble_output_root
                    / f"{holdout}_{task}_inner_ensemble.npz"
                )
                arrays = {
                    "nodes": node_reference,
                    "dates": date_reference,
                    "prediction": ensemble.astype(np.float32),
                    "target": target_reference,
                    "metadata": json.dumps({
                        "protocol": "fixed_equal_weight_seed_ensemble",
                        "holdout_market": holdout.upper(),
                        "task": task,
                        "seeds": list(seeds),
                        "seed_selection_by_profit": False,
                    }, ensure_ascii=False, sort_keys=True),
                }
                if target_da_reference is not None:
                    arrays["target_da"] = target_da_reference
                np.savez_compressed(ensemble_path, **arrays)
                result["equal_weight_ensemble_path"] = _report_path(ensemble_path)
            report["results"][f"{holdout}_{task}"] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", default="data/results/universal_lomo_multiseed_v1"
    )
    parser.add_argument("--holdouts", default="ercot")
    parser.add_argument("--tasks", default="da,rt_at_da,rt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--output", default="data/results/universal_lomo_multiseed_v1/summary.json"
    )
    parser.add_argument("--ensemble-output-root", default="")
    args = parser.parse_args()
    holdouts = tuple(item.strip().lower() for item in args.holdouts.split(",") if item.strip())
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    if not holdouts or not tasks or not seeds:
        raise ValueError("holdouts/tasks/seeds不能为空")
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_tasks:
        raise ValueError(f"未知任务: {unknown_tasks}")
    report = summarize(
        ROOT / args.result_root,
        holdouts,
        tasks,
        seeds,
        ensemble_output_root=(
            ROOT / args.ensemble_output_root
            if args.ensemble_output_root else None
        ),
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
