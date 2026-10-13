#!/usr/bin/env python
"""只用inner-train内部的2022留出集比较XGBoost稳健目标。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.baselines_global import (  # noqa: E402
    fit_xgboost_multioutput,
    node_datasets,
    prediction_metrics,
    xgboost_arrays,
)
from decision_aware.config import PilotConfig  # noqa: E402
from decision_aware.contracts import AnnualStage, FORMAL_EXPERIMENT_CONTRACT  # noqa: E402
from decision_aware.global_data import (  # noqa: E402
    build_global_task_datasets,
    load_formal_entity_mappings,
)


OBJECTIVES = {
    "squared": "reg:squarederror",
    "absolute": "reg:absoluteerror",
    "pseudohuber": "reg:pseudohubererror",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_aware/global_ercot_huber_v1.yaml")
    parser.add_argument("--task", default="rt_at_da", choices=("da", "rt_at_da", "rt"))
    parser.add_argument("--objectives", default="squared,absolute,pseudohuber")
    parser.add_argument("--split-date", default="2022-01-01")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument(
        "--output",
        default="data/results/global_baselines_v1/xgboost_robustness_train_internal.json",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers必须至少为1")
    names = [value.strip().lower() for value in args.objectives.split(",") if value.strip()]
    unknown = sorted(set(names) - set(OBJECTIVES))
    if not names or unknown:
        raise ValueError(f"未知或空objective: {unknown}")

    cfg = PilotConfig.from_yaml(args.config)
    FORMAL_EXPERIMENT_CONTRACT.validate_pilot_config(cfg)
    mappings = load_formal_entity_mappings(ROOT / "configs" / "nodes.yaml").subset(["ERCOT"])
    entities = [("ERCOT", node) for node in mappings.node_to_id]
    train_dataset, _ = build_global_task_datasets(
        cfg,
        args.task,
        entities,
        mappings,
        train_stage=AnnualStage.INNER_TRAIN,
        validation_stage=AnnualStage.INNER_VALIDATION,
    )
    per_node = {node: dataset for _, node, dataset in node_datasets(train_dataset)}

    def evaluate(node: str, name: str) -> tuple[str, str, dict]:
        dataset = per_node[node]
        features, targets = xgboost_arrays(dataset, args.task)
        if args.task == "rt":
            dates = np.asarray([
                str(delivery_date)
                for delivery_date, hourly_indices in dataset.days
                for _ in hourly_indices
            ])
        else:
            dates = np.asarray([str(window.delivery_date) for window in dataset.windows])
        fit_mask = dates < args.split_date
        holdout_mask = dates >= args.split_date
        if not fit_mask.any() or not holdout_mask.any():
            raise ValueError(f"{node}的内部训练/留出集为空")
        model = fit_xgboost_multioutput(
            features[fit_mask],
            targets[fit_mask],
            seed=args.seed,
            n_estimators=args.n_estimators,
            objective=OBJECTIVES[name],
        )
        prediction = model.predict(features[holdout_mask]).astype(np.float32)
        metrics = prediction_metrics(targets[holdout_mask], prediction)
        return node, name, {
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "fit_samples": int(fit_mask.sum()),
            "holdout_samples": int(holdout_mask.sum()),
        }

    started = perf_counter()
    reports = {name: {} for name in names}
    jobs = [(node, name) for _, node in entities for name in names]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate, node, name): (node, name)
            for node, name in jobs
        }
        for count, future in enumerate(as_completed(futures), start=1):
            node, name, metrics = future.result()
            reports[name][node] = metrics
            print(json.dumps({
                "event": "internal_holdout_complete",
                "completed": count,
                "total": len(jobs),
                "node": node,
                "objective": name,
                "mae": metrics["mae"],
            }, ensure_ascii=False), flush=True)

    summary = {}
    for name in names:
        ordered = {node: reports[name][node] for _, node in entities}
        summary[name] = {
            "objective": OBJECTIVES[name],
            "macro_mae": float(np.mean([item["mae"] for item in ordered.values()])),
            "macro_rmse": float(np.mean([item["rmse"] for item in ordered.values()])),
            "nodes": ordered,
        }
    ranking = sorted(
        ({"rank": 0, "name": name, "macro_mae": item["macro_mae"]}
         for name, item in summary.items()),
        key=lambda item: item["macro_mae"],
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    report = {
        "experiment": "xgboost_objective_selection_train_internal",
        "selection_data": "2020-2021 fit; 2022 holdout; 2023 never used for selection",
        "task": args.task,
        "split_date": args.split_date,
        "contract_hash": FORMAL_EXPERIMENT_CONTRACT.contract_hash,
        "n_estimators": args.n_estimators,
        "summary": summary,
        "ranking": ranking,
        "runtime_seconds": perf_counter() - started,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "diagnosis_saved",
        "path": output.relative_to(ROOT).as_posix(),
        "ranking": ranking,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
