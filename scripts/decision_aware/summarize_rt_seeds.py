#!/usr/bin/env python
"""汇总独立RT多随机种子，并且只按验证收益选后续checkpoint。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reports", nargs="*", default=[
            "data/results/rt_transformer_seed0.json",
            "data/results/rt_transformer_seed1.json",
            "data/results/rt_transformer_seed2.json",
        ]
    )
    parser.add_argument("--output", default="data/results/rt_transformer_3seed_summary.json")
    args = parser.parse_args()

    runs = []
    for report_path in args.reports:
        path = Path(report_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["run_type"] != "training":
            raise ValueError(f"不是正式训练报告: {path}")
        if report["selection_metric"] != "validation RT-only rolling mean daily revenue":
            raise ValueError(f"选模指标不一致: {path}")
        checkpoint = Path(report["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint不存在: {checkpoint}")
        runs.append({
            "report": str(path),
            "seed": int(report["history"][0].get("seed", report["config"]["seed"]))
            if "config" in report else int(Path(report_path).stem.rsplit("seed", 1)[1]),
            "checkpoint": str(checkpoint),
            "epochs_ran": len(report["history"]),
            "validation_mean_daily_revenue": report["best_validation"]["rolling_rt_only"]["mean_daily_revenue"],
            "test_mean_daily_revenue": report["test"]["rolling_rt_only"]["mean_daily_revenue"],
            "test_executed_hour_mae": report["test"]["mae_executed_hour"],
        })

    selected = max(runs, key=lambda row: row["validation_mean_daily_revenue"])
    test_revenues = np.asarray([row["test_mean_daily_revenue"] for row in runs])
    test_maes = np.asarray([row["test_executed_hour_mae"] for row in runs])
    summary = {
        "selection_rule": "maximum validation RT-only rolling mean daily revenue; test never selects seed",
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "runs": runs,
        "test_across_seeds": {
            "mean_daily_revenue_mean": float(test_revenues.mean()),
            "mean_daily_revenue_sample_std": float(test_revenues.std(ddof=1)) if len(runs) > 1 else 0.0,
            "executed_hour_mae_mean": float(test_maes.mean()),
            "executed_hour_mae_sample_std": float(test_maes.std(ddof=1)) if len(runs) > 1 else 0.0,
        },
        "scope_warning": (
            "This selects an RT forecasting checkpoint with an RT-only diagnostic. "
            "It is not a final full dual-settlement model-selection claim."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
