#!/usr/bin/env python
"""审计跨年份Global连续MPC收益尺度、成本分解和尖峰集中度。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_LABEL = (
    "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
)


def _system(report: dict) -> dict:
    systems = report["systems"]
    return systems.get(BASELINE_LABEL, next(iter(systems.values())))


def _summarize(report: dict) -> dict:
    system = _system(report)
    days = sorted(system["macro_daily_net_revenue"])
    daily = np.asarray([
        system["macro_daily_net_revenue"][day] for day in days
    ], dtype=np.float64)
    ordered = np.sort(daily)
    nodes = system["nodes"]
    node_names = sorted(nodes)
    components = {
        name: float(np.mean([
            nodes[node]["revenue_components"][name] / nodes[node]["days"]
            for node in node_names
        ]))
        for name in (
            "da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty",
        )
    }
    reconstructed = (
        components["da_leg"] + components["rt_deviation_leg"]
        - components["degradation_cost"] - components["deviation_penalty"]
    )
    mean = float(np.mean(daily))
    if not np.isclose(mean, reconstructed, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"成本分解不能重构净收益: mean={mean}, reconstructed={reconstructed}"
        )
    total = float(np.sum(daily))
    return {
        "split": report["split"],
        "days": len(days),
        "node_count": len(node_names),
        "mean_daily_revenue": mean,
        "median_daily_revenue": float(np.median(daily)),
        "p10_daily_revenue": float(np.quantile(daily, 0.10)),
        "p90_daily_revenue": float(np.quantile(daily, 0.90)),
        "minimum_daily_revenue": float(np.min(daily)),
        "maximum_daily_revenue": float(np.max(daily)),
        "positive_day_rate": float(np.mean(daily > 0.0)),
        "top_5_days_total_revenue_share": float(np.sum(ordered[-5:]) / total),
        "top_10_days_total_revenue_share": float(np.sum(ordered[-10:]) / total),
        "mean_daily_revenue_without_top_5_days": float(np.mean(ordered[:-5])),
        "mean_daily_actual_throughput_mwh": float(np.mean([
            nodes[node]["actual_throughput_mwh"] / nodes[node]["days"]
            for node in node_names
        ])),
        "mean_daily_revenue_components": components,
        "reconstructed_mean_daily_revenue": float(reconstructed),
        "total_soc_violation_count": int(sum(
            nodes[node]["soc_violation_count"] for node in node_names
        )),
        "policy": report["policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner",
        default="data/results/global_xgboost_robust_continuous_mpc_backtest_v1.json",
    )
    parser.add_argument(
        "--outer",
        default="data/results/global_xgboost_robust_continuous_mpc_outer_v1.json",
    )
    parser.add_argument(
        "--output", default="data/results/global_revenue_scale_audit_v1.json"
    )
    args = parser.parse_args()

    inner_report = json.loads((ROOT / args.inner).read_text(encoding="utf-8"))
    outer_report = json.loads((ROOT / args.outer).read_text(encoding="utf-8"))
    inner = _summarize(inner_report)
    outer = _summarize(outer_report)
    for key in (
        "deviation_penalty_enabled",
        "degradation_usd_per_actual_throughput_mwh",
        "terminal_soc_monetized_in_settlement",
    ):
        if inner["policy"][key] != outer["policy"][key]:
            raise ValueError(f"跨年份政策字段不一致: {key}")
    report = {
        "experiment": "global_continuous_mpc_revenue_scale_audit",
        "purpose": "explain mean revenue scale without changing model selection",
        "selection_allowed": False,
        "inner_2023": inner,
        "outer_2024_baseline": outer,
        "comparison": {
            "mean_daily_revenue_difference_2024_minus_2023": (
                outer["mean_daily_revenue"] - inner["mean_daily_revenue"]
            ),
            "median_daily_revenue_difference_2024_minus_2023": (
                outer["median_daily_revenue"] - inner["median_daily_revenue"]
            ),
            "interpretation": (
                "same settlement contract; annual means differ mainly because "
                "electricity-price opportunity distributions are heavy-tailed"
            ),
        },
        "sources": [args.inner, args.outer],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "global_revenue_scale_audit_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "inner_mean": inner["mean_daily_revenue"],
        "outer_mean": outer["mean_daily_revenue"],
        "inner_median": inner["median_daily_revenue"],
        "outer_median": outer["median_daily_revenue"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
