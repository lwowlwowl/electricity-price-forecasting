#!/usr/bin/env python
"""汇总ERCOT三seed盈利稳定性与相对XGBoost的配对风险。

只读取已经冻结的2023 inner结算JSON；不选取最高盈利seed，也不读取2025。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED_PATTERN = re.compile(r"^da_seed(\d+)__rt_at_da_seed\1__rt_seed\1$")
BASELINE = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
ENSEMBLE = "da_ensemble__rt_at_da_ensemble__rt_ensemble"


def circular_block_ci(
    values: np.ndarray, *, block: int = 7, draws: int = 5000, seed: int = 20261016
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("块Bootstrap要求非空一维序列")
    if block < 1 or draws < 1:
        raise ValueError("block和draws必须为正整数")
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(values.size / block))
    starts = rng.integers(0, values.size, size=(draws, block_count))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % values.size
    samples = values[indices.reshape(draws, -1)[:, : values.size]]
    means = samples.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_report(candidate: dict, baseline: dict, *, seed: int) -> dict:
    candidate_daily = candidate["macro_daily_net_revenue"]
    baseline_daily = baseline["macro_daily_net_revenue"]
    dates = sorted(baseline_daily)
    if dates != sorted(candidate_daily):
        raise ValueError("候选与XGBoost的逐日键不一致")
    difference = np.asarray(
        [candidate_daily[day] - baseline_daily[day] for day in dates],
        dtype=np.float64,
    )
    return {
        "days": len(dates),
        "mean_daily_difference": float(difference.mean()),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=seed
        ),
    }


def summarize(report: dict) -> dict:
    split = str(report.get("split", "")).lower()
    if "inner" not in split or "2023" not in split or "outer" in split:
        raise ValueError("多seed盈利汇总只接受冻结的2023 inner结果")
    systems = report["systems"]
    if BASELINE not in systems or ENSEMBLE not in systems:
        raise ValueError("结算JSON缺少XGBoost或固定等权集成")
    seed_systems = []
    for name, item in systems.items():
        match = SEED_PATTERN.fullmatch(name)
        if match:
            seed_systems.append((int(match.group(1)), name, item))
    seed_systems.sort()
    if [item[0] for item in seed_systems] != [0, 1, 2]:
        raise ValueError("必须同时存在seed 0、1、2，且不得按盈利筛选seed")
    baseline = systems[BASELINE]
    revenues = np.asarray(
        [item[2]["macro_mean_daily_revenue"] for item in seed_systems],
        dtype=np.float64,
    )
    dates = sorted(baseline["macro_daily_net_revenue"])
    seed_mean_daily = {
        day: float(np.mean([
            item[2]["macro_daily_net_revenue"][day] for item in seed_systems
        ]))
        for day in dates
    }
    seed_mean_system = {"macro_daily_net_revenue": seed_mean_daily}
    return {
        "experiment": "universal_lomo_ercot_multiseed_profit_stability_v1",
        "status": "2023_inner_frozen_report_only",
        "test_2025_read": False,
        "seed_selection_by_profit": False,
        "seed_macro_mean_daily_revenue": {
            str(seed): float(item["macro_mean_daily_revenue"])
            for seed, _, item in seed_systems
        },
        "across_seed_mean_daily_revenue": float(revenues.mean()),
        "across_seed_sample_std_daily_revenue": float(revenues.std(ddof=1)),
        "across_seed_min_daily_revenue": float(revenues.min()),
        "across_seed_max_daily_revenue": float(revenues.max()),
        "fixed_equal_weight_ensemble": {
            key: systems[ENSEMBLE][key]
            for key in (
                "macro_mean_daily_revenue",
                "macro_median_daily_revenue",
                "macro_positive_day_rate",
                "macro_mean_daily_revenue_without_top_5_days",
                "top_5_days_total_revenue_share",
                "top_10_days_total_revenue_share",
                "worst_node_mean_daily_revenue",
                "best_node_mean_daily_revenue",
                "macro_mean_daily_revenue_7day_block_bootstrap_95ci",
            )
        },
        "xgboost_local_supervised_upper_baseline": {
            "macro_mean_daily_revenue": baseline["macro_mean_daily_revenue"],
            "macro_mean_daily_revenue_7day_block_bootstrap_95ci": baseline[
                "macro_mean_daily_revenue_7day_block_bootstrap_95ci"
            ],
        },
        "paired_vs_xgboost": {
            **{
                f"seed_{seed}": paired_report(item, baseline, seed=20261016 + seed)
                for seed, _, item in seed_systems
            },
            "across_seed_daily_mean": paired_report(
                seed_mean_system, baseline, seed=20261026
            ),
            "fixed_equal_weight_ensemble": paired_report(
                systems[ENSEMBLE], baseline, seed=20261036
            ),
        },
        "source": report.get("experiment", "unknown"),
    }


def markdown(summary: dict) -> str:
    lines = [
        "# ERCOT无ID模型三seed盈利稳定性", "",
        "> 2023 inner冻结评价；不按盈利选seed；未读取2025。", "",
        "| 系统 | 日均净收益（USD/节点·日） | 相对逐节点XGBoost日均差 | 7日块95%区间 |", 
        "|---|---:|---:|---|",
    ]
    paired = summary["paired_vs_xgboost"]
    for seed, revenue in summary["seed_macro_mean_daily_revenue"].items():
        item = paired[f"seed_{seed}"]
        lines.append(
            f"| seed {seed} | {revenue:.2f} | {item['mean_daily_difference']:.2f} | "
            f"[{item['seven_day_block_bootstrap_95ci'][0]:.2f}, "
            f"{item['seven_day_block_bootstrap_95ci'][1]:.2f}] |"
        )
    ensemble = summary["fixed_equal_weight_ensemble"]
    item = paired["fixed_equal_weight_ensemble"]
    lines.append(
        f"| 固定等权三seed集成 | {ensemble['macro_mean_daily_revenue']:.2f} | "
        f"{item['mean_daily_difference']:.2f} | "
        f"[{item['seven_day_block_bootstrap_95ci'][0]:.2f}, "
        f"{item['seven_day_block_bootstrap_95ci'][1]:.2f}] |"
    )
    lines.extend((
        "", "## 跨seed汇总", "",
        f"- 三seed日均净收益均值：{summary['across_seed_mean_daily_revenue']:.2f}。",
        f"- 三seed样本标准差：{summary['across_seed_sample_std_daily_revenue']:.2f}。",
        f"- 固定等权集成去掉最高5日后：{ensemble['macro_mean_daily_revenue_without_top_5_days']:.2f}。",
        "- 等权集成是预先固定组合，不是从三个seed中挑选最高盈利者。",
    ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/results/universal_lomo_ercot_multiseed_continuous_mpc_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/universal_lomo_ercot_multiseed_profit_summary_v1",
    )
    args = parser.parse_args()
    source = ROOT / args.input
    report = json.loads(source.read_text(encoding="utf-8"))
    result = summarize(report)
    result["source_path"] = args.input
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "event": "multiseed_profit_summary_complete",
        "output_dir": args.output_dir,
        "seed_mean": result["across_seed_mean_daily_revenue"],
        "seed_std": result["across_seed_sample_std_daily_revenue"],
        "ensemble": result["fixed_equal_weight_ensemble"]["macro_mean_daily_revenue"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
