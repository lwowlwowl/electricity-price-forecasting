#!/usr/bin/env python
"""汇总当前无ID协议的ERCOT结构消融与逐任务替换效应。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.summarize_universal_lomo_multiseed_profit import (
    circular_block_ci,
)


CURRENT = "da_current__rt_at_da_current__rt_current"
PURE = {
    "当前结构 F1/F0/F1": CURRENT,
    "全F0": "da_all_f0__rt_at_da_all_f0__rt_all_f0",
    "全F1": "da_all_f1__rt_at_da_all_f1__rt_all_f1",
}
SURGICAL = {
    "仅DA改为F0": "da_all_f0__rt_at_da_current__rt_current",
    "仅RT-at-DA改为F1": "da_current__rt_at_da_all_f1__rt_current",
    "仅滚动RT改为F0": "da_current__rt_at_da_current__rt_all_f0",
}


def paired(candidate: dict, current: dict, *, seed: int) -> dict:
    candidate_daily = candidate["macro_daily_net_revenue"]
    current_daily = current["macro_daily_net_revenue"]
    dates = sorted(current_daily)
    if dates != sorted(candidate_daily):
        raise ValueError("消融候选与当前结构的逐日键不一致")
    difference = np.asarray(
        [candidate_daily[day] - current_daily[day] for day in dates],
        dtype=np.float64,
    )
    return {
        "mean_daily_difference": float(difference.mean()),
        "median_daily_difference": float(np.median(difference)),
        "positive_difference_day_rate": float(np.mean(difference > 0.0)),
        "seven_day_block_bootstrap_95ci": circular_block_ci(
            difference, block=7, draws=5000, seed=seed
        ),
    }


def summarize(report: dict) -> dict:
    if not report.get("selection_allowed"):
        raise ValueError("结构消融结果未允许用于当前协议工程选择")
    systems = report["systems"]
    current = systems[CURRENT]
    pure = {
        label: {
            "system": name,
            "macro_mean_daily_revenue": systems[name]["macro_mean_daily_revenue"],
            "seven_day_block_bootstrap_95ci": systems[name][
                "macro_mean_daily_revenue_7day_block_bootstrap_95ci"
            ],
        }
        for label, name in PURE.items()
    }
    surgical = {}
    for index, (label, name) in enumerate(SURGICAL.items()):
        surgical[label] = {
            "system": name,
            "macro_mean_daily_revenue": systems[name]["macro_mean_daily_revenue"],
            "paired_vs_current": paired(
                systems[name], current, seed=20261016 + index
            ),
        }
    return {
        "experiment": "universal_lomo_ercot_structure_ablation_summary_v1",
        "status": "2023_inner_ercot_seed0_current_protocol",
        "test_2025_read": False,
        "pure_structures": pure,
        "single_task_replacements": surgical,
        "conclusion": (
            "当前F1/F0/F1明显优于全F0和全F1；仅把滚动RT换为F0的点估计"
            "提高不足1 USD/节点·日且配对区间跨0，不构成稳定替换证据。"
        ),
    }


def markdown(summary: dict) -> str:
    lines = [
        "# ERCOT当前无ID协议结构消融", "",
        "> 固定seed 0、任务损失、训练预算、LP/MPC和结算合同；未读取2025。", "",
        "## 三个预注册整套结构", "",
        "| 结构 | 日均净收益（USD/节点·日） | 自身7日块95%区间 |",
        "|---|---:|---|",
    ]
    best = max(
        item["macro_mean_daily_revenue"]
        for item in summary["pure_structures"].values()
    )
    for label, item in summary["pure_structures"].items():
        value = f"{item['macro_mean_daily_revenue']:.2f}"
        if item["macro_mean_daily_revenue"] == best:
            value = f"**{value}**"
        interval = item["seven_day_block_bootstrap_95ci"]
        lines.append(f"| {label} | {value} | [{interval[0]:.2f}, {interval[1]:.2f}] |")
    lines.extend((
        "", "## 单任务替换诊断", "",
        "| 相对当前结构的唯一改动 | 日均净收益 | 配对收益差 | 配对7日块95%区间 |",
        "|---|---:|---:|---|",
    ))
    for label, item in summary["single_task_replacements"].items():
        paired_item = item["paired_vs_current"]
        interval = paired_item["seven_day_block_bootstrap_95ci"]
        lines.append(
            f"| {label} | {item['macro_mean_daily_revenue']:.2f} | "
            f"{paired_item['mean_daily_difference']:+.2f} | "
            f"[{interval[0]:.2f}, {interval[1]:.2f}] |"
        )
    lines.extend(("", summary["conclusion"], "", "该结论只支持ERCOT当前协议，不外推为四市场普遍最优。"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/results/universal_lomo_ercot_ablation_continuous_mpc_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/universal_lomo_ercot_ablation_summary_v1",
    )
    args = parser.parse_args()
    report = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    result = summarize(report)
    result["source_path"] = args.input
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "event": "structure_ablation_summary_complete",
        "output_dir": args.output_dir,
        "current": result["pure_structures"]["当前结构 F1/F0/F1"][
            "macro_mean_daily_revenue"
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
