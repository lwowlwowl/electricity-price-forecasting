#!/usr/bin/env python
"""汇总27个新/稳定Global小组合，定位哪一个任务真正改善盈利。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.decision_aware.evaluate_global_continuous_mpc import (  # noqa: E402
    _block_bootstrap_ci,
)
from scripts.decision_aware.summarize_global_taskloss_v1 import (  # noqa: E402
    _economic_summary,
    _parse_triplet,
)


def _paired(left: dict, right: dict) -> dict:
    left_daily = left["macro_daily_net_revenue"]
    right_daily = right["macro_daily_net_revenue"]
    if set(left_daily) != set(right_daily):
        raise ValueError("配对系统的日期不一致")
    difference = np.asarray([
        left_daily[day] - right_daily[day] for day in sorted(left_daily)
    ])
    return {
        "mean_daily_revenue_difference": float(difference.mean()),
        "7day_block_bootstrap_95ci": _block_bootstrap_ci(
            difference, seed=20261010
        ),
    }


def _best_label(backtest: dict, predicate) -> str:
    for row in backtest["ranking"]:
        label = row["triplet"]
        try:
            mapping = _parse_triplet(label)
        except ValueError:
            continue
        if predicate(mapping):
            return label
    raise ValueError("结算结果中没有满足条件的三元组")


def summarize(backtest: dict) -> dict:
    stable_label = "da_stable0__rt_at_da_stable0__rt_stable0"
    xgb_label = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
    systems = backtest["systems"]
    if stable_label not in systems or xgb_label not in systems:
        raise ValueError("结算结果缺少stable0或稳健XGBoost纯三元组")

    best_mixed = _best_label(
        backtest, lambda mapping: any(value.startswith("new") for value in mapping.values())
    )
    replacements = {}
    for task in ("da", "rt_at_da", "rt"):
        label = _best_label(
            backtest,
            lambda mapping, selected=task: (
                mapping[selected].startswith("new")
                and all(
                    value == "stable0"
                    for name, value in mapping.items() if name != selected
                )
            ),
        )
        replacements[task] = {
            "triplet": label,
            "economics": _economic_summary(systems[label]),
            "paired_vs_stable0": _paired(systems[label], systems[stable_label]),
        }

    best_system = systems[best_mixed]
    return {
        "experiment": "global_ercot_taskloss_small_mixed_diagnosis_v1",
        "status": "inner_validation_candidate_not_final",
        "2024_used": False,
        "2025_used": False,
        "searched_triplets": 27,
        "best_triplet_using_at_least_one_new_task": best_mixed,
        "best_economics": _economic_summary(best_system),
        "stable0_economics": _economic_summary(systems[stable_label]),
        "xgboost_robust_economics": _economic_summary(systems[xgb_label]),
        "paired_best_vs_stable0": _paired(best_system, systems[stable_label]),
        "paired_best_vs_xgboost": _paired(best_system, systems[xgb_label]),
        "single_task_replacements": replacements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest",
        default="data/results/global_taskloss_mixed_continuous_mpc_v1.json",
    )
    parser.add_argument(
        "--output", default="data/results/global_taskloss_mixed_summary_v1.json"
    )
    args = parser.parse_args()
    backtest = json.loads((ROOT / args.backtest).read_text(encoding="utf-8"))
    report = summarize(backtest)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "saved": output.relative_to(ROOT).as_posix(),
        "best": report["best_triplet_using_at_least_one_new_task"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
