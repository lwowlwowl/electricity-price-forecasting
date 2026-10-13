#!/usr/bin/env python
"""审计冻结Global候选相对基线的配对收益稳定性，不参与选模。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = (
    "da_blendgm15__rt_at_da_xgboost_robust__rt_xgboost_robust"
)
DEFAULT_BASELINE = (
    "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
)


def _block_ci(
    values: np.ndarray, block: int, *, draws: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    starts = np.arange(len(values))
    offsets = np.arange(block)
    block_count = int(np.ceil(len(values) / block))
    chosen = rng.choice(starts, size=(draws, block_count))
    indices = (
        chosen[:, :, None] + offsets[None, None, :]
    ) % len(values)
    indices = indices.reshape(draws, -1)[:, :len(values)]
    means = np.mean(values[indices], axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/results/global_da_median_weight_screen_continuous_mpc_v1.json",
    )
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--blocks", default="1,7,14,28")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="data/results/global_da_median_weight_paired_robustness_v1.json",
    )
    args = parser.parse_args()

    report = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    candidate = report["systems"][args.candidate]
    baseline = report["systems"][args.baseline]
    dates = sorted(baseline["macro_daily_net_revenue"])
    if dates != sorted(candidate["macro_daily_net_revenue"]):
        raise ValueError("候选与基线日期不一致")
    daily = np.asarray([
        candidate["macro_daily_net_revenue"][day]
        - baseline["macro_daily_net_revenue"][day]
        for day in dates
    ], dtype=np.float64)
    nodes = sorted(baseline["nodes"])
    if nodes != sorted(candidate["nodes"]):
        raise ValueError("候选与基线节点不一致")
    node_differences = {
        node: float(
            candidate["nodes"][node]["mean_daily_revenue"]
            - baseline["nodes"][node]["mean_daily_revenue"]
        )
        for node in nodes
    }
    blocks = tuple(int(value) for value in args.blocks.split(",") if value.strip())
    if not blocks or any(value < 1 for value in blocks):
        raise ValueError("blocks必须是正整数")
    output_report = {
        "experiment": "global_da_frozen_candidate_paired_revenue_robustness",
        "status": "2023_inner_validation_report_only_after_weight_freeze",
        "selection_allowed": False,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "days": len(dates),
        "nodes": len(nodes),
        "mean_daily_difference": float(np.mean(daily)),
        "median_daily_difference": float(np.median(daily)),
        "positive_difference_day_rate": float(np.mean(daily > 0.0)),
        "p10_daily_difference": float(np.quantile(daily, 0.10)),
        "p90_daily_difference": float(np.quantile(daily, 0.90)),
        "minimum_daily_difference": float(np.min(daily)),
        "maximum_daily_difference": float(np.max(daily)),
        "block_bootstrap_95ci": {
            str(block): _block_ci(
                daily, block, draws=args.draws, seed=20261009 + block
            )
            for block in blocks
        },
        "positive_node_rate": float(np.mean([
            value > 0.0 for value in node_differences.values()
        ])),
        "node_mean_daily_differences": node_differences,
        "warning": (
            "diagnostic only; weights were frozen before this sensitivity audit"
        ),
        "source": args.input,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "event": "paired_revenue_robustness_saved",
        "output": output.relative_to(ROOT).as_posix(),
        "mean": output_report["mean_daily_difference"],
        "median": output_report["median_daily_difference"],
        "positive_day_rate": output_report["positive_difference_day_rate"],
        "positive_node_rate": output_report["positive_node_rate"],
        "intervals": output_report["block_bootstrap_95ci"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
