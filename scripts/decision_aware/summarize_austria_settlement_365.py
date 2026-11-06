#!/usr/bin/env python
"""Summarize the paired 365-day Austria settlement comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "decision_aware_v4_final_2to1_blend"
COMPARATOR = "three_feature_tail_expert_router"


def block_bootstrap_ci(
    values: np.ndarray,
    *,
    block: int = 7,
    draws: int = 5000,
    seed: int = 20261105,
) -> list[float]:
    """Circular moving-block interval for the paired mean difference."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap输入必须是一维非空序列")
    rng = np.random.default_rng(seed)
    starts = np.arange(values.size)
    block_count = int(np.ceil(values.size / block))
    means = np.empty(draws, dtype=np.float64)
    offsets = np.arange(block)
    for draw in range(draws):
        selected = rng.choice(starts, size=block_count, replace=True)
        sample = np.concatenate([values[(start + offsets) % values.size] for start in selected])
        means[draw] = sample[: values.size].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def load_daily(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["settlement"]["daily"]


def component_means(daily: dict[str, dict[str, float]], dates: list[str]) -> dict[str, float]:
    keys = (
        "da_leg",
        "rt_deviation_leg",
        "degradation_cost",
        "deviation_penalty",
        "throughput_mwh",
    )
    return {
        key: float(np.mean([daily[day][key] for day in dates]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", default="data/results/austria_settlement_365_v1"
    )
    parser.add_argument(
        "--output", default="data/results/austria_settlement_365_v1/comparison.json"
    )
    args = parser.parse_args()

    result_dir = ROOT / args.result_dir
    candidate = load_daily(result_dir / f"{CANDIDATE}.json")
    comparator = load_daily(result_dir / f"{COMPARATOR}.json")
    dates = sorted(candidate)
    if dates != sorted(comparator):
        raise ValueError("两套系统的结算日期不一致")

    candidate_values = np.asarray(
        [candidate[day]["net_revenue"] for day in dates], dtype=np.float64
    )
    comparator_values = np.asarray(
        [comparator[day]["net_revenue"] for day in dates], dtype=np.float64
    )
    difference = candidate_values - comparator_values
    remove_count = min(5, difference.size - 1)
    keep_without_best = np.argsort(difference)[: difference.size - remove_count]

    report = {
        "protocol": "paired_austria_settlement_365_v1",
        "candidate": CANDIDATE,
        "comparator": COMPARATOR,
        "days": len(dates),
        "mean_difference_eur_per_day": float(difference.mean()),
        "median_paired_difference_eur_per_day": float(np.median(difference)),
        "candidate_better_day_rate": float(np.mean(difference > 0.0)),
        "paired_7day_block_bootstrap_95ci": block_bootstrap_ci(difference),
        "mean_difference_without_best_5_difference_days": float(
            difference[keep_without_best].mean()
        ),
        "candidate_mean_without_its_top_5_revenue_days": float(
            np.delete(candidate_values, np.argsort(candidate_values)[-5:]).mean()
        ),
        "comparator_mean_without_its_top_5_revenue_days": float(
            np.delete(comparator_values, np.argsort(comparator_values)[-5:]).mean()
        ),
        "mean_daily_components": {
            CANDIDATE: component_means(candidate, dates),
            COMPARATOR: component_means(comparator, dates),
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
