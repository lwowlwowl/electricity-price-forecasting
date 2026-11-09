#!/usr/bin/env python
"""Paired daily robustness statistics for the controlled Austria DA-leg settlement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import AustriaExternalData  # noqa: E402
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    group_counts,
)


ANCHOR = "decision_aware_v4_final_2to1_blend"
CANDIDATES = (
    "timesfm3_mainu11_proxy6",
    "three_feature_tail_expert_router_da_only",
    "chronos2_fat_proxy14plus3",
    "d1_persist",
)
GROUPS = ("all", "neg", "spike", "ordinary")


def load_daily(path: Path) -> pd.Series:
    payload = json.loads(path.read_text(encoding="utf-8"))
    daily = payload["settlement"]["daily"]
    series = pd.Series(
        {pd.Timestamp(day).date(): float(item["net_revenue"]) for day, item in daily.items()},
        dtype=np.float64,
    ).sort_index()
    if len(series) != 365 or series.index.has_duplicates or not np.isfinite(series).all():
        raise ValueError(f"逐日结算序列异常: {path}")
    return series


def paired_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("配对bootstrap输入无效")
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        count = min(chunk_size, remaining)
        positions = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[positions].mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_v4_advantage_eur_per_day": float(values.mean()),
        "median_v4_advantage_eur_per_day": float(np.median(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_probability_v4_advantage_positive": float(
            np.mean(bootstrap_means > 0.0)
        ),
        "v4_daily_win_rate": float(np.mean(values > 0.0)),
    }


def circular_block_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    block_length: int,
    rng: np.random.Generator,
    chunk_size: int = 500,
) -> dict[str, float]:
    """Bootstrap consecutive circular blocks to retain short-range dependence."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < block_length or block_length < 1:
        raise ValueError("区块bootstrap输入无效")
    block_offsets = np.arange(block_length, dtype=np.int64)
    blocks_per_sample = int(np.ceil(len(values) / block_length))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        count = min(chunk_size, remaining)
        starts = rng.integers(
            0, len(values), size=(count, blocks_per_sample, 1)
        )
        positions = (starts + block_offsets.reshape(1, 1, -1)) % len(values)
        sampled = values[positions.reshape(count, -1)[:, : len(values)]]
        means.append(sampled.mean(axis=1))
        remaining -= count
    bootstrap_means = np.concatenate(means)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "block_length_days": int(block_length),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_probability_v4_advantage_positive": float(
            np.mean(bootstrap_means > 0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--settlement-dir", default="data/results/austria_controlled_da_settlement_v1"
    )
    parser.add_argument(
        "--output", default=(
            "data/results/austria_controlled_da_settlement_v1/robustness.json"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove-extreme-days", type=int, default=5)
    args = parser.parse_args()

    settlement_dir = ROOT / args.settlement_dir
    anchor = load_daily(settlement_dir / f"{ANCHOR}.json")
    data, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    quarter = data.evaluation_quarters(complete_settlement=False)
    local = pd.DatetimeIndex(quarter["timestamp_local"])
    daily_groups = classify_days(
        local,
        quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64),
        spike_floor=SPIKE_FLOOR_EUR_PER_MWH,
    )
    if not anchor.index.equals(daily_groups.index):
        raise ValueError("结算日期与DA日分组不一致")

    results = {}
    for candidate_index, name in enumerate(CANDIDATES):
        candidate = load_daily(settlement_dir / f"{name}.json")
        if not candidate.index.equals(anchor.index):
            raise ValueError(f"{name}逐日日期与v4不一致")
        advantage = anchor - candidate
        by_group = {}
        for group_index, group in enumerate(GROUPS):
            selected = (
                advantage.index
                if group == "all"
                else daily_groups.index[daily_groups["day_group"] == group]
            )
            values = advantage.loc[selected].to_numpy(np.float64)
            rng = np.random.default_rng(
                args.seed + 10_000 * candidate_index + group_index
            )
            by_group[group] = {
                "days": int(len(values)),
                **paired_bootstrap(
                    values,
                    samples=args.bootstrap_samples,
                    rng=rng,
                ),
            }

        extreme = advantage.abs().nlargest(args.remove_extreme_days).index
        retained = advantage.drop(extreme)
        trimmed_rng = np.random.default_rng(
            args.seed + 10_000 * candidate_index + 9_999
        )
        frozen_prefix = advantage.iloc[:335]
        results[name] = {
            "by_day_group": by_group,
            "all_days_circular_block_bootstrap": circular_block_bootstrap(
                advantage.to_numpy(np.float64),
                samples=args.bootstrap_samples,
                block_length=7,
                rng=np.random.default_rng(
                    args.seed + 10_000 * candidate_index + 8_888
                ),
            ),
            "frozen_335_day_prefix_sensitivity": {
                "days": int(len(frozen_prefix)),
                **paired_bootstrap(
                    frozen_prefix.to_numpy(np.float64),
                    samples=args.bootstrap_samples,
                    rng=np.random.default_rng(
                        args.seed + 10_000 * candidate_index + 7_777
                    ),
                ),
                "circular_block_bootstrap": circular_block_bootstrap(
                    frozen_prefix.to_numpy(np.float64),
                    samples=args.bootstrap_samples,
                    block_length=7,
                    rng=np.random.default_rng(
                        args.seed + 10_000 * candidate_index + 7_778
                    ),
                ),
            },
            "remove_largest_absolute_paired_differences": {
                "removed_days": [str(day) for day in extreme],
                "removed_v4_advantage_eur": [
                    float(advantage.loc[day]) for day in extreme
                ],
                "retained_days": int(len(retained)),
                **paired_bootstrap(
                    retained.to_numpy(np.float64),
                    samples=args.bootstrap_samples,
                    rng=trimmed_rng,
                ),
            },
        }

    summary = {
        "protocol": "austria_controlled_da_robustness_v1",
        "status": "complete",
        "comparison_direction": "decision-aware v4 revenue minus candidate revenue",
        "positive_value_means": "decision-aware v4 has higher revenue",
        "bootstrap": {
            "method": "paired nonparametric bootstrap over local delivery days",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "confidence_interval": "percentile 95%",
            "time_dependence_sensitivity": "7-day circular moving-block bootstrap on all days",
        },
        "evaluation": {
            "days": int(len(anchor)),
            "group_counts": group_counts(daily_groups),
            "selection_or_training_on_austria": False,
        },
        "anchor": ANCHOR,
        "results": results,
        "raw_parquet_sha256": protocol["raw_sha256"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
