#!/usr/bin/env python
"""Aggregate frozen Austria system reports without any model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def block_bootstrap(values: np.ndarray, *, draws: int = 5000, seed: int = 20261025) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    block = 7
    starts = np.arange(len(values))
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pieces = []
        length = 0
        while length < len(values):
            start = int(rng.choice(starts))
            piece = values[(start + np.arange(block)) % len(values)]
            pieces.append(piece)
            length += len(piece)
        means[draw] = np.concatenate(pieces)[:len(values)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired(candidate: dict, baseline: dict) -> dict:
    left = candidate["settlement"]["daily"]
    right = baseline["settlement"]["daily"]
    dates = sorted(set(left) & set(right))
    if dates != sorted(left) or dates != sorted(right):
        raise ValueError("系统结算日期不完全一致")
    delta = np.asarray([
        left[day]["net_revenue"] - right[day]["net_revenue"] for day in dates
    ], dtype=np.float64)
    order = np.argsort(delta)
    worst_count = max(1, int(np.ceil(0.10 * len(delta))))
    keep_without_best5 = order[:-5]
    return {
        "days": len(delta),
        "mean_difference_eur_per_day": float(delta.mean()),
        "paired_7day_block_bootstrap_95ci": block_bootstrap(delta),
        "positive_day_rate": float(np.mean(delta > 0.0)),
        "mean_difference_without_best_5_delta_days": float(delta[keep_without_best5].mean()),
        "worst_10pct_mean_difference": float(delta[order[:worst_count]].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/decision_aware/austria_external_15min_v1.json")
    parser.add_argument("--result-dir", default="data/results/austria_external_15min_v1")
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol).read_text(encoding="utf-8"))
    directory = ROOT / args.result_dir
    systems = {}
    missing = []
    for name in protocol["eligible_systems"]:
        path = directory / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("strict_external_test") or report.get("austria_training_used"):
            raise ValueError(f"{name}不是严格外部测试输出")
        systems[name] = report
    if missing:
        raise FileNotFoundError(f"参评系统结果缺失: {missing}")
    ranking = sorted([
        {
            "rank": 0,
            "system": name,
            "mean_daily_net_revenue_eur": report["settlement"]["mean_daily_net_revenue_eur"],
            "da_mae_eur_per_mwh": report["forecast_metrics_eur_per_mwh"]["da"]["mae"],
            "rolling_rt_mae_eur_per_mwh": report["forecast_metrics_eur_per_mwh"]["rolling_rt_executed_step"]["mae"],
            "soc_violation_count": report["settlement"]["soc_violation_count"],
        }
        for name, report in systems.items()
    ], key=lambda item: item["mean_daily_net_revenue_eur"], reverse=True)
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank

    strongest_baseline = max(
        (name for name in ("seasonal_lag168", "pooled_xgboost", "dlinear", "timesfm25_lora", "timesfm3_frozen")),
        key=lambda name: systems[name]["settlement"]["mean_daily_net_revenue_eur"],
    )
    primary = "decision_aware_v4_final_2to1_blend"
    comparisons = {
        name: paired(systems[primary], systems[name])
        for name in (
            strongest_baseline, "dlinear", "pooled_xgboost",
            "transformer_three_seed_equal", "three_feature_tail_expert_router",
        )
    }
    for name, value in comparisons.items():
        baseline_value = systems[name]["settlement"]["mean_daily_net_revenue_eur"]
        value["relative_improvement_percent"] = (
            100.0 * value["mean_difference_eur_per_day"] / abs(baseline_value)
            if abs(baseline_value) > 1e-12 else None
        )

    component_names = ("da_leg", "rt_deviation_leg", "degradation_cost", "deviation_penalty")
    components = {}
    for name, report in systems.items():
        daily = report["settlement"]["daily"]
        components[name] = {
            key: float(np.mean([item[key] for item in daily.values()]))
            for key in component_names
        }
    summary = {
        "protocol": protocol["protocol_version"],
        "status": "complete_external_report_only_no_selection",
        "selection_allowed": False,
        "austria_used_for_training_finetuning_seed_or_threshold_selection": False,
        "days": next(iter(systems.values()))["days"],
        "quarter_rows": next(iter(systems.values()))["quarter_rows"],
        "unit": "EUR_per_Austria_bidding_zone_day",
        "strongest_transferable_frozen_baseline": strongest_baseline,
        "primary_frozen_method": primary,
        "ranking": ranking,
        "primary_paired_comparisons": comparisons,
        "mean_daily_revenue_components": components,
        "ineligible_systems": protocol["ineligible_systems"],
        "raw_sha256": protocol["raw_sha256"],
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
