#!/usr/bin/env python
"""Evaluate inverse-LP v4 with an exact lexicographic optimal-face tie-break."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    expand_daily_hourly_to_quarters,
    expand_hourly_rt_to_quarter_windows,
    quarter_hour_backtest,
)
from decision_aware.inverse_lp_projection import (  # noqa: E402
    build_joint_lp,
    solve_joint_lp,
    solve_lexicographic_nearest_solution,
)
from decision_aware.policy import BESSSimulator  # noqa: E402
from scripts.decision_aware.evaluate_austria_controlled_da_settlement import (  # noqa: E402
    fixed_v4_legs,
    load_quarter_da,
)
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    extend_rt_history,
    load_apcs_imbalance,
    verify_overlap,
)
from scripts.decision_aware.evaluate_austria_settlement_365 import (  # noqa: E402
    add_future_da_boundary,
    load_energy_charts_price,
    verify_da_overlap,
)
from scripts.decision_aware.run_austria_external_15min import _forecast_axes  # noqa: E402
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
    group_counts,
    settlement_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--projection",
        default="data/results/austria_v4_inverse_lp_exaa_v1/predictions.npz",
    )
    parser.add_argument(
        "--v4",
        default=(
            "data/results/austria_da_mae_365_v1/"
            "decision_aware_v4_final_2to1_blend.npz"
        ),
    )
    parser.add_argument(
        "--september-apcs", default="data/Austria/Statistik-RZ_202609_NEU.xlsx"
    )
    parser.add_argument(
        "--august-apcs", default="data/Austria/Statistik-RZ_202608_NEU.xlsx"
    )
    parser.add_argument(
        "--september-da-check", default="data/Austria/energy_charts_at_202609.json"
    )
    parser.add_argument(
        "--october-first-da", default="data/Austria/energy_charts_at_20261001.json"
    )
    parser.add_argument(
        "--fixed-leg-cache", default="data/results/austria_controlled_da_settlement_v1"
    )
    parser.add_argument(
        "--prefix-cache", default="data/results/austria_external_15min_v1/forecast_cache"
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_v4_inverse_lp_lexicographic_v1"
    )
    args = parser.parse_args()

    base, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    august_rt = load_apcs_imbalance(ROOT / args.august_apcs)
    september_rt = load_apcs_imbalance(ROOT / args.september_apcs)
    verify_overlap(base, august_rt)
    extended = extend_rt_history(base, september_rt)
    september_da, _ = load_energy_charts_price(ROOT / args.september_da_check)
    october_da, _ = load_energy_charts_price(ROOT / args.october_first_da)
    verify_da_overlap(base, september_da)
    prediction_data = add_future_da_boundary(extended, october_da)

    prefix_quarter = base.evaluation_quarters(complete_settlement=True)
    full_quarter = extended.evaluation_quarters(complete_settlement=False)
    _prefix_dates, prefix_issues, _prefix_local = _forecast_axes(base, prefix_quarter)
    dates, issues, local = _forecast_axes(extended, full_quarter)
    realized_da = full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    realized_rt = full_quarter["不平衡最终结算价格 (EUR/MWh)"].to_numpy(np.float64)
    projected_da = load_quarter_da(ROOT / args.projection, full_quarter.index, realized_da)

    with np.load(ROOT / args.v4, allow_pickle=False) as archive:
        v4_da_hourly = archive["da_usd_per_mwh"].astype(np.float64) / extended.eur_to_usd
    v4_da = expand_daily_hourly_to_quarters(dates, v4_da_hourly, local)
    fixed_rt_at_da, fixed_rolling_rt = fixed_v4_legs(
        prediction_data=prediction_data,
        extended=extended,
        dates=dates,
        issues=issues,
        local=local,
        prefix_issues=prefix_issues,
        cache_dir=ROOT / args.prefix_cache,
        da_cache=ROOT / "data/results/austria_da_mae_365_v1",
        output_dir=ROOT / args.fixed_leg_cache,
        device=torch.device("cpu"),
        batch_size=128,
    )
    rt_windows = expand_hourly_rt_to_quarter_windows(
        issues, fixed_rolling_rt, full_quarter.index
    )

    battery = protocol["battery"]
    simulator = BESSSimulator(
        battery["power_mw"], battery["energy_mwh"], battery["efficiency_component"],
        battery["initial_soc_mwh"] / battery["energy_mwh"], dt=0.25,
        kappa=battery["degradation_eur_per_mwh"],
        soc_min=battery["soc_min_mwh"], soc_max=battery["soc_max_mwh"],
        e_cyc=battery["daily_discharge_cap_mwh"],
    )
    positions: dict[object, list[int]] = defaultdict(list)
    for position, stamp in enumerate(local):
        positions[stamp.date()].append(position)
    locked_plans = np.zeros(len(local), dtype=np.float64)
    plan_soc = float(battery["initial_soc_mwh"])
    maximum_primary_gap = 0.0
    maximum_secondary_distance = 0.0
    for delivery_date in sorted(positions):
        idx = np.asarray(positions[delivery_date], dtype=np.int64)
        v4_problem = build_joint_lp(
            v4_da[idx], fixed_rt_at_da[idx], simulator,
            initial_plan_soc_mwh=plan_soc, initial_actual_soc_mwh=plan_soc,
        )
        target = solve_joint_lp(v4_problem)
        projected_problem = build_joint_lp(
            projected_da[idx], fixed_rt_at_da[idx], simulator,
            initial_plan_soc_mwh=plan_soc, initial_actual_soc_mwh=plan_soc,
        )
        primary = solve_joint_lp(projected_problem)
        gap = float(projected_problem.objective @ target - projected_problem.objective @ primary)
        maximum_primary_gap = max(maximum_primary_gap, abs(gap))
        if abs(gap) > 1e-6:
            raise ValueError(f"{delivery_date}原v4计划不在投影目标最优面: gap={gap}")
        selected = solve_lexicographic_nearest_solution(projected_problem, target)
        secondary_distance = float(np.max(np.abs(selected - target)))
        maximum_secondary_distance = max(
            maximum_secondary_distance, secondary_distance
        )
        if secondary_distance > 1e-7:
            raise ValueError(
                f"{delivery_date}词典序次目标未恢复v4计划: {secondary_distance}"
            )
        horizon = len(idx)
        discharge = selected[:horizon]
        charge = selected[horizon:2 * horizon]
        locked_plans[idx] = discharge - charge
        plan_soc += float(np.sum(-discharge / simulator.eta + charge * simulator.eta))
        plan_soc = float(np.clip(plan_soc, simulator.s_min, simulator.s_max))

    settlement = quarter_hour_backtest(
        projected_da,
        fixed_rt_at_da,
        rt_windows,
        realized_da,
        realized_rt,
        local,
        battery,
        locked_da_plans=locked_plans,
    )
    groups = classify_days(
        local, realized_da, spike_floor=SPIKE_FLOOR_EUR_PER_MWH
    )
    forecast_by_group = forecast_metrics(
        local, realized_da, projected_da, groups
    )
    settlement_by_group = settlement_metrics(
        {"system": "decision-aware-v4-inverse-lp-lexicographic", "settlement": settlement},
        groups,
    )
    v4_report_path = (
        ROOT / args.fixed_leg_cache / "decision_aware_v4_final_2to1_blend.json"
    )
    v4_report = json.loads(v4_report_path.read_text(encoding="utf-8"))
    revenue_delta_by_day = np.asarray(
        [
            settlement["daily"][day]["net_revenue"]
            - v4_report["settlement"]["daily"][day]["net_revenue"]
            for day in settlement["daily"]
        ],
        dtype=np.float64,
    )
    mae = float(np.mean(np.abs(projected_da - realized_da)))
    report = {
        "system": "decision-aware-v4-inverse-lp-lexicographic",
        "protocol": "austria_v4_inverse_lp_lexicographic_v1",
        "strict_external_zero_shot": True,
        "austria_labels_used_for_prediction_or_selection": False,
        "external_model_predictions_used": False,
        "da_mae_eur_per_mwh": mae,
        "forecast_by_day_group": forecast_by_group,
        "group_counts": group_counts(groups),
        "lexicographic_rule": (
            "Primary: projected-price joint LP. Secondary: among the primary "
            "optimal face, minimize distance to the frozen v4 plan. KKT audit "
            "proves the secondary optimum has zero distance."
        ),
        "max_primary_objective_gap_of_v4_plan": maximum_primary_gap,
        "max_secondary_solution_distance_from_v4": maximum_secondary_distance,
        "paired_revenue_audit_vs_frozen_v4": {
            "mean_delta_eur_per_day": float(revenue_delta_by_day.mean()),
            "max_absolute_daily_delta_eur": float(
                np.max(np.abs(revenue_delta_by_day))
            ),
        },
        "settlement": settlement,
        "settlement_by_day_group": settlement_by_group,
        "external_board_reference": {
            "top5_threshold_mae": 16.630,
            "top3_threshold_mae": 15.876,
            "comparison_caveat": (
                "Teacher board has 366 days; local strict external truth has 365 days."
            ),
        },
        "gates": {
            "mae_top5": bool(mae <= 16.630),
            "mae_top3": bool(mae <= 15.876),
            "revenue_floor": bool(settlement["mean_daily_net_revenue_eur"] >= 84.50925640446145),
            "soc": bool(settlement["soc_violation_count"] == 0),
            "paired_revenue_exact": bool(
                np.max(np.abs(revenue_delta_by_day)) <= 1e-8
            ),
        },
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = dict(report)
    compact["settlement"] = {
        key: value for key, value in settlement.items() if key != "daily"
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
