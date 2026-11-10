#!/usr/bin/env python
"""Prepare a v4 DA forecast by inverse-LP projection toward causal EXAA."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_aware.austria_exaa import exaa_quarter_hour_covariate  # noqa: E402
from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    expand_daily_hourly_to_quarters,
    sha256,
)
from decision_aware.inverse_lp_projection import (  # noqa: E402
    project_da_to_target_normal_cone,
)
from decision_aware.policy import BESSSimulator  # noqa: E402
from scripts.decision_aware.run_austria_external_15min import (  # noqa: E402
    _forecast_axes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--v4",
        default=(
            "data/results/austria_da_mae_365_v1/"
            "decision_aware_v4_final_2to1_blend.npz"
        ),
    )
    parser.add_argument(
        "--exaa", default="data/Austria/external_public/EXAA_AT_2025_2026_hourly.csv"
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_v4_inverse_lp_exaa_v1"
    )
    parser.add_argument("--max-days", type=int, default=0)
    args = parser.parse_args()

    base, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    quarter = base.evaluation_quarters(complete_settlement=False)
    dates, _issues, local = _forecast_axes(base, quarter)
    if len(dates) != 365 or len(quarter) != 35040:
        raise ValueError("奥地利365日时间轴错误")

    exaa_path = ROOT / args.exaa
    exaa = pd.read_csv(exaa_path, parse_dates=["timestamp_utc"]).set_index(
        "timestamp_utc"
    )
    anchor = exaa_quarter_hour_covariate(exaa, quarter.index).to_numpy(np.float64)

    v4_path = ROOT / args.v4
    with np.load(v4_path, allow_pickle=False) as archive:
        frozen_dates = [pd.Timestamp(str(item)).date() for item in archive["delivery_dates"]]
        v4_da_hourly = archive["da_usd_per_mwh"].astype(np.float64) / base.eur_to_usd
        v4_rt_hourly = archive["rt_at_da_usd_per_mwh"].astype(np.float64) / base.eur_to_usd
    if frozen_dates != dates:
        raise ValueError("v4冻结日期与奥地利评价日期不一致")
    v4_da = expand_daily_hourly_to_quarters(dates, v4_da_hourly, local)
    # The official controlled-settlement cache stores this leg as float32.
    # Project against that exact numerical contract; normal cones can change
    # under even tiny coefficient perturbations at degenerate LP vertices.
    v4_rt = expand_daily_hourly_to_quarters(
        dates, v4_rt_hourly, local
    ).astype(np.float32).astype(np.float64)

    battery = protocol["battery"]
    simulator = BESSSimulator(
        battery["power_mw"],
        battery["energy_mwh"],
        battery["efficiency_component"],
        battery["initial_soc_mwh"] / battery["energy_mwh"],
        dt=0.25,
        kappa=battery["degradation_eur_per_mwh"],
        soc_min=battery["soc_min_mwh"],
        soc_max=battery["soc_max_mwh"],
        e_cyc=battery["daily_discharge_cap_mwh"],
    )
    positions: dict[object, list[int]] = defaultdict(list)
    for position, stamp in enumerate(local):
        positions[stamp.date()].append(position)
    ordered_dates = sorted(positions)
    if args.max_days > 0:
        ordered_dates = ordered_dates[: args.max_days]

    projected = np.full(len(quarter), np.nan, dtype=np.float64)
    plan_soc = float(battery["initial_soc_mwh"])
    diagnostics: list[dict[str, float | int | str]] = []
    for day_index, delivery_date in enumerate(ordered_dates):
        idx = np.asarray(positions[delivery_date], dtype=np.int64)
        result = project_da_to_target_normal_cone(
            anchor[idx],
            v4_da[idx],
            v4_rt[idx],
            simulator,
            initial_plan_soc_mwh=plan_soc,
            initial_actual_soc_mwh=plan_soc,
        )
        projected[idx] = result.projected_da
        horizon = len(idx)
        discharge = result.target_solution[:horizon]
        charge = result.target_solution[horizon:2 * horizon]
        plan_soc += float(
            np.sum(-discharge / simulator.eta + charge * simulator.eta)
        )
        plan_soc = float(np.clip(plan_soc, simulator.s_min, simulator.s_max))
        diagnostics.append(
            {
                "date": str(delivery_date),
                "quarters": int(horizon),
                "active_constraints": result.active_constraint_count,
                "target_objective_gap": result.target_objective_gap,
                "max_stationarity_residual": result.max_stationarity_residual,
                "max_plan_difference_mwh": result.max_plan_difference_mwh,
                "mean_abs_projection_from_exaa": float(
                    np.mean(np.abs(result.projected_da - anchor[idx]))
                ),
            }
        )
        if (day_index + 1) % 25 == 0 or day_index + 1 == len(ordered_dates):
            print(
                json.dumps(
                    {
                        "event": "inverse_projection_progress",
                        "days": day_index + 1,
                        "last_date": str(delivery_date),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if args.max_days > 0:
        selected = np.isfinite(projected)
    else:
        selected = np.ones(len(projected), dtype=bool)
        if not np.isfinite(projected).all():
            raise ValueError("365日逆投影存在缺失预测")
    summary = {
        "system": "decision-aware-v4-inverse-lp-exaa",
        "status": "diagnostic" if args.max_days > 0 else "frozen",
        "strict_external_zero_shot": True,
        "austria_labels_used": False,
        "external_model_predictions_used": False,
        "method": (
            "L1 projection of causal EXAA onto the active-set normal cone of "
            "the frozen v4 joint DA/RT-at-DA LP"
        ),
        "days": len(ordered_dates),
        "mean_abs_projection_from_exaa_eur_per_mwh": float(
            np.mean(np.abs(projected[selected] - anchor[selected]))
        ),
        "max_target_objective_gap": float(
            max(item["target_objective_gap"] for item in diagnostics)
        ),
        "max_stationarity_residual": float(
            max(item["max_stationarity_residual"] for item in diagnostics)
        ),
        "max_plan_difference_mwh": float(
            max(item["max_plan_difference_mwh"] for item in diagnostics)
        ),
        "sources": {
            "v4_sha256": sha256(v4_path),
            "exaa_sha256": sha256(exaa_path),
            "protocol_sha256": sha256(ROOT / args.protocol),
        },
        "daily_diagnostics": diagnostics,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        timestamp_unix_seconds=np.asarray(
            [int(item.timestamp()) for item in quarter.index[selected]], dtype=np.int64
        ),
        # Preserve the exact normal-cone projection.  Casting here to float32 can
        # move the objective outside the audited optimal face on degenerate LPs.
        prediction_eur_per_mwh=projected[selected].astype(np.float64),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "daily_diagnostics"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
