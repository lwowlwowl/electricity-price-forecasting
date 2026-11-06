#!/usr/bin/env python
"""Summarize frozen Austria DA forecasts by the teacher-provided day groups."""
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

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    expand_daily_hourly_to_quarters,
)
from scripts.decision_aware.run_austria_external_15min import (  # noqa: E402
    _forecast_axes,
    _load_forecast,
)


SPIKE_FLOOR_EUR_PER_MWH = 212.4560
GROUP_ORDER = ("neg", "spike", "ordinary")


def classify_days(
    local_timestamps: pd.DatetimeIndex,
    realized_da: np.ndarray,
    *,
    spike_floor: float = SPIKE_FLOOR_EUR_PER_MWH,
) -> pd.DataFrame:
    """Assign mutually exclusive groups: negative first, then spike, then ordinary."""
    frame = pd.DataFrame(
        {
            "delivery_date": pd.Index(local_timestamps.date),
            "realized_da": np.asarray(realized_da, dtype=np.float64),
        }
    )
    daily = frame.groupby("delivery_date", sort=True)["realized_da"].agg(
        minimum="min", maximum="max", quarters="size"
    )
    daily["has_negative"] = daily["minimum"] < 0.0
    daily["has_spike"] = daily["maximum"] >= float(spike_floor)
    daily["day_group"] = "ordinary"
    daily.loc[daily["has_spike"], "day_group"] = "spike"
    daily.loc[daily["has_negative"], "day_group"] = "neg"
    return daily


def group_counts(daily: pd.DataFrame) -> dict[str, int]:
    counts = daily["day_group"].value_counts()
    return {name: int(counts.get(name, 0)) for name in GROUP_ORDER}


def forecast_metrics(
    local_timestamps: pd.DatetimeIndex,
    realized_da: np.ndarray,
    predicted_da: np.ndarray,
    daily_groups: pd.DataFrame,
) -> dict:
    realized = np.asarray(realized_da, dtype=np.float64)
    predicted = np.asarray(predicted_da, dtype=np.float64)
    if realized.shape != predicted.shape:
        raise ValueError(f"DA真值和预测形状不一致: {realized.shape} vs {predicted.shape}")
    dates = pd.Index(local_timestamps.date)
    errors = np.abs(predicted - realized)
    per_day = pd.Series(errors).groupby(dates).mean()

    def one(mask: np.ndarray, selected_days: pd.Index) -> dict:
        if len(selected_days) == 0:
            return {
                "days": 0,
                "quarters": 0,
                "point_weighted_mae_eur_per_mwh": None,
                "daily_macro_mae_eur_per_mwh": None,
            }
        return {
            "days": int(len(selected_days)),
            "quarters": int(mask.sum()),
            "point_weighted_mae_eur_per_mwh": float(errors[mask].mean()),
            "daily_macro_mae_eur_per_mwh": float(per_day.loc[selected_days].mean()),
        }

    output = {
        "all": one(np.ones(len(errors), dtype=bool), daily_groups.index),
    }
    labels = daily_groups["day_group"]
    for name in GROUP_ORDER:
        selected = labels.index[labels == name]
        output[name] = one(np.asarray(dates.isin(selected)), selected)
    return output


def settlement_metrics(report: dict, daily_groups: pd.DataFrame) -> dict:
    daily = report["settlement"]["daily"]
    values = pd.Series(
        {pd.Timestamp(day).date(): float(item["net_revenue"]) for day, item in daily.items()},
        dtype=np.float64,
    ).sort_index()
    if not values.index.equals(daily_groups.index):
        raise ValueError(f"{report['system']}的结算日期与DA分组日期不一致")

    output = {
        "all": {
            "days": int(len(values)),
            "mean_net_revenue_eur_per_day": float(values.mean()),
            "positive_day_rate": float((values > 0.0).mean()),
        }
    }
    for name in GROUP_ORDER:
        selected = daily_groups.index[daily_groups["day_group"] == name]
        subset = values.loc[selected]
        output[name] = {
            "days": int(len(subset)),
            "mean_net_revenue_eur_per_day": float(subset.mean()),
            "positive_day_rate": float((subset > 0.0).mean()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--result-dir", default="data/results/austria_external_15min_v1"
    )
    parser.add_argument(
        "--output", default="data/results/austria_da_day_groups_v1/summary.json"
    )
    parser.add_argument("--spike-floor", type=float, default=SPIKE_FLOOR_EUR_PER_MWH)
    args = parser.parse_args()

    protocol_path = ROOT / args.protocol
    data, protocol = AustriaExternalData.load(ROOT, protocol_path)
    result_dir = ROOT / args.result_dir

    # Reconcile the teacher board against every DA truth row available locally.
    full_quarter = data.evaluation_quarters(complete_settlement=False)
    full_local = pd.DatetimeIndex(full_quarter["timestamp_local"])
    full_daily = classify_days(
        full_local,
        full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64),
        spike_floor=args.spike_floor,
    )

    # Frozen project models have a strict 335-day common interval because RT truth/history
    # is missing in September 2026.  Do not impute it merely to match the external board.
    quarter = data.evaluation_quarters(complete_settlement=True)
    dates, _issues, local = _forecast_axes(data, quarter)
    realized_da = quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    daily_groups = classify_days(local, realized_da, spike_floor=args.spike_floor)

    systems = {}
    for cache_path in sorted((result_dir / "forecast_cache").glob("*.npz")):
        system = cache_path.stem
        report_path = result_dir / f"{system}.json"
        if not report_path.is_file():
            continue
        values, metadata = _load_forecast(cache_path)
        predicted_da = expand_daily_hourly_to_quarters(
            dates, values["da"].astype(np.float64) / data.eur_to_usd, local
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        systems[system] = {
            "forecast": forecast_metrics(local, realized_da, predicted_da, daily_groups),
            "complete_system_settlement": settlement_metrics(report, daily_groups),
            "forecast_resolution": "frozen hourly forecast repeated to native quarter-hours",
            "strict_external_test": bool(report["strict_external_test"]),
            "evidence": metadata.get("evidence", report.get("evidence", {})),
        }

    ranking = sorted(
        (
            {
                "rank": 0,
                "system": name,
                "all_mae": item["forecast"]["all"]["point_weighted_mae_eur_per_mwh"],
                "neg_mae": item["forecast"]["neg"]["point_weighted_mae_eur_per_mwh"],
                "spike_mae": item["forecast"]["spike"]["point_weighted_mae_eur_per_mwh"],
                "ordinary_mae": item["forecast"]["ordinary"]["point_weighted_mae_eur_per_mwh"],
                "mean_complete_system_revenue_eur_per_day": item[
                    "complete_system_settlement"
                ]["all"]["mean_net_revenue_eur_per_day"],
            }
            for name, item in systems.items()
        ),
        key=lambda item: item["all_mae"],
    )
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank

    board_counts = {"neg": 63, "spike": 112, "ordinary": 191}
    local_counts = group_counts(full_daily)
    summary = {
        "protocol": "austria_da_day_groups_v1",
        "status": "our_frozen_systems_complete_teacher_models_pending_predictions",
        "selection_allowed": False,
        "spike_floor_eur_per_mwh": float(args.spike_floor),
        "group_precedence": "negative first; otherwise spike; otherwise ordinary",
        "teacher_board_reconciliation": {
            "teacher_title": "2025-10-01 to 2026-10-01",
            "teacher_days": 366,
            "teacher_group_counts": board_counts,
            "local_truth_interval": [str(full_local[0]), str(full_local[-1])],
            "local_days": int(len(full_daily)),
            "local_group_counts": local_counts,
            "interpretation": (
                "Local truth exactly matches teacher negative and spike counts; the board has "
                "one additional ordinary day, consistent with including 2026-10-01 while the "
                "local file ends at that day's 00:00 local boundary."
            ),
        },
        "frozen_common_interval": {
            "days": int(len(daily_groups)),
            "quarters": int(len(quarter)),
            "local_interval": [str(local[0]), str(local[-1])],
            "group_counts": group_counts(daily_groups),
        },
        "ranking_by_da_mae": ranking,
        "systems": systems,
        "limitations": [
            "Teacher-board model predictions are not available, so their MAE cannot be recomputed on the 335-day intersection.",
            "Teacher-board model settlement cannot be reconstructed from aggregate MAE.",
            "Complete-system revenue includes each frozen system's own three forecast legs and is not a DA-only controlled-leg comparison.",
        ],
        "raw_sha256": protocol["raw_sha256"],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

