#!/usr/bin/env python
"""Settle candidate DA forecasts with one frozen v4 RT-at-DA/rolling-RT pair."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

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
    sha256,
)
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    extend_rt_history,
    load_apcs_imbalance,
    verify_overlap,
)
from scripts.decision_aware.evaluate_austria_settlement_365 import (  # noqa: E402
    add_future_da_boundary,
    append_rt_prefix,
    load_energy_charts_price,
    predict_rt_suffix,
    robust_rt,
    verify_da_overlap,
)
from scripts.decision_aware.run_austria_external_15min import (  # noqa: E402
    V4_ROOTS,
    _forecast_axes,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
    group_counts,
    settlement_metrics,
)


QUARTER_DA_CANDIDATES = {
    "timesfm3_mainu11_proxy6": (
        "data/results/austria_timesfm3_mainu11_proxy_v1/predictions.npz"
    ),
    "chronos2_fat_proxy14plus3": (
        "data/results/austria_chronos2_fat_proxy_v1/predictions.npz"
    ),
    "d1_persist": (
        "data/results/austria_teacher_simple_baselines_v1/d1_persist_predictions.npz"
    ),
}
DAILY_DA_CANDIDATES = {
    "decision_aware_v4_final_2to1_blend": (
        "data/results/austria_da_mae_365_v1/decision_aware_v4_final_2to1_blend.npz"
    ),
    "three_feature_tail_expert_router_da_only": (
        "data/results/austria_da_mae_365_v1/three_feature_tail_expert_router.npz"
    ),
}
STRICT_EXTERNAL_STATUS = {
    "decision_aware_v4_final_2to1_blend": True,
    "three_feature_tail_expert_router_da_only": True,
    "timesfm3_mainu11_proxy6": True,
    "chronos2_fat_proxy14plus3": True,
    "d1_persist": True,
}


def load_quarter_da(
    path: Path,
    evaluation_index: pd.DatetimeIndex,
    truth: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        timestamps = archive["timestamp_unix_seconds"].astype(np.int64)
        prediction = archive["prediction_eur_per_mwh"].astype(np.float64)
        cached_truth = archive["truth_eur_per_mwh"].astype(np.float64)
    expected = np.asarray([int(item.timestamp()) for item in evaluation_index], dtype=np.int64)
    if not np.array_equal(timestamps, expected):
        raise ValueError(f"候选DA时间轴不一致: {path}")
    if prediction.shape != truth.shape or not np.allclose(cached_truth, truth, atol=1e-5):
        raise ValueError(f"候选DA真值或预测形状不一致: {path}")
    return prediction


def load_daily_da(
    path: Path,
    dates: list,
    local: pd.DatetimeIndex,
    eur_to_usd: float,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["da_usd_per_mwh"].astype(np.float64) / eur_to_usd
    if prediction.shape != (len(dates), 24):
        raise ValueError(f"冻结DA日预测形状错误: {path}: {prediction.shape}")
    return expand_daily_hourly_to_quarters(dates, prediction, local)


def fixed_v4_legs(
    *,
    prediction_data: AustriaExternalData,
    extended: AustriaExternalData,
    dates: list,
    issues: pd.DatetimeIndex,
    local: pd.DatetimeIndex,
    prefix_issues: pd.DatetimeIndex,
    cache_dir: Path,
    da_cache: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = output_dir / "fixed_v4_legs.npz"
    expected_issue_seconds = np.asarray(
        [int(item.timestamp()) for item in issues], dtype=np.int64
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_issues = archive["issue_unix_seconds"].astype(np.int64)
            rt_at_da_quarter = archive["rt_at_da_quarter_eur_per_mwh"].astype(np.float64)
            rolling_rt = archive["rolling_rt_hourly_eur_per_mwh"].astype(np.float64)
        if np.array_equal(cached_issues, expected_issue_seconds):
            return rt_at_da_quarter, rolling_rt

    suffix_issues = issues[len(prefix_issues):]
    members: list[np.ndarray] = []
    for seed in (0, 1, 2):
        suffix = predict_rt_suffix(
            prediction_data,
            suffix_issues,
            V4_ROOTS["rt"],
            seed,
            device,
            batch_size,
        )
        members.append(
            append_rt_prefix(
                cache_dir,
                f"decision_aware_v4_seed{seed}_member",
                suffix,
                len(prefix_issues),
            )
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rolling_rt = robust_rt(members, issues, extended.timezone)
    rolling_rt = (
        (2.0 / 3.0) * members[0] + (1.0 / 3.0) * rolling_rt
    ).astype(np.float64) / extended.eur_to_usd

    v4_path = da_cache / "decision_aware_v4_final_2to1_blend.npz"
    with np.load(v4_path, allow_pickle=False) as archive:
        rt_at_da = archive["rt_at_da_usd_per_mwh"].astype(np.float64) / extended.eur_to_usd
    rt_at_da_quarter = expand_daily_hourly_to_quarters(dates, rt_at_da, local)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        issue_unix_seconds=expected_issue_seconds,
        rt_at_da_quarter_eur_per_mwh=rt_at_da_quarter.astype(np.float32),
        rolling_rt_hourly_eur_per_mwh=rolling_rt.astype(np.float32),
    )
    return rt_at_da_quarter, rolling_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
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
        "--prefix-cache", default="data/results/austria_external_15min_v1/forecast_cache"
    )
    parser.add_argument("--da-cache", default="data/results/austria_da_mae_365_v1")
    parser.add_argument(
        "--output-dir", default="data/results/austria_controlled_da_settlement_v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    started = perf_counter()
    base, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    september_rt_path = ROOT / args.september_apcs
    august_rt_path = ROOT / args.august_apcs
    september_rt = load_apcs_imbalance(september_rt_path)
    august_rt = load_apcs_imbalance(august_rt_path)
    overlap = verify_overlap(base, august_rt)
    extended = extend_rt_history(base, september_rt)

    september_da, _ = load_energy_charts_price(ROOT / args.september_da_check)
    october_da, _ = load_energy_charts_price(ROOT / args.october_first_da)
    verify_da_overlap(base, september_da)
    prediction_data = add_future_da_boundary(extended, october_da)

    prefix_quarter = base.evaluation_quarters(complete_settlement=True)
    full_quarter = extended.evaluation_quarters(complete_settlement=False)
    _prefix_dates, prefix_issues, _prefix_local = _forecast_axes(base, prefix_quarter)
    dates, issues, local = _forecast_axes(extended, full_quarter)
    if len(dates) != 365 or len(issues) != 8760 or len(full_quarter) != 35040:
        raise ValueError("受控DA腿365天轴错误")

    realized_da = full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    realized_rt = full_quarter["不平衡最终结算价格 (EUR/MWh)"].to_numpy(np.float64)
    if not np.isfinite(realized_da).all() or not np.isfinite(realized_rt).all():
        raise ValueError("受控DA腿结算真值不完整")
    groups = classify_days(local, realized_da, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    fixed_rt_at_da, fixed_rolling_rt = fixed_v4_legs(
        prediction_data=prediction_data,
        extended=extended,
        dates=dates,
        issues=issues,
        local=local,
        prefix_issues=prefix_issues,
        cache_dir=ROOT / args.prefix_cache,
        da_cache=ROOT / args.da_cache,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
    )
    rt_windows = expand_hourly_rt_to_quarter_windows(
        issues, fixed_rolling_rt, full_quarter.index
    )

    candidates: dict[str, np.ndarray] = {}
    for name, relative in DAILY_DA_CANDIDATES.items():
        candidates[name] = load_daily_da(
            ROOT / relative, dates, local, extended.eur_to_usd
        )
    for name, relative in QUARTER_DA_CANDIDATES.items():
        candidates[name] = load_quarter_da(
            ROOT / relative, full_quarter.index, realized_da
        )

    systems = {}
    for name, da_quarter in candidates.items():
        report_path = output_dir / f"{name}.json"
        settlement = None
        if report_path.is_file():
            cached_report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                cached_report.get("protocol")
                == "austria_controlled_da_settlement_v1"
                and cached_report.get("fixed_legs")
                == "decision-aware v4 RT-at-DA and rolling RT"
            ):
                settlement = cached_report.get("settlement")
        if settlement is None:
            settlement = quarter_hour_backtest(
                da_quarter,
                fixed_rt_at_da,
                rt_windows,
                realized_da,
                realized_rt,
                local,
                protocol["battery"],
            )
        da_metrics = forecast_metrics(local, realized_da, da_quarter, groups)
        systems[name] = {
            "strict_external_zero_shot": STRICT_EXTERNAL_STATUS[name],
            "da_mae_eur_per_mwh": float(
                da_metrics["all"]["point_weighted_mae_eur_per_mwh"]
            ),
            "settlement": {
                key: value for key, value in settlement.items() if key != "daily"
            },
            "by_day_group": settlement_metrics({"settlement": settlement}, groups),
        }
        report_path.write_text(
            json.dumps(
                {
                    "system": name,
                    "protocol": "austria_controlled_da_settlement_v1",
                    "fixed_legs": "decision-aware v4 RT-at-DA and rolling RT",
                    "settlement": settlement,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "candidate_complete",
                    "system": name,
                    "da_mae": systems[name]["da_mae_eur_per_mwh"],
                    "mean_daily_net_revenue_eur": settlement[
                        "mean_daily_net_revenue_eur"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    anchor = systems["decision_aware_v4_final_2to1_blend"]["settlement"][
        "mean_daily_net_revenue_eur"
    ]
    for item in systems.values():
        item["delta_revenue_vs_v4_eur_per_day"] = (
            item["settlement"]["mean_daily_net_revenue_eur"] - anchor
        )
    ranking = sorted(
        (
            {
                "rank": 0,
                "system": name,
                "da_mae_eur_per_mwh": item["da_mae_eur_per_mwh"],
                "mean_daily_net_revenue_eur": item["settlement"][
                    "mean_daily_net_revenue_eur"
                ],
                "delta_revenue_vs_v4_eur_per_day": item[
                    "delta_revenue_vs_v4_eur_per_day"
                ],
            }
            for name, item in systems.items()
        ),
        key=lambda item: item["mean_daily_net_revenue_eur"],
        reverse=True,
    )
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank

    prior_v4 = json.loads(
        (ROOT / "data/results/austria_settlement_365_v1/summary.json").read_text(
            encoding="utf-8"
        )
    )["systems"]["decision_aware_v4_final_2to1_blend"]["settlement"][
        "mean_daily_net_revenue_eur"
    ]
    regression_gap = float(anchor - prior_v4)
    if abs(regression_gap) > 1e-3:
        raise ValueError(f"固定v4腿未复现既有结算: 差值={regression_gap}")

    summary = {
        "protocol": "austria_controlled_da_settlement_v1",
        "status": "complete",
        "interpretation": (
            "Only the DA forecast changes. Every candidate uses the identical frozen "
            "decision-aware v4 RT-at-DA and rolling-RT forecasts, battery, LP/MPC, and truth."
        ),
        "not_full_system_claim": True,
        "evaluation": {
            "days": 365,
            "quarters": int(len(full_quarter)),
            "group_counts": group_counts(groups),
        },
        "fixed_legs": {
            "rt_at_da": "decision_aware_v4_final_2to1_blend",
            "rolling_rt": "decision_aware_v4_final_2to1_blend",
        },
        "v4_regression_check_eur_per_day": {
            "recomputed": anchor,
            "previous": prior_v4,
            "gap": regression_gap,
        },
        "ranking_by_mean_daily_net_revenue": ranking,
        "systems": systems,
        "battery": protocol["battery"],
        "sources": {
            "september_rt_sha256": sha256(september_rt_path),
            "august_overlap_verification": overlap,
        },
        "runtime_seconds": float(perf_counter() - started),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
