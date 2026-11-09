#!/usr/bin/env python
"""Evaluate a frozen zero-shot TimesFM3 proxy6 DA/RT-at-DA/rolling-RT triplet."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "external/timesfm/src"))

from decision_aware.austria_external import (  # noqa: E402
    AustriaExternalData,
    PRICE_RT,
    quarter_hour_backtest,
)
from decision_aware.austria_mainu11_proxy import (  # noqa: E402
    PROXY_COLUMNS,
    build_mainu11_proxy_covariates,
)
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    extend_rt_history,
    load_apcs_imbalance,
    verify_overlap,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    group_counts,
    settlement_metrics,
)


MODEL_NAME = "timesfm3_mainu11_proxy6_full_triplet"


def timesfm_batch(model, tasks: list[dict[str, object]], *, horizon: int) -> list[np.ndarray]:
    outputs = model.predict_batch(
        contexts=[np.asarray(item["context"], dtype=np.float32) for item in tasks],
        horizon=horizon,
        past_future_covariates=[
            np.concatenate(
                [np.asarray(item["hist_cov"]), np.asarray(item["future_cov"])],
                axis=0,
            ).T.astype(np.float32)
            for item in tasks
        ],
        return_quantiles=False,
        use_symmetric_averaging=False,
        make_positive=False,
    )
    return [np.asarray(item.forecast, dtype=np.float32).reshape(horizon) for item in outputs]


def context_and_covariates(
    rt: pd.Series,
    covariates: pd.DataFrame,
    issue: pd.Timestamp,
    future_index: pd.DatetimeIndex,
    *,
    context_length: int,
) -> dict[str, object]:
    issue = pd.Timestamp(issue).tz_convert("UTC")
    context_index = pd.date_range(
        issue - pd.Timedelta(minutes=15 * context_length),
        issue - pd.Timedelta(minutes=15),
        freq="15min",
        tz="UTC",
    )
    context = rt.reindex(context_index).to_numpy(np.float32)
    hist_cov = covariates.reindex(context_index).to_numpy(np.float32)
    future_cov = covariates.reindex(future_index).to_numpy(np.float32)
    if (
        len(context) != context_length
        or not np.isfinite(context).all()
        or not np.isfinite(hist_cov).all()
        or not np.isfinite(future_cov).all()
    ):
        raise ValueError(f"{issue}的TimesFM3 RT上下文或协变量不完整")
    return {"context": context, "hist_cov": hist_cov, "future_cov": future_cov}


def predict_rt_at_da(
    model,
    rt: pd.Series,
    covariates: pd.DataFrame,
    dates: list,
    evaluation_index: pd.DatetimeIndex,
    *,
    timezone: str,
    context_length: int,
    batch_size: int,
) -> np.ndarray:
    output = pd.Series(np.nan, index=evaluation_index, dtype=np.float64)
    grouped: dict[int, list[dict[str, object]]] = {}
    for delivery_date in dates:
        delivery_start = pd.Timestamp(delivery_date).tz_localize(timezone)
        delivery_end = (pd.Timestamp(delivery_date) + pd.Timedelta(days=1)).tz_localize(timezone)
        gate = (
            pd.Timestamp(delivery_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        ).tz_localize(timezone)
        gate_utc = gate.tz_convert("UTC")
        forecast_index = pd.date_range(
            gate_utc, delivery_end.tz_convert("UTC"), freq="15min", inclusive="left"
        )
        delivery_index = pd.date_range(
            delivery_start.tz_convert("UTC"),
            delivery_end.tz_convert("UTC"),
            freq="15min",
            inclusive="left",
        )
        task = context_and_covariates(
            rt, covariates, gate_utc, forecast_index, context_length=context_length
        )
        task["delivery_index"] = delivery_index
        task["delivery_offset"] = int(
            (delivery_index[0] - gate_utc) / pd.Timedelta(minutes=15)
        )
        grouped.setdefault(len(forecast_index), []).append(task)

    for horizon, tasks in sorted(grouped.items()):
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            forecasts = timesfm_batch(model, batch, horizon=horizon)
            for task, forecast in zip(batch, forecasts):
                offset = int(task["delivery_offset"])
                delivery_index = task["delivery_index"]
                values = forecast[offset : offset + len(delivery_index)]
                if len(values) != len(delivery_index):
                    raise ValueError("RT-at-DA交付日截取长度错误")
                output.loc[delivery_index] = values
    if output.isna().any():
        raise ValueError("RT-at-DA预测未覆盖完整评价轴")
    return output.to_numpy(np.float64)


def predict_rolling_rt(
    model,
    rt: pd.Series,
    covariates: pd.DataFrame,
    issues: pd.DatetimeIndex,
    *,
    context_length: int,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(issues), batch_size):
        batch_issues = issues[start : start + batch_size]
        tasks = []
        for issue in batch_issues:
            future_index = pd.date_range(issue, periods=16, freq="15min", tz="UTC")
            tasks.append(
                context_and_covariates(
                    rt, covariates, issue, future_index, context_length=context_length
                )
            )
        predictions.extend(timesfm_batch(model, tasks, horizon=16))
        if start == 0 or start + batch_size >= len(issues) or start % 1024 == 0:
            print(json.dumps({
                "event": "rolling_rt_progress",
                "completed_issues": min(start + batch_size, len(issues)),
                "total_issues": len(issues),
            }), flush=True)
    result = np.stack(predictions).astype(np.float64)
    if result.shape != (len(issues), 16):
        raise ValueError(f"rolling RT预测形状错误: {result.shape}")
    return result


def native_rt_windows(
    issues: pd.DatetimeIndex,
    predictions: np.ndarray,
    quarter_index: pd.DatetimeIndex,
) -> list[np.ndarray]:
    issue_to_row = {stamp: row for row, stamp in enumerate(issues)}
    windows = []
    for stamp in quarter_index:
        issue = stamp.floor("1h")
        offset = int((stamp - issue) / pd.Timedelta(minutes=15))
        windows.append(predictions[issue_to_row[issue], offset:])
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument("--raw", default="data/Austria/奥地利_20250801_20260930.parquet")
    parser.add_argument(
        "--september-apcs", default="data/Austria/Statistik-RZ_202609_NEU.xlsx"
    )
    parser.add_argument(
        "--august-apcs", default="data/Austria/Statistik-RZ_202608_NEU.xlsx"
    )
    parser.add_argument(
        "--da-predictions",
        default="data/results/austria_timesfm3_mainu11_proxy_v1/predictions.npz",
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_timesfm3_mainu11_full_triplet_v1"
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    started = perf_counter()
    base, protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    september_rt = load_apcs_imbalance(ROOT / args.september_apcs)
    august_rt = load_apcs_imbalance(ROOT / args.august_apcs)
    overlap = verify_overlap(base, august_rt)
    extended = extend_rt_history(base, september_rt)
    full_quarter = extended.evaluation_quarters(complete_settlement=False)
    evaluation_index = full_quarter.index
    local = pd.DatetimeIndex(full_quarter["timestamp_local"])
    dates = sorted(set(local.date))
    issues = pd.date_range(
        evaluation_index[0].floor("1h"),
        evaluation_index[-1].floor("1h"),
        freq="1h",
        tz="UTC",
    )
    if len(dates) != 365 or len(issues) != 8760 or len(evaluation_index) != 35040:
        raise ValueError("TimesFM3完整三腿评价轴错误")

    raw = pd.read_parquet(ROOT / args.raw)
    covariates = build_mainu11_proxy_covariates(raw, future_quarters=16)
    rt = extended.quarter[PRICE_RT].astype(np.float64)
    with np.load(ROOT / args.da_predictions, allow_pickle=False) as archive:
        da_timestamps = archive["timestamp_unix_seconds"].astype(np.int64)
        da_prediction = archive["prediction_eur_per_mwh"].astype(np.float64)
    expected_seconds = np.asarray([int(item.timestamp()) for item in evaluation_index], dtype=np.int64)
    if not np.array_equal(da_timestamps, expected_seconds):
        raise ValueError("TimesFM3 DA预测时间轴不一致")

    from timesfm3 import ModelConfig, TimesFM3Evaluator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=16 if device == "cuda" else 4,
        device=device,
        cache_dir=str(ROOT / "hf_cache/hub"),
        local_files_only=True,
    ))
    rt_at_da = predict_rt_at_da(
        model, rt, covariates, dates, evaluation_index,
        timezone=extended.timezone, context_length=args.context_length,
        batch_size=args.batch_size,
    )
    print(json.dumps({"event": "rt_at_da_complete"}), flush=True)
    rolling_rt = predict_rolling_rt(
        model, rt, covariates, issues,
        context_length=args.context_length, batch_size=args.batch_size,
    )

    realized_da = full_quarter["日前价格 (EUR/MWh)"].to_numpy(np.float64)
    realized_rt = full_quarter["不平衡最终结算价格 (EUR/MWh)"].to_numpy(np.float64)
    windows = native_rt_windows(issues, rolling_rt, evaluation_index)
    settlement = quarter_hour_backtest(
        da_prediction, rt_at_da, windows, realized_da, realized_rt,
        local, protocol["battery"],
    )
    groups = classify_days(local, realized_da, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)
    report = {"settlement": settlement}
    rolling_point = np.asarray([window[0] for window in windows], dtype=np.float64)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        timestamp_unix_seconds=expected_seconds,
        issue_unix_seconds=np.asarray([int(item.timestamp()) for item in issues], dtype=np.int64),
        da_eur_per_mwh=da_prediction.astype(np.float32),
        rt_at_da_eur_per_mwh=rt_at_da.astype(np.float32),
        rolling_rt_issue_forecasts_eur_per_mwh=rolling_rt.astype(np.float32),
    )
    summary = {
        "protocol": "austria_timesfm3_mainu11_full_triplet_v1",
        "status": "complete",
        "model": MODEL_NAME,
        "strict_external_zero_shot": True,
        "austria_training_finetuning_or_selection": False,
        "teacher_model_claim": False,
        "interpretation": (
            "Project extension of the teacher-family DA proxy: the same frozen "
            "TimesFM3 and proxy6 feature family produce causal RT-at-DA and rolling RT legs."
        ),
        "features": list(PROXY_COLUMNS),
        "timing": {
            "rt_at_da_gate": "D-1 12:00 Europe/Vienna",
            "rt_at_da_forecast": "forecast gate-to-delivery-end; retain delivery day only",
            "rolling_rt_issue": "every UTC hour",
            "rolling_rt_horizon": "16 native quarter-hours (4 hours)",
        },
        "metrics": {
            "rt_at_da_mae_eur_per_mwh": float(np.mean(np.abs(rt_at_da - realized_rt))),
            "rolling_rt_first_available_mae_eur_per_mwh": float(
                np.mean(np.abs(rolling_point - realized_rt))
            ),
        },
        "settlement": {key: value for key, value in settlement.items() if key != "daily"},
        "by_day_group": settlement_metrics(report, groups),
        "group_counts": group_counts(groups),
        "battery": protocol["battery"],
        "data_checks": {"august_apcs_overlap": overlap, "raw_parquet_modified": False},
        "device": device,
        "runtime_seconds": float(perf_counter() - started),
    }
    (output_dir / "report.json").write_text(json.dumps({
        "system": MODEL_NAME,
        "protocol": summary["protocol"],
        "settlement": settlement,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
