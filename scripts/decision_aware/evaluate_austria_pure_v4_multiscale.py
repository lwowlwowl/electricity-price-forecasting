#!/usr/bin/env python
"""Evaluate a source-trained pure-v4 decoder on frozen Austria data.

The decoder never sees Austria price labels during training or selection.  Its
only target-horizon inputs are public day-ahead load, wind and solar forecasts;
EXAA and external price-model predictions are explicitly absent.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
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
    normalize,
    robust_context_stats,
)
from decision_aware.native_multiscale import PureV4MultiscaleDecoder  # noqa: E402
from scripts.decision_aware.evaluate_austria_da_mae_365 import (  # noqa: E402
    DA_TASKS,
    append_pair,
    extend_rt_history,
    load_apcs_imbalance,
    load_prefix_pair,
    predict_transformer_pair,
)
from scripts.decision_aware.run_austria_external_15min import (  # noqa: E402
    V4_ROOTS,
    _forecast_axes,
)
from scripts.decision_aware.summarize_austria_da_day_groups import (  # noqa: E402
    SPIKE_FLOOR_EUR_PER_MWH,
    classify_days,
    forecast_metrics,
)


PUBLIC_TYPES = ("load", "wind", "solar")
PUBLIC_PRODUCTION_TYPES = {
    "load": "load",
    "wind": "wind_onshore",
    "solar": "solar",
}


def _stable_physical_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    stats = robust_context_stats(values)
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return {
        "mean": float(stats["mean"]),
        "std": max(
            float(stats["std"]),
            float(np.std(values)),
            0.20 * float(q95 - q05),
            0.05 * abs(float(stats["mean"])),
            1.0,
        ),
    }


def _load_public_forecasts(directory: Path) -> pd.DataFrame:
    columns = {}
    expected_index = None
    for name in PUBLIC_TYPES:
        path = directory / f"energy_charts_at_{name}_day_ahead_20251001_20260930.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload["production_type"] != PUBLIC_PRODUCTION_TYPES[name]
            or payload["forecast_type"] != "day-ahead"
        ):
            raise ValueError(f"Energy-Charts {name} metadata mismatch")
        index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
        values = np.asarray(payload["forecast_values"], dtype=np.float64)
        if len(index) != 35040 or values.shape != (35040,) or not np.isfinite(values).all():
            raise ValueError(f"Energy-Charts {name} coverage mismatch")
        if expected_index is None:
            expected_index = index
        elif not index.equals(expected_index):
            raise ValueError("Energy-Charts physical forecast axes differ")
        columns[name] = values
    return pd.DataFrame(columns, index=expected_index)


def _local_hour_profile(
    frame: pd.DataFrame,
    delivery_date,
    timezone: str,
) -> np.ndarray:
    local = frame.index.tz_convert(timezone)
    selected = frame.loc[local.date == delivery_date].copy()
    selected["hour"] = selected.index.tz_convert(timezone).hour
    grouped = selected.groupby("hour")[list(PUBLIC_TYPES)].mean()
    # The spring-forward hour has no native settlement rows and is ignored by
    # quarter-hour scoring.  Interpolation merely keeps the fixed 24-query
    # model interface finite; the fall-back repeated hour is averaged.
    grouped = grouped.reindex(range(24)).interpolate(limit_direction="both")
    if grouped.isna().any().any():
        raise ValueError(f"{delivery_date} physical forecast profile incomplete")
    return grouped.to_numpy(np.float32)


def _historical_hour_profile(
    hourly: pd.DataFrame,
    delivery_date,
    timezone: str,
    lag_days: int = 7,
) -> np.ndarray:
    local = hourly.index.tz_convert(timezone)
    selected = hourly.loc[
        local.date == delivery_date - timedelta(days=lag_days), "price_da"
    ]
    frame = pd.DataFrame({"value": selected.to_numpy()}, index=selected.index)
    frame["hour"] = frame.index.tz_convert(timezone).hour
    grouped = frame.groupby("hour")["value"].mean().reindex(range(24))
    grouped = grouped.interpolate(limit_direction="both")
    if grouped.isna().any():
        raise ValueError(f"{delivery_date} seasonal anchor incomplete")
    return grouped.to_numpy(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="configs/decision_aware/austria_external_15min_v1.json"
    )
    parser.add_argument(
        "--checkpoint", default="data/results/pure_v4_multiscale_decoder_v1/decoder.pt"
    )
    parser.add_argument(
        "--prefix-cache", default="data/results/austria_external_15min_v1/forecast_cache"
    )
    parser.add_argument(
        "--public-dir", default="data/Austria/external_public"
    )
    parser.add_argument(
        "--september-apcs", default="data/Austria/Statistik-RZ_202609_NEU.xlsx"
    )
    parser.add_argument(
        "--output-dir", default="data/results/austria_pure_v4_multiscale_v1"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    base, _protocol = AustriaExternalData.load(ROOT, ROOT / args.protocol)
    extended = extend_rt_history(
        base, load_apcs_imbalance(ROOT / args.september_apcs)
    )
    quarter = extended.evaluation_quarters(complete_settlement=False)
    dates, _issues, local = _forecast_axes(extended, quarter)
    if len(dates) != 365 or len(quarter) != 35040:
        raise ValueError("Austria 365-day evaluation axis mismatch")
    prefix_dates, _a, _b = _forecast_axes(
        base, base.evaluation_quarters(complete_settlement=True)
    )
    suffix_dates = dates[len(prefix_dates):]
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    members = []
    cache_dir = ROOT / args.prefix_cache
    for seed in (0, 1, 2):
        prefix = load_prefix_pair(
            cache_dir, f"decision_aware_v4_seed{seed}_member", len(prefix_dates)
        )
        suffix = predict_transformer_pair(
            extended, suffix_dates, V4_ROOTS, seed, device, args.batch_size
        )
        members.append(append_pair(prefix, suffix))

    public = _load_public_forecasts(ROOT / args.public_dir)
    if not public.index.equals(quarter.index):
        raise ValueError("public physical forecasts do not match Austria evaluation axis")
    context_rows = []
    calendar_rows = []
    known_rows = []
    seasonal_rows = []
    aligned_rows = []
    means = []
    stds = []
    for delivery_date in dates:
        sample = extended.day_sample(delivery_date)
        streams = []
        for key in ("price_da_ctx", "price_rt_ctx", "load_ctx", "system_ctx", "cal_ctx"):
            value = sample[key].numpy()
            if value.ndim == 1:
                value = value[:, None]
            streams.append(value)
        context_rows.append(np.concatenate(streams, axis=-1))
        calendar_rows.append(sample["cal_tgt"].numpy())
        means.append(float(sample["price_da_mean"]))
        stds.append(float(sample["price_da_std"]))
        seasonal_rows.append(
            _historical_hour_profile(
                extended.hourly, delivery_date, extended.timezone, lag_days=7
            )
        )
        aligned_rows.append(np.stack([
            (
                _historical_hour_profile(
                    extended.hourly, delivery_date, extended.timezone,
                    lag_days=lag,
                )
                - float(sample["price_da_mean"])
            ) / max(float(sample["price_da_std"]), 1.0e-4)
            for lag in range(7, 0, -1)
        ]))

        issue_local = (
            pd.Timestamp(delivery_date - timedelta(days=1)).tz_localize(extended.timezone)
            + pd.Timedelta(hours=12)
        )
        context_positions = extended._context_positions(issue_local.tz_convert("UTC"))
        history = extended.hourly.iloc[context_positions]
        physical = _local_hour_profile(public, delivery_date, extended.timezone)
        normalized = []
        for column, history_name in enumerate(("load", "wind", "solar")):
            stats = _stable_physical_stats(history[history_name].to_numpy(np.float32))
            normalized.append(normalize(physical[:, column], stats))
        known_rows.append(np.stack(normalized, axis=-1))

    experts = np.stack(
        [members[seed]["da"] for seed in (0, 1, 2)]
        + [np.stack(seasonal_rows)],
        axis=-1,
    ).astype(np.float32)
    context = torch.from_numpy(np.stack(context_rows).astype(np.float32))
    aligned = torch.from_numpy(np.stack(aligned_rows).astype(np.float32))
    calendar = torch.from_numpy(np.stack(calendar_rows).astype(np.float32))
    known = torch.from_numpy(np.stack(known_rows).astype(np.float32))
    experts_tensor = torch.from_numpy(experts)
    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = PureV4MultiscaleDecoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(dates), args.batch_size):
            stop = min(start + args.batch_size, len(dates))
            output = model(
                context[start:stop].to(device),
                aligned[start:stop].to(device),
                calendar[start:stop].to(device),
                torch.zeros(stop - start, 24, 8, device=device),
                experts_tensor[start:stop].to(device),
                mean_tensor[start:stop].to(device),
                std_tensor[start:stop].to(device),
                known_fundamentals=known[start:stop].to(device),
            )
            predictions.append(output["prediction"].cpu().numpy())
    hourly_usd = np.concatenate(predictions).astype(np.float32)
    hourly_eur = hourly_usd.astype(np.float64) / extended.eur_to_usd
    quarter_prediction = expand_daily_hourly_to_quarters(dates, hourly_eur, local)
    truth = quarter["\u65e5\u524d\u4ef7\u683c (EUR/MWh)"].to_numpy(np.float64)
    groups = classify_days(local, truth, spike_floor=SPIKE_FLOOR_EUR_PER_MWH)
    metrics = forecast_metrics(local, truth, quarter_prediction, groups)
    summary = {
        "system": "pure-v4-source-multiscale-physical-gate",
        "protocol": "austria_pure_v4_multiscale_v1",
        "uses_exaa": False,
        "uses_external_price_model_predictions": False,
        "uses_austria_price_labels_for_training_or_selection": False,
        "day_ahead_physical_forecasts": list(PUBLIC_TYPES),
        "source_selected_epoch": int(checkpoint["selected_epoch"]),
        "forecast": metrics,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        delivery_dates=np.asarray([str(item) for item in dates]),
        da_usd_per_mwh=hourly_usd,
        da_eur_per_mwh=hourly_eur.astype(np.float32),
        expert_da_usd_per_mwh=experts,
        expert_names=np.asarray([
            "v4_seed0", "v4_seed1", "v4_seed2", "seasonal_lag168"
        ]),
        # Persist the causal, label-free inputs as an explicit external-test
        # feature cache.  This lets source-only calibrators be evaluated on
        # exactly the same frozen Austria interface without reconstructing or
        # silently changing any covariate.
        context=context.numpy().astype(np.float32),
        calendar=calendar.numpy().astype(np.float32),
        known_fundamentals=known.numpy().astype(np.float32),
        aligned_da_history=aligned.numpy().astype(np.float32),
        price_context_mean_usd_per_mwh=np.asarray(means, dtype=np.float32),
        price_context_std_usd_per_mwh=np.asarray(stds, dtype=np.float32),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
