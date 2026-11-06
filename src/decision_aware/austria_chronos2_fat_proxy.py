"""Teacher-family-only Austria proxies for Chronos2-ZS-fat8192."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .austria_external import PRICE_DA, TIMESTAMP


KNOWN_FUTURE_COLUMNS: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow",
    "is_weekend",
    "month_sin",
    "month_cos",
    "is_holiday",
    "at_load_da_proxy",
    "at_solar_da_proxy",
    "at_resload_da_proxy",
    "at_gen_da_a71_proxy",
    "wx_temp_c_proxy",
    "wx_wind10_ms_proxy",
    "wx_sw_rad_proxy",
)

PAST_ONLY_COLUMNS: tuple[str, ...] = (
    "gen_solar_lag24",
    "gen_resload_lag24",
    "gen_renewable_share_lag24",
)

UNAVAILABLE_TEACHER_FEATURES: tuple[str, ...] = (
    "wx_cloud_pct",
    "ttf",
    "nine unavailable gen_*_lag24 series",
)

LOAD = "实际负荷 (MW)"
WIND = "风电实际 (MW)"
SOLAR = "光伏实际 (MW)"
TEMPERATURE = "实际气温 (°C)"
WEATHER_WIND = "实际10米风速 (m/s)"
WEATHER_RADIATION = "实际短波辐射 (W/m²)"
HOLIDAY = "是否节假日"


def build_chronos2_fat_proxy(raw: pd.DataFrame) -> pd.DataFrame:
    required = (
        TIMESTAMP,
        PRICE_DA,
        LOAD,
        WIND,
        SOLAR,
        TEMPERATURE,
        WEATHER_WIND,
        WEATHER_RADIATION,
        HOLIDAY,
    )
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"奥地利数据缺少Chronos2代理源列: {missing}")
    timestamp = pd.DatetimeIndex(pd.to_datetime(raw[TIMESTAMP], utc=True))
    if not timestamp.is_unique or not timestamp.is_monotonic_increasing:
        raise ValueError("奥地利UTC时间必须唯一递增")
    if not (timestamp.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError("奥地利数据不是连续15分钟")

    source = raw.copy()
    source.index = timestamp
    local = timestamp.tz_convert("Europe/Vienna")
    lag_week = 7 * 24 * 4
    lag_day = 24 * 4
    load = pd.to_numeric(source[LOAD], errors="coerce")
    wind = pd.to_numeric(source[WIND], errors="coerce")
    solar = pd.to_numeric(source[SOLAR], errors="coerce")
    resload = load - wind - solar

    output = pd.DataFrame(index=timestamp)
    output["y"] = pd.to_numeric(source[PRICE_DA], errors="coerce")
    hour = local.hour.to_numpy(dtype=np.float64) + local.minute.to_numpy(dtype=np.float64) / 60.0
    month = local.month.to_numpy(dtype=np.float64)
    output["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    output["dow"] = local.dayofweek.to_numpy(dtype=np.float64)
    output["is_weekend"] = (local.dayofweek >= 5).astype(np.float64)
    output["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    output["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    # The raw holiday flag was attached by UTC date.  Taking the local-day
    # majority preserves the supplied holiday calendar while fixing boundary rows.
    holiday = pd.Series(pd.to_numeric(source[HOLIDAY], errors="coerce").to_numpy(), index=timestamp)
    local_dates = pd.Index(local.date)
    daily_holiday = holiday.groupby(local_dates).transform(lambda values: float(values.mean() >= 0.5))
    output["is_holiday"] = daily_holiday.to_numpy(np.float64)

    output["at_load_da_proxy"] = load.shift(lag_week)
    output["at_solar_da_proxy"] = solar.shift(lag_week)
    output["at_resload_da_proxy"] = resload.shift(lag_week)
    output["at_gen_da_a71_proxy"] = load.shift(lag_week)
    output["wx_temp_c_proxy"] = pd.to_numeric(source[TEMPERATURE], errors="coerce").shift(lag_week)
    output["wx_wind10_ms_proxy"] = pd.to_numeric(source[WEATHER_WIND], errors="coerce").shift(lag_week)
    output["wx_sw_rad_proxy"] = pd.to_numeric(source[WEATHER_RADIATION], errors="coerce").shift(lag_week)

    output["gen_solar_lag24"] = solar.shift(lag_day)
    output["gen_resload_lag24"] = resload.shift(lag_day)
    output["gen_renewable_share_lag24"] = ((wind + solar) / load.replace(0.0, np.nan)).shift(lag_day)
    output.index.name = "timestamp_utc"
    output = output.dropna()
    if not np.isfinite(output.to_numpy(np.float64)).all():
        raise ValueError("Chronos2代理表包含非有限值")
    return output


def make_chronos2_tasks(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    max_context_length: int = 8192,
    horizon: int = 96,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for origin in origins:
        stop = int(frame.index.searchsorted(origin, side="left"))
        if stop <= 0 or stop + horizon > len(frame):
            continue
        history = frame.iloc[max(0, stop - max_context_length):stop]
        future = frame.iloc[stop:stop + horizon]
        expected = pd.date_range(origin, periods=horizon, freq="15min", tz="UTC")
        if not future.index.equals(expected):
            raise ValueError(f"{origin}的预测窗口不连续")
        past_covariates = {
            name: history[name].to_numpy(np.float32)
            for name in (*KNOWN_FUTURE_COLUMNS, *PAST_ONLY_COLUMNS)
        }
        future_covariates = {
            name: future[name].to_numpy(np.float32) for name in KNOWN_FUTURE_COLUMNS
        }
        tasks.append(
            {
                "target": history["y"].to_numpy(np.float32),
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "truth": future["y"].to_numpy(np.float32),
                "index": expected,
            }
        )
    return tasks

