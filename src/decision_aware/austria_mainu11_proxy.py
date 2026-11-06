"""Leakage-safe proxies for the available subset of the teacher's mainU11.

This module intentionally adds no features outside the teacher-provided list.
Five unavailable inputs (four MaxBex series and TTF) are omitted rather than
fabricated.  Target-window proxies use observations from exactly seven days
earlier and are therefore available before the day-ahead gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .austria_external import PRICE_DA, TIMESTAMP


PROXY_COLUMNS: tuple[str, ...] = (
    "at_gen_da_a71_proxy",
    "at_resload_da_proxy",
    "at_solar_da_proxy",
    "at_wind_da_proxy",
    "wx_sw_rad_proxy",
    "wx_wind10_ms_proxy",
)

UNAVAILABLE_MAINU11_COLUMNS: tuple[str, ...] = (
    "maxbex_at_de",
    "maxbex_at_hu",
    "maxbex_cz_at",
    "maxbex_de_at",
    "ttf",
)

LOAD = "实际负荷 (MW)"
WIND = "风电实际 (MW)"
SOLAR = "光伏实际 (MW)"
WEATHER_WIND = "实际10米风速 (m/s)"
WEATHER_RADIATION = "实际短波辐射 (W/m²)"


def build_mainu11_proxy(raw: pd.DataFrame) -> pd.DataFrame:
    required = (TIMESTAMP, PRICE_DA, LOAD, WIND, SOLAR, WEATHER_WIND, WEATHER_RADIATION)
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"奥地利数据缺少mainU11代理源列: {missing}")
    timestamp = pd.DatetimeIndex(pd.to_datetime(raw[TIMESTAMP], utc=True))
    if not timestamp.is_unique or not timestamp.is_monotonic_increasing:
        raise ValueError("奥地利UTC时间必须唯一递增")
    if not (timestamp.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError("奥地利数据不是连续15分钟")
    source = raw.copy()
    source.index = timestamp
    lag = 7 * 24 * 4
    load = pd.to_numeric(source[LOAD], errors="coerce").shift(lag)
    wind = pd.to_numeric(source[WIND], errors="coerce").shift(lag)
    solar = pd.to_numeric(source[SOLAR], errors="coerce").shift(lag)
    output = pd.DataFrame(index=timestamp)
    output["y"] = pd.to_numeric(source[PRICE_DA], errors="coerce")
    # The file has no total-generation DA forecast. Weekly load is retained as
    # the nearest same-scale system-balance proxy and is labelled explicitly.
    output["at_gen_da_a71_proxy"] = load
    output["at_resload_da_proxy"] = load - wind - solar
    output["at_solar_da_proxy"] = solar
    output["at_wind_da_proxy"] = wind
    output["wx_sw_rad_proxy"] = pd.to_numeric(
        source[WEATHER_RADIATION], errors="coerce"
    ).shift(lag)
    output["wx_wind10_ms_proxy"] = pd.to_numeric(
        source[WEATHER_WIND], errors="coerce"
    ).shift(lag)
    output.index.name = "timestamp_utc"
    output = output.dropna()
    if not np.isfinite(output.to_numpy(dtype=np.float64)).all():
        raise ValueError("mainU11代理表包含非有限值")
    return output


def make_tasks(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    context_length: int = 4096,
    horizon: int = 96,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for origin in origins:
        stop = int(frame.index.searchsorted(origin, side="left"))
        if stop < context_length or stop + horizon > len(frame):
            continue
        history = frame.iloc[stop - context_length:stop]
        future = frame.iloc[stop:stop + horizon]
        expected = pd.date_range(origin, periods=horizon, freq="15min", tz="UTC")
        if not future.index.equals(expected):
            raise ValueError(f"{origin}的预测窗口不连续")
        tasks.append({
            "context": history["y"].to_numpy(np.float32),
            "hist_cov": history[list(PROXY_COLUMNS)].to_numpy(np.float32),
            "future_cov": future[list(PROXY_COLUMNS)].to_numpy(np.float32),
            "truth": future["y"].to_numpy(np.float32),
            "index": expected,
        })
    return tasks
