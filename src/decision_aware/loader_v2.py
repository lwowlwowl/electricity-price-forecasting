"""loader_v2.py — 读 ERCOT 统一小时数据（2020-2026，DA+RT 双价，含天气）.

从 data/unified/ERCOT_统一小时数据_20200101_20260601.parquet 读取，
返回宽表 DataFrame（每列一个变量），供 DecisionAwareDataset 使用。

与 loader.py 的 load_slice_model_ready 区别：
- 读新统一表（DA+RT+load+wind+solar+calendar+weather 全在一个文件）
- 支持真双结算（返回 DA 价 + RT 价两列）
- 支持真节假日（is_holiday 列）
- 统一表已含天气实际值与+24h预测值（17列）
"""
from __future__ import annotations
import os
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "../../data")

UNIFIED_PATH = os.path.join(_DATA_DIR, "unified",
                            "ERCOT_统一小时数据_20200101_20260601.parquet")


def load_ercot_unified(node: str = "LZ_LCRA",
                       start: str = "2020-01-01",
                       end: str = "2026-06-02",
                       dropna: bool = True,
                       include_weather: bool = False) -> pd.DataFrame:
    """读 ERCOT 统一小时数据，返回宽表。

    返回列：
      timestamp_utc (DatetimeIndex), price_da, price_rt, load, wind, solar,
      hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
      is_weekend, is_holiday。日历特征以 ERCOT 原始"本地时间"派生。
      若 include_weather=True，额外返回 9 列 actual_* 天气实际值和
      8 列 forecast_* +24h 天气预测值。
    """
    df = pd.read_parquet(UNIFIED_PATH)
    # 筛节点
    df = df[df["node"] == node].copy()
    # 以 UTC 索引保证全表按绝对时间排序
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], utc=True).dt.tz_convert("America/Chicago")
    df = df.set_index("timestamp_utc").sort_index()
    # 重命名为统一列名
    out = pd.DataFrame({
        "price_da": df["day_ahead_price_usd_mwh"].astype(float),
        "price_rt": df["real_time_price_usd_mwh"].astype(float),
        "load": df["actual_load_mw"].astype(float),
        "wind": df["wind_actual_mw"].astype(float),
        "solar": df["solar_actual_mw"].astype(float),
    })
    # Calendar：从 ERCOT 本地时间而不是 UTC 派生，避免日/周周期与市场运行时钟错位。
    local_idx = pd.DatetimeIndex(df["timestamp_local"])
    hour = local_idx.hour
    dow = local_idx.dayofweek
    month = local_idx.month
    out["hour_sin"] = np_sin(hour, 24)
    out["hour_cos"] = np_cos(hour, 24)
    out["dow_sin"] = np_sin(dow, 7)
    out["dow_cos"] = np_cos(dow, 7)
    out["month_sin"] = np_sin(month - 1, 12)
    out["month_cos"] = np_cos(month - 1, 12)
    # 与原表标志交叉一致；保留原始标签避免依赖时区转换边界。
    out["is_weekend"] = df["is_weekend"].astype(float).values
    out["is_holiday"] = df["is_holiday"].astype(float).values
    # 天气列（可选）
    if include_weather:
        weather_cols = [
            "actual_temperature_2m_c", "actual_relative_humidity_2m_pct",
            "actual_dew_point_2m_c", "actual_surface_pressure_hpa",
            "actual_cloud_cover_pct", "actual_precipitation_mm",
            "actual_wind_speed_10m_ms", "actual_wind_gusts_10m_ms",
            "actual_shortwave_radiation_wm2",
            "forecast_temperature_2m_c", "forecast_relative_humidity_2m_pct",
            "forecast_dew_point_2m_c", "forecast_cloud_cover_pct",
            "forecast_precipitation_mm", "forecast_wind_speed_10m_ms",
            "forecast_wind_gust_10m_ms", "forecast_shortwave_radiation_wm2",
        ]
        for col in weather_cols:
            if col in df.columns:
                out[col] = df[col].astype(float).values
    # 时间裁剪
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index <= pd.Timestamp(end, tz="UTC")]
    if dropna:
        out = out.dropna()
    out.index.name = "timestamp_utc"
    return out


def np_sin(x, period):
    import numpy as np
    return np.sin(2 * np.pi * x / period)


def np_cos(x, period):
    import numpy as np
    return np.cos(2 * np.pi * x / period)
