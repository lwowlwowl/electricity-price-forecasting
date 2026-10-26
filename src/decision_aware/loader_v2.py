"""loader_v2.py — 读四市场统一小时数据（2020-2026，DA+RT 双价，含天气）.

从 data/markets/<MARKET>/<MARKET>_统一小时数据_20200101_20260601.parquet 读取，
返回宽表 DataFrame（每列一个变量），供 DecisionAwareDataset 使用。

统一表特点：
- DA+RT+load+wind+solar+calendar+weather 全在一个 Parquet 文件
- 支持真双结算（返回 DA 价 + RT 价两列）
- 支持真节假日（is_holiday 列）
- 统一表已含天气实际值与+24h预测值（17列）
"""
from __future__ import annotations
import os
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../data/markets"))
_MARKET_TIMEZONES = {
    "CAISO": "America/Los_Angeles",
    "ERCOT": "America/Chicago",
    "NYISO": "America/New_York",
    "PJM": "America/New_York",
}


def market_timezone(market: str) -> str:
    """返回市场本地时区；未知市场立即报错，不静默套用ERCOT。"""
    market = market.upper()
    if market not in _MARKET_TIMEZONES:
        raise ValueError(
            f"不支持的市场 {market!r}；可选：{sorted(_MARKET_TIMEZONES)}"
        )
    return _MARKET_TIMEZONES[market]


_UNIFIED_FILENAME = "ERCOT_统一小时数据_20200101_20260601.parquet"
UNIFIED_PATH = os.path.join(_DATA_ROOT, "ERCOT", _UNIFIED_FILENAME)


def resolve_market_unified_path(market: str) -> str:
    """返回一个市场当前唯一的统一 Parquet 路径。"""
    market = market.upper()
    market_timezone(market)
    filename = f"{market}_统一小时数据_20200101_20260601.parquet"
    path = os.path.abspath(os.path.join(_DATA_ROOT, market, filename))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 {market} 统一小时数据。请确认仓库中的 Parquet 已完整拉取。\n"
            f"预期路径：{path}"
        )
    return path


def resolve_ercot_unified_path() -> str:
    """兼容旧调用：返回 ERCOT 统一 Parquet 路径。"""
    return resolve_market_unified_path("ERCOT")


def _prepare_market_node(raw_df: pd.DataFrame,
                         market: str,
                         node: str,
                         start: str,
                         end: str,
                         dropna: bool,
                         include_weather: bool) -> pd.DataFrame:
    """从已加载的只读市场表构造一个节点宽表。"""
    df = raw_df[raw_df["node"] == node].copy()
    if df.empty:
        raise ValueError(f"{market} 数据中找不到节点 {node!r}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["timestamp_local"] = (
        pd.to_datetime(df["timestamp_local"], utc=True)
        .dt.tz_convert(market_timezone(market))
    )
    df = df.set_index("timestamp_utc").sort_index()
    out = pd.DataFrame({
        "price_da": df["day_ahead_price_usd_mwh"].astype(float),
        "price_rt": df["real_time_price_usd_mwh"].astype(float),
        "load": df["actual_load_mw"].astype(float),
        "wind": df["wind_actual_mw"].astype(float),
        "solar": df["solar_actual_mw"].astype(float),
    })
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
    out["is_weekend"] = df["is_weekend"].astype(float).values
    out["is_holiday"] = df["is_holiday"].astype(float).values
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
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index <= pd.Timestamp(end, tz="UTC")]
    if dropna:
        out = out.dropna()
    out.index.name = "timestamp_utc"
    return out


def load_market_unified(market: str,
                        node: str,
                        start: str = "2020-01-01",
                        end: str = "2026-06-02",
                        dropna: bool = True,
                        include_weather: bool = False) -> pd.DataFrame:
    """读一个市场、一个节点的统一小时数据，返回宽表。

    返回列：
      timestamp_utc (DatetimeIndex), price_da, price_rt, load, wind, solar,
      hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
      is_weekend, is_holiday。日历特征以该市场原始"本地时间"派生。
      若 include_weather=True，额外返回 9 列 actual_* 天气实际值和
      8 列 forecast_* +24h 天气预测值。
    """
    market = market.upper()
    raw_df = pd.read_parquet(resolve_market_unified_path(market))
    return _prepare_market_node(
        raw_df, market, node, start, end, dropna, include_weather
    )


def load_market_unified_nodes(market: str,
                              nodes: list[str] | tuple[str, ...],
                              start: str = "2020-01-01",
                              end: str = "2026-06-02",
                              dropna: bool = True,
                              include_weather: bool = False,
                              read_end_utc: str | None = None,
                              ) -> dict[str, pd.DataFrame]:
    """只读一次市场Parquet，返回多个节点宽表。"""
    market = market.upper()
    if not nodes:
        raise ValueError("nodes不能为空")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes不能包含重复节点")
    # Predicate pushdown keeps locked future labels out of the loaded table.
    read_options = {}
    if read_end_utc is not None:
        cutoff = pd.Timestamp(read_end_utc)
        if cutoff.tzinfo is None:
            raise ValueError("read_end_utc必须包含时区")
        read_options["filters"] = [("timestamp_utc", "<", cutoff.tz_convert("UTC"))]
    raw_df = pd.read_parquet(resolve_market_unified_path(market), **read_options)
    return {
        node: _prepare_market_node(
            raw_df, market, node, start, end, dropna, include_weather
        )
        for node in nodes
    }


def load_ercot_unified(node: str = "LZ_LCRA",
                       start: str = "2020-01-01",
                       end: str = "2026-06-02",
                       dropna: bool = True,
                       include_weather: bool = False) -> pd.DataFrame:
    """兼容旧调用：读取 ERCOT 的一个节点。"""
    return load_market_unified(
        market="ERCOT",
        node=node,
        start=start,
        end=end,
        dropna=dropna,
        include_weather=include_weather,
    )


def np_sin(x, period):
    import numpy as np
    return np.sin(2 * np.pi * x / period)


def np_cos(x, period):
    import numpy as np
    return np.cos(2 * np.pi * x / period)
