"""loader_v2.py — 读 ERCOT 统一小时数据（2020-2026，DA+RT 双价）.

从 data/raw/ERCOT/processed/ercot_unified_hourly_2020_2026.csv 读取，
返回宽表 DataFrame（每列一个变量），供 DecisionAwareDataset 使用。

与 loader.py 的 load_slice_model_ready 区别：
- 读新统一表（DA+RT+load+wind+solar+calendar 全在一个文件）
- 支持真双结算（返回 DA 价 + RT 价两列）
- 支持真节假日（is_holiday 列）
"""
from __future__ import annotations
import os
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW_DIR = os.path.join(_SCRIPT_DIR, "../../data/raw")

UNIFIED_PATH = os.path.join(_RAW_DIR, "ERCOT", "processed",
                            "ercot_unified_hourly_2020_2026.csv")


def load_ercot_unified(node: str = "LZ_LCRA",
                       start: str = "2020-01-01",
                       end: str = "2026-06-02",
                       dropna: bool = True) -> pd.DataFrame:
    """读 ERCOT 统一小时数据，返回宽表。

    返回列：
      timestamp_utc (DatetimeIndex), price_da, price_rt, load, wind, solar,
      hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_holiday
    """
    df = pd.read_csv(UNIFIED_PATH)
    # 筛节点
    df = df[df["节点/地区"] == node].copy()
    # 时间列
    df["timestamp_utc"] = pd.to_datetime(df["UTC时间"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    # 重命名为统一列名
    out = pd.DataFrame({
        "price_da": df["日前价格 (USD/MWh)"].astype(float),
        "price_rt": df["实时价格 (USD/MWh)"].astype(float),
        "load": df["实际负荷 (MW)"].astype(float),
        "wind": df["风电实际 (MW)"].astype(float),
        "solar": df["光伏实际 (MW)"].astype(float),
    })
    # Calendar（用文件自带的节假日 + 派生 sin/cos）
    idx = out.index
    hour = idx.hour
    dow = idx.dayofweek
    out["hour_sin"] = np_sin(hour, 24)
    out["hour_cos"] = np_cos(hour, 24)
    out["dow_sin"] = np_sin(dow, 7)
    out["dow_cos"] = np_cos(dow, 7)
    out["is_weekend"] = (dow >= 5).astype(float)
    # 真节假日（从原表取，对齐索引）
    out["is_holiday"] = df["是否节假日"].astype(float).values
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
