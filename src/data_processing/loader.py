"""
统一数据加载器 load_slice()
============================
把 data/raw/{市场}/processed/ 下的"长表"市场数据，按需切片成一张
对齐好的"宽表"DataFrame，供所有实验脚本统一使用。

核心理念：永远只取一次实验需要的那一小片对齐数据，而不是把大文件全读进内存。

数据结构说明（阶段0 勘探结论）：
  - 原始文件是【长表】：timestamp_utc, location, variable, value, ...
    同一时刻的不同节点堆叠成多行。
  - 电价(price)有 15 个节点：HB_*（结算枢纽）、LZ_*（负荷区）。
  - 负荷(load)、风电(wind)、光伏(solar) 只有 1 个 SYSTEM 聚合节点
    → 协变量是"系统级"的，对所有电价节点是同一条序列。
  - 天气在 weather/processed/by_ba/{BA_CODE}_weather_hourly.csv，宽表格式，
    每个变量一列；ERCOT 对应 ERCO，CAISO 对应 CISO。
  - 所有文件用 timestamp_utc（ISO8601, UTC）作对齐主轴。

用法示例：
    from loader import load_slice
    df = load_slice(
        market="ERCOT",
        nodes=["HB_NORTH", "HB_HOUSTON"],
        freq="1h",
        covariates=["load", "wind", "solar", "temperature"],
        start="2025-06-01", end="2025-09-01",
    )
    # df 列：price__HB_NORTH, price__HB_HOUSTON, load, wind, solar, temperature
    # df 索引：timestamp_utc（DatetimeIndex, UTC）
"""

import os
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(SCRIPT_DIR, "../../data/raw")

# 市场 → 天气文件的 BA 代码映射
MARKET_TO_BA = {
    "ERCOT": "ERCO",
    "CAISO": "CISO",
    "NYISO": "NYIS",
    "PJM":   "PJM",
}

# 频率别名归一化（接受多种写法）
FREQ_ALIAS = {
    "1h": "hourly", "h": "hourly", "hourly": "hourly", "60min": "hourly",
    "15min": "15min", "15m": "15min",
    "5min": "5min", "5m": "5min",
}

# 市场变量 → 文件名前缀
VAR_TO_PREFIX = {
    "price": "actual_price",
    "load":  "actual_load",
    "wind":  "wind_actual",
    "solar": "solar_actual",
}

# 天气变量名 → 天气文件中的实际列名
WEATHER_COL = {
    "temperature":   "temperature_2m_c",
    "humidity":      "relative_humidity_2m_pct",
    "dew_point":     "dew_point_2m_c",
    "pressure":      "surface_pressure_hpa",
    "cloud_cover":   "cloud_cover_pct",
    "precipitation": "precipitation_mm",
    "wind_speed":    "wind_speed_10m_ms",
    "radiation":     "shortwave_radiation_wm2",
}
WEATHER_VARS = set(WEATHER_COL.keys())
MARKET_COVARS = {"load", "wind", "solar"}   # 来自市场目录的系统级协变量


def _norm_freq(freq: str) -> str:
    f = FREQ_ALIAS.get(str(freq).lower())
    if f is None:
        raise ValueError(f"不支持的频率 {freq!r}，可选：1h / 15min / 5min")
    return f


def _read_long(market: str, variable: str, freq: str) -> pd.DataFrame:
    """读取一个市场长表文件，返回 [timestamp_utc, location, value]。"""
    prefix = VAR_TO_PREFIX[variable]
    fname  = f"{prefix}_{freq}.csv"
    path   = os.path.join(RAW_DIR, market, "processed", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}（该市场可能没有 {variable} 的 {freq} 频率）")
    df = pd.read_csv(path, usecols=["timestamp_utc", "location", "value"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _pivot_price(market: str, freq: str, nodes) -> pd.DataFrame:
    """读电价长表并透视成宽表，列名形如 price__HB_NORTH。"""
    long_df = _read_long(market, "price", freq)
    all_nodes = sorted(long_df["location"].unique())
    if nodes:
        missing = [n for n in nodes if n not in all_nodes]
        if missing:
            raise ValueError(f"节点不存在：{missing}\n可用节点：{all_nodes}")
        long_df = long_df[long_df["location"].isin(nodes)]
    wide = long_df.pivot_table(index="timestamp_utc", columns="location", values="value")
    wide.columns = [f"price__{c}" for c in wide.columns]
    return wide


def _system_series(market: str, variable: str, freq: str) -> pd.Series:
    """读取系统级协变量（load/wind/solar），返回以 timestamp_utc 为索引的 Series。"""
    long_df = _read_long(market, variable, freq)
    # 系统级只有一个 location（SYSTEM）；若有多个则取均值兜底
    s = long_df.groupby("timestamp_utc")["value"].mean()
    s.name = variable
    return s


def _weather_series(market: str, vars_wanted, freq: str) -> pd.DataFrame:
    """读取天气文件中需要的变量列；天气原生为 1h，需要更高频时前向填充。"""
    ba = MARKET_TO_BA.get(market)
    if ba is None:
        raise ValueError(f"未知市场 {market!r}，无对应天气 BA 代码")
    path = os.path.join(RAW_DIR, "weather", "processed", "by_ba", f"{ba}_weather_hourly.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到天气文件：{path}")
    cols = [WEATHER_COL[v] for v in vars_wanted]
    df = pd.read_csv(path, usecols=["timestamp_utc"] + cols)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    df.columns = list(vars_wanted)   # 重命名为用户友好的变量名
    return df


def load_slice(
    market: str,
    nodes=None,
    freq: str = "1h",
    covariates=None,
    start=None,
    end=None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    取出一片对齐好的电价 + 协变量数据。

    参数
    ----
    market     : "ERCOT" / "CAISO" / "NYISO" / "PJM"
    nodes      : 电价节点列表，如 ["HB_NORTH"]；None 表示全部节点
    freq       : "1h" / "15min" / "5min"
    covariates : 协变量列表，可包含：
                   市场系统级： load / wind / solar
                   天气：       temperature / humidity / dew_point / pressure /
                                cloud_cover / precipitation / wind_speed / radiation
                 None 或 [] 表示不要协变量
    start, end : 时间窗口（字符串或 Timestamp，按 UTC 解析）；None 表示不裁剪
    dropna     : 是否丢弃含缺失值的行（默认 True，保证下游模型可直接用）

    返回
    ----
    DataFrame，索引为 timestamp_utc（UTC DatetimeIndex），
    列为 price__<节点> 若干列 + 各协变量列。
    """
    freq = _norm_freq(freq)
    covariates = covariates or []

    # 1) 电价（目标）
    out = _pivot_price(market, freq, nodes)

    # 2) 协变量
    for cov in covariates:
        if cov in MARKET_COVARS:
            s = _system_series(market, cov, freq)
            out = out.join(s, how="left")
        elif cov in WEATHER_VARS:
            w = _weather_series(market, [cov], freq)
            # 天气是 1h；若目标频率更高，按时间索引前向填充对齐
            out = out.join(w, how="left")
            if freq != "hourly":
                out[cov] = out[cov].ffill()
        else:
            raise ValueError(
                f"未知协变量 {cov!r}。市场级可选 {sorted(MARKET_COVARS)}；"
                f"天气可选 {sorted(WEATHER_VARS)}"
            )

    # 3) 时间窗口裁剪
    out = out.sort_index()
    if start is not None:
        out = out[out.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        out = out[out.index <= pd.to_datetime(end, utc=True)]

    # 4) 缺失处理
    if dropna:
        out = out.dropna()

    out.index.name = "timestamp_utc"
    return out


# ── 自测：直接运行本文件时做一次冒烟测试 ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("load_slice 冒烟测试 —— ERCOT")
    print("=" * 70)

    df = load_slice(
        market="ERCOT",
        nodes=["HB_NORTH", "HB_HOUSTON"],
        freq="1h",
        covariates=["load", "wind", "solar", "temperature"],
        start="2025-06-01", end="2025-09-01",
    )
    print(f"\n形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")
    print(f"时间范围：{df.index[0]}  →  {df.index[-1]}")
    print(f"缺失值总数：{int(df.isna().sum().sum())}")
    print("\n前 3 行：")
    print(df.head(3).to_string())
    print("\n统计描述：")
    print(df.describe().round(2).to_string())
    print("\n✅ load_slice 工作正常")
