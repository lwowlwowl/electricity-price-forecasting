"""
06_eia_fuel_mix_features.py
===========================
从已有的 EIA generation_by_fuel 数据中提取供给侧燃料结构特征。

这些特征捕捉的是供给侧的"事件"信号——天然气发电占比突变、
可再生能源出力剧变、核电意外停机等，它们直接影响市场供需平衡和电价。

数据源: data/raw/EIA/csv/generation_by_fuel/{BA}.csv (已有)
  - ERCO (ERCOT), PJM, CISO (CAISO), NYIS (NYISO)
  - 小时频，2020-01-01 到 2026-06-01

产出特征 (每市场每小时):
  - gas_share:      天然气发电占总发电的比例 (边际定价燃料)
  - renewable_share: 风+光+水发电占比
  - gas_gen_mwh:    天然气发电量 (MWh)
  - wind_gen_mwh:   风力发电量
  - solar_gen_mwh:  太阳能发电量
  - total_gen_mwh:  总发电量
  - gas_share_diff: gas_share 相比前一小时的变化量
  - renewable_shock: renewable_share 的 24 小时滚动 z-score (>2 表示异常高/低)

产出文件:
  data/covariates/generation_mix/hourly/{market}_fuel_mix_hourly.csv

用法:
  python scripts/covariates/06_eia_fuel_mix_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EIA_DIR = ROOT / "data" / "raw" / "EIA" / "csv" / "generation_by_fuel"
OUT_DIR = ROOT / "data" / "covariates" / "generation_mix" / "hourly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_BA = {
    "ercot": "ERCO",
    "pjm":   "PJM",
    "caiso": "CISO",
    "nyiso": "NYIS",
}

# 只保留 2025-01-01 到 2026-06-02 与电价时间范围一致
START = "2025-01-01"
END = "2026-06-02"


def process_market(market, ba_code):
    path = EIA_DIR / f"{ba_code}.csv"
    if not path.exists():
        print(f"  ⚠ {path} 不存在，跳过")
        return

    print(f"  读取 {ba_code}.csv...")
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"].str.replace("Z", ""), utc=True)
    df = df[(df["timestamp_utc"] >= START) & (df["timestamp_utc"] <= END)]

    # 透视: 每行一个时间戳，列为各燃料类型的发电量
    pivot = df.pivot_table(
        index="timestamp_utc",
        columns="fuel_type",
        values="value",
        aggfunc="sum",
    ).sort_index()

    # 确保必要的列存在，不存在则为0
    for col in ["NG", "WND", "SUN", "WAT", "COL", "NUC", "OIL", "OTH"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # 填充缺失值为0
    pivot = pivot.fillna(0)

    # 计算特征
    result = pd.DataFrame(index=pivot.index)
    result["total_gen_mwh"] = pivot.sum(axis=1)
    result["gas_gen_mwh"] = pivot["NG"]
    result["wind_gen_mwh"] = pivot["WND"]
    result["solar_gen_mwh"] = pivot["SUN"]

    # 比例特征 (避免除以0)
    total_safe = result["total_gen_mwh"].replace(0, np.nan)
    result["gas_share"] = (result["gas_gen_mwh"] / total_safe).fillna(0)
    result["renewable_share"] = (
        (pivot["WND"] + pivot["SUN"] + pivot["WAT"]) / total_safe
    ).fillna(0)

    # 差分特征
    result["gas_share_diff"] = result["gas_share"].diff()

    # 可再生能源异常信号: 24h 滚动 z-score
    rolling_mean = result["renewable_share"].rolling(24, min_periods=6).mean()
    rolling_std = result["renewable_share"].rolling(24, min_periods=6).std()
    result["renewable_shock"] = (
        (result["renewable_share"] - rolling_mean) / rolling_std.replace(0, np.nan)
    ).fillna(0)

    # 保存
    result.index.name = "timestamp_utc"
    out_path = OUT_DIR / f"{market}_fuel_mix_hourly.csv"
    result.to_csv(out_path)
    print(f"  → {out_path} ({len(result)} 行)")
    print(f"    gas_share 均值: {result['gas_share'].mean():.3f}, "
          f"renewable_share 均值: {result['renewable_share'].mean():.3f}")


def main():
    print("=" * 60)
    print("EIA 燃料结构 → 供给侧小时频特征")
    print("=" * 60)

    for market, ba_code in MARKET_BA.items():
        print(f"\n处理 {market.upper()} ({ba_code})...")
        process_market(market, ba_code)

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
