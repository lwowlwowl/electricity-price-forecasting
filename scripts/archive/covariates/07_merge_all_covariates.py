"""
07_merge_all_covariates.py
==========================
将所有协变量 (P0 + P1 + P2) 合并成每个市场的统一小时级文件。

输入:
  P0: henry_hub, natgas_storage           (03_align 产出)
  P1: wti_crude, steel_ppi, steel_prod    (03_align 产出)
  P2-Storm: storm 7 个特征                 (05_download_noaa_storm 产出)
  P2-Fuel:  fuel_mix 8 个特征              (06_eia_fuel_mix_features 产出)

产出:
  data/covariates/merged/{market}_all_covariates_hourly.csv

用法:
  python scripts/covariates/07_merge_all_covariates.py
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = ROOT / "data" / "covariates" / "merged"
STORM_HOURLY = ROOT / "data" / "covariates" / "storm" / "hourly"
FUEL_HOURLY = ROOT / "data" / "covariates" / "generation_mix" / "hourly"

MARKETS = ["ercot", "pjm", "caiso", "nyiso"]


def load_csv(path, index_col="timestamp_utc"):
    """加载 CSV 并统一时间索引为 UTC datetime"""
    if not path.exists():
        print(f"    ⚠ {path.name} 不存在，跳过")
        return None
    df = pd.read_csv(path, parse_dates=[index_col], index_col=index_col)
    # 统一为 UTC-aware 或 UTC-naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    return df


def merge_market(market):
    print(f"\n合并 {market.upper()}...")

    frames = []

    # P0+P1: 已有的合并文件 (03_align 产出)
    p01_path = MERGED_DIR / f"{market}_covariates_hourly.csv"
    p01 = load_csv(p01_path)
    if p01 is not None:
        print(f"  P0+P1: {p01.shape[1]} 列, {len(p01)} 行")
        frames.append(p01)

    # P2-Storm
    storm_path = STORM_HOURLY / f"{market}_storm_hourly.csv"
    storm = load_csv(storm_path)
    if storm is not None:
        print(f"  Storm: {storm.shape[1]} 列, {len(storm)} 行")
        frames.append(storm)

    # P2-Fuel Mix
    fuel_path = FUEL_HOURLY / f"{market}_fuel_mix_hourly.csv"
    fuel = load_csv(fuel_path)
    if fuel is not None:
        print(f"  Fuel:  {fuel.shape[1]} 列, {len(fuel)} 行")
        frames.append(fuel)

    if not frames:
        print("  ⚠ 没有找到任何协变量文件")
        return

    # 统一所有索引为 tz-naive UTC (去掉 timezone info 以便 join)
    unified = []
    for df in frames:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        unified.append(df)

    # 按时间索引 outer join (保留所有时间点)
    merged = unified[0]
    for df in unified[1:]:
        merged = merged.join(df, how="outer", rsuffix="_dup")
        # 删除重复列
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

    merged = merged.sort_index()

    # 保存
    out_path = MERGED_DIR / f"{market}_all_covariates_hourly.csv"
    merged.index.name = "timestamp_utc"
    merged.to_csv(out_path)
    print(f"  → {out_path}")
    print(f"    {merged.shape[1]} 列, {len(merged)} 行")
    print(f"    时间范围: {merged.index.min()} → {merged.index.max()}")
    print(f"    列: {list(merged.columns)}")
    # 缺失率
    missing = merged.isna().mean()
    if missing.any():
        print(f"    缺失率 > 0 的列:")
        for col in missing[missing > 0].index:
            print(f"      {col}: {missing[col]:.1%}")


def main():
    print("=" * 60)
    print("合并所有协变量 (P0+P1+P2) → 统一小时级文件")
    print("=" * 60)

    for market in MARKETS:
        merge_market(market)

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()
