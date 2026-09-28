"""
13_download_ercot_wind_solar_forecast.py
========================================
使用 gridstatus 下载 ERCOT 日前风电和光伏预报。

数据源：ERCOT MIS API (via gridstatus)
  - get_hourly_wind_report: 风电实际出力 + STWPF/WGRPP 预测
  - get_hourly_solar_report: 光伏实际出力 + STPPF/PVGRPP 预测

输出：
  data/raw/forecasts/wind_solar/ERCOT_wind_forecast_hourly.csv
  data/raw/forecasts/wind_solar/ERCOT_solar_forecast_hourly.csv
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'raw', 'forecasts', 'wind_solar')
os.makedirs(OUT_DIR, exist_ok=True)

# 测试期
START_DATE = "2025-01-01"
END_DATE = "2026-06-05"


def download_ercot_wind_forecast():
    """下载 ERCOT 风电预报（逐月分批，避免超时）"""
    import gridstatus
    e = gridstatus.Ercot()

    # 按月分批下载
    start = pd.Timestamp(START_DATE, tz='US/Central')
    end = pd.Timestamp(END_DATE, tz='US/Central')
    all_dfs = []

    current = start
    while current < end:
        month_end = (current + pd.offsets.MonthEnd(1)).normalize()
        if month_end > end:
            month_end = end
        print(f"  Wind: {current.strftime('%Y-%m-%d')} ~ {month_end.strftime('%Y-%m-%d')}...", end=' ', flush=True)
        try:
            df = e.get_hourly_wind_report(
                date=current.strftime('%Y-%m-%d'),
                end=month_end.strftime('%Y-%m-%d'),
                verbose=False,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as ex:
            print(f"FAILED: {ex}")
        current = month_end + pd.Timedelta(days=1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        # 提取预报相关列
        print(f"\n  Columns: {list(combined.columns)}")
        out_path = os.path.join(OUT_DIR, 'ERCOT_wind_forecast_hourly.csv')
        combined.to_csv(out_path, index=False)
        print(f"  → {out_path} ({len(combined)} rows)")
        return combined
    return None


def download_ercot_solar_forecast():
    """下载 ERCOT 光伏预报（逐月分批）"""
    import gridstatus
    e = gridstatus.Ercot()

    start = pd.Timestamp(START_DATE, tz='US/Central')
    end = pd.Timestamp(END_DATE, tz='US/Central')
    all_dfs = []

    current = start
    while current < end:
        month_end = (current + pd.offsets.MonthEnd(1)).normalize()
        if month_end > end:
            month_end = end
        print(f"  Solar: {current.strftime('%Y-%m-%d')} ~ {month_end.strftime('%Y-%m-%d')}...", end=' ', flush=True)
        try:
            df = e.get_hourly_solar_report(
                date=current.strftime('%Y-%m-%d'),
                end=month_end.strftime('%Y-%m-%d'),
                verbose=False,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as ex:
            print(f"FAILED: {ex}")
        current = month_end + pd.Timedelta(days=1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        print(f"\n  Columns: {list(combined.columns)}")
        out_path = os.path.join(OUT_DIR, 'ERCOT_solar_forecast_hourly.csv')
        combined.to_csv(out_path, index=False)
        print(f"  → {out_path} ({len(combined)} rows)")
        return combined
    return None


def main():
    print("=" * 60)
    print("下载 ERCOT 风电 + 光伏预报（gridstatus）")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    print("\n--- 风电 ---")
    wind_df = download_ercot_wind_forecast()

    print("\n--- 光伏 ---")
    solar_df = download_ercot_solar_forecast()

    print("\n✓ ERCOT 风光预报下载完成")


if __name__ == '__main__':
    main()
