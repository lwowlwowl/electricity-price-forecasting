#!/usr/bin/env python3
"""
download_iso_wind_solar.py
==========================
在【非公司管理机】上运行此脚本，下载 ERCOT 和 CAISO 的官方风光 MW 预报。

使用方法：
  1. 确保已安装 gridstatus: pip install gridstatus
  2. 确保 Python >= 3.10（如果 3.9，需要 from __future__ import annotations）
  3. 运行: python download_iso_wind_solar.py
  4. 生成的 CSV 文件丢到项目的 data/raw/forecasts/wind_solar/ 目录下

输出文件：
  ERCOT_wind_forecast_hourly.csv   — 含风电实际出力 + STWPF/WGRPP 预测
  ERCOT_solar_forecast_hourly.csv — 含光伏实际出力 + STPPF/PVGRPP 预测
  CAISO_wind_solar_forecast_hourly.csv — 含风电+光伏 DAM 预测
"""
import os
import sys
import pandas as pd
from datetime import datetime

# 时间范围（覆盖训练上下文 + 测试期）
START_DATE = "2025-01-01"
END_DATE = "2026-06-05"


def download_ercot():
    """下载 ERCOT 风电和光伏的小时报告（含实际值和预报值）"""
    import gridstatus
    e = gridstatus.Ercot()

    # --- 风电 ---
    print(f"\n=== ERCOT 风电预报 ({START_DATE} ~ {END_DATE}) ===")
    wind_dfs = []
    current = pd.Timestamp(START_DATE, tz='US/Central')
    end = pd.Timestamp(END_DATE, tz='US/Central')
    while current < end:
        month_end = min((current + pd.offsets.MonthEnd(1)).normalize(), end)
        date_str = current.strftime('%Y-%m-%d')
        end_str = month_end.strftime('%Y-%m-%d')
        print(f"  {date_str} ~ {end_str}...", end=' ', flush=True)
        try:
            df = e.get_hourly_wind_report(date=date_str, end=end_str, verbose=False)
            if df is not None and len(df) > 0:
                wind_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as ex:
            print(f"FAILED: {ex}")
        current = month_end + pd.Timedelta(days=1)

    if wind_dfs:
        wind = pd.concat(wind_dfs, ignore_index=True).drop_duplicates()
        wind.to_csv("ERCOT_wind_forecast_hourly.csv", index=False)
        print(f"  → ERCOT_wind_forecast_hourly.csv ({len(wind)} rows)")
        print(f"  Columns: {list(wind.columns)}")

    # --- 光伏 ---
    print(f"\n=== ERCOT 光伏预报 ({START_DATE} ~ {END_DATE}) ===")
    solar_dfs = []
    current = pd.Timestamp(START_DATE, tz='US/Central')
    while current < end:
        month_end = min((current + pd.offsets.MonthEnd(1)).normalize(), end)
        date_str = current.strftime('%Y-%m-%d')
        end_str = month_end.strftime('%Y-%m-%d')
        print(f"  {date_str} ~ {end_str}...", end=' ', flush=True)
        try:
            df = e.get_hourly_solar_report(date=date_str, end=end_str, verbose=False)
            if df is not None and len(df) > 0:
                solar_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as ex:
            print(f"FAILED: {ex}")
        current = month_end + pd.Timedelta(days=1)

    if solar_dfs:
        solar = pd.concat(solar_dfs, ignore_index=True).drop_duplicates()
        solar.to_csv("ERCOT_solar_forecast_hourly.csv", index=False)
        print(f"  → ERCOT_solar_forecast_hourly.csv ({len(solar)} rows)")
        print(f"  Columns: {list(solar.columns)}")


def download_caiso():
    """下载 CAISO 日前风电+光伏预报"""
    import gridstatus
    c = gridstatus.CAISO()

    print(f"\n=== CAISO 风电+光伏预报 ({START_DATE} ~ {END_DATE}) ===")
    ren_dfs = []
    current = pd.Timestamp(START_DATE, tz='US/Pacific')
    end = pd.Timestamp(END_DATE, tz='US/Pacific')
    while current < end:
        month_end = min((current + pd.offsets.MonthEnd(1)).normalize(), end)
        date_str = current.strftime('%Y-%m-%d')
        end_str = month_end.strftime('%Y-%m-%d')
        print(f"  {date_str} ~ {end_str}...", end=' ', flush=True)
        try:
            df = c.get_solar_and_wind_forecast_dam(date=date_str, end=end_str, verbose=False)
            if df is not None and len(df) > 0:
                ren_dfs.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("EMPTY")
        except Exception as ex:
            print(f"FAILED: {ex}")
        current = month_end + pd.Timedelta(days=1)

    if ren_dfs:
        ren = pd.concat(ren_dfs, ignore_index=True).drop_duplicates()
        ren.to_csv("CAISO_wind_solar_forecast_hourly.csv", index=False)
        print(f"  → CAISO_wind_solar_forecast_hourly.csv ({len(ren)} rows)")
        print(f"  Columns: {list(ren.columns)}")


if __name__ == '__main__':
    print("=" * 60)
    print("ISO 官方风光 MW 预报下载脚本")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("请在有正常网络访问的机器上运行（非公司管理机）")
    print("=" * 60)

    try:
        download_ercot()
    except Exception as e:
        print(f"\nERCOT 下载失败: {e}")

    try:
        download_caiso()
    except Exception as e:
        print(f"\nCAISO 下载失败: {e}")

    print("\n" + "=" * 60)
    print("完成！请将生成的 CSV 文件复制到项目目录:")
    print("  data/raw/forecasts/wind_solar/")
    print("=" * 60)
