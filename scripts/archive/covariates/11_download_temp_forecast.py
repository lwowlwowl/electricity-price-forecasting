"""
11_download_temp_forecast.py
============================
从 Open-Meteo Historical Forecast API 下载四个市场的【日前温度预报】。

数据源：Open-Meteo Historical Forecast API (https://historical-forecast-api.open-meteo.com)
  - 存档从 2022 年起，覆盖回测测试期 2025-07-01 ~ 2026-06-01
  - 免费、无需 API Key
  - 使用与 ERA5 相同的 BA 坐标点（request_points.csv），每个 BA 3 个点取均值
  - 与现有 weather/processed/by_ba/{BA}_weather_hourly.csv 格式对齐

输出：data/raw/forecasts/temperature/{BA}_temp_forecast_hourly.csv
  列：timestamp_utc, ba_code, temperature_forecast_c, source_points
"""
import os
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'raw', 'forecasts', 'temperature')
os.makedirs(OUT_DIR, exist_ok=True)

# 与现有 weather 数据相同的 BA → 坐标点映射（来自 request_points.csv）
BA_POINTS = {
    'ERCO': [
        ('ERCO_P01', 31.2268, -98.76126),
        ('ERCO_P02', 25.98387, -97.22144),
        ('ERCO_P03', 34.92231, -101.71967),
    ],
    'PJM': [
        ('PJM_P01', 39.24158, -80.15963),
        ('PJM_P02', 42.13, -89.81223),
        ('PJM_P03', 40.42068, -73.99294),
    ],
    'CISO': [
        ('CISO_P01', 36.92412, -119.50531),
        ('CISO_P02', 40.81255, -124.17419),
        ('CISO_P03', 33.52657, -115.27327),
    ],
    'NYIS': [
        ('NYIS_P01', 42.98733, -75.5762),
        ('NYIS_P02', 42.36041, -79.66425),
        ('NYIS_P03', 41.28292, -71.9351),
    ],
}

# 下载时间范围：覆盖训练上下文期 + 测试期
START_DATE = '2025-01-01'
END_DATE = '2026-06-05'

API_BASE = 'https://historical-forecast-api.open-meteo.com/v1/forecast'


def fetch_point_forecast(lat, lon, start_date, end_date):
    """从 Open-Meteo Historical Forecast API 拉取单点温度预报。"""
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m',
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'UTC',
        'format': 'json',
        'model': 'best_match',
    }
    r = requests.get(API_BASE, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    times = data['hourly']['time']
    temps = data['hourly']['temperature_2m']
    df = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(times, utc=True),
        'temperature_forecast_c': temps,
    })
    return df


def main():
    print("=" * 60)
    print("下载日前温度预报（Open-Meteo Historical Forecast API）")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    all_ba_dfs = []

    for ba_code, points in BA_POINTS.items():
        print(f"\n--- {ba_code} ({len(points)} points) ---")
        point_dfs = []
        for point_id, lat, lon in points:
            print(f"  {point_id} ({lat}, {lon})...", end=' ')
            try:
                df = fetch_point_forecast(lat, lon, START_DATE, END_DATE)
                df['point_id'] = point_id
                point_dfs.append(df)
                print(f"OK ({len(df)} rows)")
                time.sleep(0.5)  # 礼貌限速
            except Exception as e:
                print(f"FAILED: {e}")
                # 单点失败时用 NaN 填充
                ts = pd.date_range(start=START_DATE, end=END_DATE, freq='1h', tz='UTC')
                df = pd.DataFrame({
                    'timestamp_utc': ts,
                    'temperature_forecast_c': [None] * len(ts),
                    'point_id': point_id,
                })
                point_dfs.append(df)

        # 合并多点，取均值（与现有 weather 数据处理方式一致）
        combined = pd.concat(point_dfs, ignore_index=True)
        avg = combined.groupby('timestamp_utc')['temperature_forecast_c'].mean().reset_index()
        avg['ba_code'] = ba_code
        avg['source_points'] = len(points)
        avg = avg[['timestamp_utc', 'ba_code', 'temperature_forecast_c', 'source_points']]

        out_path = os.path.join(OUT_DIR, f'{ba_code}_temp_forecast_hourly.csv')
        avg.to_csv(out_path, index=False)
        print(f"  → {out_path} ({len(avg)} rows)")
        print(f"  温度范围: {avg['temperature_forecast_c'].min():.1f}°C ~ {avg['temperature_forecast_c'].max():.1f}°C")
        all_ba_dfs.append(avg)

    # 合并所有市场
    all_df = pd.concat(all_ba_dfs, ignore_index=True)
    all_path = os.path.join(OUT_DIR, 'all_markets_temp_forecast_hourly.csv')
    all_df.to_csv(all_path, index=False)
    print(f"\n✓ 全部完成，合并文件: {all_path} ({len(all_df)} rows)")


if __name__ == '__main__':
    main()
