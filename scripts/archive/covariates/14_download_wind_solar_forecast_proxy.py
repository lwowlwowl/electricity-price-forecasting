"""
14_download_wind_solar_forecast_proxy.py
========================================
从 Open-Meteo Historical Forecast API 下载风速和太阳辐射预报，
作为风电/光伏出力预报的代理变量。

数据源：Open-Meteo Historical Forecast API
  - wind_speed_10m: 10m 风速预报（风电代理）
  - shortwave_radiation: 短波辐射预报（光伏代理）
  - 与温度预报相同的坐标点和时间范围

输出：
  data/raw/forecasts/wind_solar/{BA}_wind_solar_forecast_hourly.csv
  列：timestamp_utc, ba_code, wind_speed_forecast_ms, solar_radiation_forecast_wm2
"""
import os
import time
import requests
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'raw', 'forecasts', 'wind_solar')
os.makedirs(OUT_DIR, exist_ok=True)

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

START_DATE = '2025-01-01'
END_DATE = '2026-06-05'
API_BASE = 'https://historical-forecast-api.open-meteo.com/v1/forecast'


def fetch_point_forecast(lat, lon, start_date, end_date):
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'wind_speed_10m,shortwave_radiation',
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
    wind = data['hourly']['wind_speed_10m']
    solar = data['hourly']['shortwave_radiation']
    df = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(times, utc=True),
        'wind_speed_forecast_ms': wind,
        'solar_radiation_forecast_wm2': solar,
    })
    return df


def main():
    print("=" * 60)
    print("下载风速 + 太阳辐射预报（Open-Meteo Historical Forecast API）")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    all_ba_dfs = []

    for ba_code, points in BA_POINTS.items():
        print(f"\n--- {ba_code} ({len(points)} points) ---")
        point_dfs = []
        for point_id, lat, lon in points:
            print(f"  {point_id} ({lat}, {lon})...", end=' ', flush=True)
            try:
                df = fetch_point_forecast(lat, lon, START_DATE, END_DATE)
                df['point_id'] = point_id
                point_dfs.append(df)
                print(f"OK ({len(df)} rows)")
                time.sleep(0.5)
            except Exception as e:
                print(f"FAILED: {e}")

        combined = pd.concat(point_dfs, ignore_index=True)
        avg = combined.groupby('timestamp_utc').agg({
            'wind_speed_forecast_ms': 'mean',
            'solar_radiation_forecast_wm2': 'mean',
        }).reset_index()
        avg['ba_code'] = ba_code
        avg = avg[['timestamp_utc', 'ba_code', 'wind_speed_forecast_ms', 'solar_radiation_forecast_wm2']]

        out_path = os.path.join(OUT_DIR, f'{ba_code}_wind_solar_forecast_hourly.csv')
        avg.to_csv(out_path, index=False)
        print(f"  → {out_path} ({len(avg)} rows)")
        print(f"  风速: {avg['wind_speed_forecast_ms'].min():.1f} ~ {avg['wind_speed_forecast_ms'].max():.1f} m/s")
        print(f"  辐射: {avg['solar_radiation_forecast_wm2'].min():.1f} ~ {avg['solar_radiation_forecast_wm2'].max():.1f} W/m²")
        all_ba_dfs.append(avg)

    all_df = pd.concat(all_ba_dfs, ignore_index=True)
    all_path = os.path.join(OUT_DIR, 'all_markets_wind_solar_forecast_hourly.csv')
    all_df.to_csv(all_path, index=False)
    print(f"\n✓ 全部完成，合并文件: {all_path} ({len(all_df)} rows)")


if __name__ == '__main__':
    main()
