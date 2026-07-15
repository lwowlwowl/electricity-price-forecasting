"""
15_build_forecast_model_ready.py
=================================
将下载的预报数据对齐到 model_ready 格式，生成"公正版"协变量文件。

策略：
  - load          → EIA 日前负荷预报
  - temperature   → Open-Meteo 日前温度预报
  - wind (ERCOT/CAISO) → Open-Meteo 风速预报（m/s，代理变量）
  - solar (ERCOT/CAISO) → Open-Meteo 太阳辐射预报（W/m²，代理变量）
  - 经济类协变量保持原样（shift(24) 不变）
  - PJM/NYISO 无 wind/solar 列，不处理

输出：data/covariates/model_ready/{market}_features_forecast_hourly.csv
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_READY_DIR = os.path.join(ROOT, 'data', 'covariates', 'model_ready')
FORECAST_RAW_DIR = os.path.join(ROOT, 'data', 'raw', 'forecasts')
OUT_DIR = os.path.join(ROOT, 'data', 'covariates', 'model_ready')

# 市场映射
MARKETS = {
    'ercot': {'ba_eia': 'ERCO', 'ba_om': 'ERCO'},
    'pjm':   {'ba_eia': 'PJM',  'ba_om': 'PJM'},
    'caiso': {'ba_eia': 'CISO', 'ba_om': 'CISO'},
    'nyiso': {'ba_eia': 'NYIS', 'ba_om': 'NYIS'},
}


def load_eia_load_forecast(ba_code):
    """加载 EIA 负荷预报，返回 timestamp_utc → load_forecast_mw 的 Series"""
    path = os.path.join(FORECAST_RAW_DIR, 'load', f'{ba_code}_load_forecast_hourly.csv')
    df = pd.read_csv(path, parse_dates=['timestamp_utc'])
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.dropna(subset=['load_forecast_mw'])
    df = df.drop_duplicates(subset=['timestamp_utc'], keep='last')
    df = df.set_index('timestamp_utc')['load_forecast_mw']
    return df


def load_om_temp_forecast(ba_code):
    """加载 Open-Meteo 温度预报"""
    path = os.path.join(FORECAST_RAW_DIR, 'temperature', f'{ba_code}_temp_forecast_hourly.csv')
    df = pd.read_csv(path, parse_dates=['timestamp_utc'])
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.dropna(subset=['temperature_forecast_c'])
    df = df.drop_duplicates(subset=['timestamp_utc'], keep='last')
    df = df.set_index('timestamp_utc')['temperature_forecast_c']
    return df


def load_om_wind_solar_forecast(ba_code):
    """加载 Open-Meteo 风速和太阳辐射预报"""
    path = os.path.join(FORECAST_RAW_DIR, 'wind_solar', f'{ba_code}_wind_solar_forecast_hourly.csv')
    df = pd.read_csv(path, parse_dates=['timestamp_utc'])
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.dropna(subset=['wind_speed_forecast_ms', 'solar_radiation_forecast_wm2'])
    df = df.drop_duplicates(subset=['timestamp_utc'], keep='last')
    df = df.set_index('timestamp_utc')
    return df['wind_speed_forecast_ms'], df['solar_radiation_forecast_wm2']


def main():
    print("=" * 60)
    print("生成预报版 model_ready 协变量文件")
    print("=" * 60)

    for market, codes in MARKETS.items():
        print(f"\n--- {market.upper()} ---")
        
        # 加载现有 model_ready 文件
        existing_path = os.path.join(MODEL_READY_DIR, f'{market}_features_hourly.csv')
        df = pd.read_csv(existing_path)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.set_index('timestamp_utc')
        print(f"  原始: {len(df)} rows, {df.index.min()} ~ {df.index.max()}")

        # 加载预报数据
        eia_load = load_eia_load_forecast(codes['ba_eia'])
        om_temp = load_om_temp_forecast(codes['ba_om'])
        
        coverage = {}

        # 替换 load
        if 'load' in df.columns:
            common_idx = df.index.intersection(eia_load.index)
            df.loc[common_idx, 'load'] = eia_load.loc[common_idx].values
            coverage['load'] = f"{len(common_idx)}/{len(df)} ({100*len(common_idx)/len(df):.1f}%)"
            print(f"  load: {coverage['load']}")

        # 替换 temperature
        if 'temperature' in df.columns:
            common_idx = df.index.intersection(om_temp.index)
            df.loc[common_idx, 'temperature'] = om_temp.loc[common_idx].values
            coverage['temperature'] = f"{len(common_idx)}/{len(df)} ({100*len(common_idx)/len(df):.1f}%)"
            print(f"  temperature: {coverage['temperature']}")

        # 替换 wind（ERCOT/CAISO only）
        if 'wind' in df.columns:
            try:
                wind_fc, _ = load_om_wind_solar_forecast(codes['ba_om'])
                common_idx = df.index.intersection(wind_fc.index)
                df.loc[common_idx, 'wind'] = wind_fc.loc[common_idx].values
                coverage['wind'] = f"{len(common_idx)}/{len(df)} ({100*len(common_idx)/len(df):.1f}%)"
                print(f"  wind (风速代理): {coverage['wind']}")
            except Exception as e:
                print(f"  wind: SKIPPED ({e})")

        # 替换 solar（ERCOT/CAISO only）
        if 'solar' in df.columns:
            try:
                _, solar_fc = load_om_wind_solar_forecast(codes['ba_om'])
                common_idx = df.index.intersection(solar_fc.index)
                df.loc[common_idx, 'solar'] = solar_fc.loc[common_idx].values
                coverage['solar'] = f"{len(common_idx)}/{len(df)} ({100*len(common_idx)/len(df):.1f}%)"
                print(f"  solar (辐射代理): {coverage['solar']}")
            except Exception as e:
                print(f"  solar: SKIPPED ({e})")

        # 输出
        df = df.reset_index()
        out_path = os.path.join(OUT_DIR, f'{market}_features_forecast_hourly.csv')
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}")

    # 生成说明文件
    readme = """# 预报版 model_ready 协变量文件

## 数据来源
- **load**: EIA Grid Monitor Day-Ahead Demand Forecast (https://www.eia.gov/electricity/gridmonitor/)
- **temperature**: Open-Meteo Historical Forecast API (https://historical-forecast-api.open-meteo.com)
- **wind** (ERCOT/CAISO): Open-Meteo Historical Forecast wind_speed_10m (m/s，作为风电出力代理)
- **solar** (ERCOT/CAISO): Open-Meteo Historical Forecast shortwave_radiation (W/m²，作为光伏出力代理)

## 说明
- 经济类协变量（gas/oil/steel/storage/storm/generation_mix）保持 shift(24) 不变
- wind/solar 使用风速和辐射预报作为代理变量（单位不同但 Chronos2 自动归一化）
- 时间范围：2025-01-01 ~ 2026-06-05（覆盖训练上下文 + 测试期）
- 所有预报数据均为"日前预报"，即在 T-1 时刻即可获取
"""
    readme_path = os.path.join(OUT_DIR, 'FORECAST_README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"\n  说明文件: {readme_path}")
    print("\n✓ 全部完成")


if __name__ == '__main__':
    main()
