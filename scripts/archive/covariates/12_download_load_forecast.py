"""
12_download_load_forecast.py
=============================
从 EIA Grid Monitor 下载四个市场的【日前负荷预测】。

数据源：EIA-930 Grid Monitor Excel (https://www.eia.gov/electricity/gridmonitor/knownissues/xls/{BA}.xlsx)
  - 每个 BA 的 Excel 包含 "Published Hourly Data" sheet
  - 含 Demand Forecast 列（日前需求预测）
  - 免费、公开，直接下载

输出：data/raw/forecasts/load/{BA}_load_forecast_hourly.csv
  列：timestamp_utc, ba_code, load_forecast_mw
"""
import os
import sys
import pandas as pd
from io import BytesIO
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'raw', 'forecasts', 'load')
os.makedirs(OUT_DIR, exist_ok=True)

# EIA Grid Monitor BA 代码 → 市场
BA_MAP = {
    'ERCO': 'ERCOT',
    'PJM':  'PJM',
    'CISO': 'CAISO',
    'NYIS': 'NYISO',
}

BASE_URL = 'https://www.eia.gov/electricity/gridmonitor/knownissues/xls/{ba}.xlsx'


def download_ba_forecast(ba_code):
    """下载单个 BA 的 EIA Grid Monitor Excel，提取 Demand Forecast。"""
    url = BASE_URL.format(ba=ba_code)
    print(f"  下载 {url}...", end=' ')
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    xls = pd.ExcelFile(BytesIO(r.content))
    # EIA Grid Monitor 的 sheet 名通常是 "Published Hourly Data"
    sheet_name = None
    for sn in xls.sheet_names:
        if 'Published Hourly' in sn or 'hourly' in sn.lower():
            sheet_name = sn
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
        print(f"(sheet: {sheet_name})", end=' ')

    df = pd.read_excel(BytesIO(r.content), sheet_name=sheet_name)

    # 找到时间列和 forecast 列
    # EIA 格式：列名可能是 "UTC time" 或 "UTC Time" 或 "date"
    time_col = None
    for col in df.columns:
        if 'utc' in str(col).lower() and 'time' in str(col).lower():
            time_col = col
            break
    if time_col is None:
        # 尝试第一列
        time_col = df.columns[0]

    # 找 Demand Forecast 列
    forecast_col = None
    for col in df.columns:
        if 'demand' in str(col).lower() and 'forecast' in str(col).lower():
            forecast_col = col
            break
    if forecast_col is None:
        # 打印所有列名供调试
        print(f"\n  ⚠ 未找到 Demand Forecast 列，可用列: {list(df.columns)}")
        return None

    out = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(df[time_col], utc=True),
        'ba_code': ba_code,
        'load_forecast_mw': pd.to_numeric(df[forecast_col], errors='coerce'),
    })
    out = out.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc').reset_index(drop=True)
    print(f"OK ({len(out)} rows, {out['timestamp_utc'].min()} ~ {out['timestamp_utc'].max()})")
    return out


def main():
    print("=" * 60)
    print("下载日前负荷预测（EIA Grid Monitor Excel）")
    print("=" * 60)

    all_dfs = []
    for ba_code, market in BA_MAP.items():
        print(f"\n--- {ba_code} ({market}) ---")
        df = download_ba_forecast(ba_code)
        if df is not None:
            out_path = os.path.join(OUT_DIR, f'{ba_code}_load_forecast_hourly.csv')
            df.to_csv(out_path, index=False)
            print(f"  → {out_path}")
            all_dfs.append(df)

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_path = os.path.join(OUT_DIR, 'all_markets_load_forecast_hourly.csv')
        all_df.to_csv(all_path, index=False)
        print(f"\n✓ 全部完成，合并文件: {all_path} ({len(all_df)} rows)")


if __name__ == '__main__':
    main()
