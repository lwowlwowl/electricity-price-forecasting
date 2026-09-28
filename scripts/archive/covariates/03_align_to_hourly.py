"""
将所有协变量（P0+P1）对齐到四个市场的小时频率电价时间轴。

对齐方法：
- 日频数据（Henry Hub, WTI）：forward-fill 填充周末/节假日空值，再展开到小时
- 周频数据（天然气库存）：forward-fill 到每日，再展开到小时
- 月频数据（钢铁PPI, 钢铁生产指数）：forward-fill 到每日，再展开到小时

输出：每个市场一个 {market}_covariates_hourly.csv，包含所有协变量列
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GAS_DIR = os.path.join(ROOT, 'data', 'covariates', 'gas', 'cleaned')
OIL_DIR = os.path.join(ROOT, 'data', 'covariates', 'oil', 'cleaned')
STEEL_DIR = os.path.join(ROOT, 'data', 'covariates', 'steel', 'cleaned')
MERGED_DIR = os.path.join(ROOT, 'data', 'covariates', 'merged')
DATA_DIR = os.path.join(ROOT, 'data', 'raw')
for d in (GAS_DIR, OIL_DIR, STEEL_DIR, MERGED_DIR):
    os.makedirs(d, exist_ok=True)

# 四个市场的电价文件和代表性节点
MARKETS = {
    'ERCOT': {
        'price_file': os.path.join(DATA_DIR, 'ERCOT', 'processed', 'actual_price_hourly.csv'),
        'location': 'HB_HUBAVG',
    },
    'PJM': {
        'price_file': os.path.join(DATA_DIR, 'PJM', 'processed', 'actual_price_hourly.csv'),
        'location': 'HUB:WESTERN HUB',
    },
    'CAISO': {
        'price_file': os.path.join(DATA_DIR, 'CAISO', 'processed', 'actual_price_hourly.csv'),
        'location': 'TH_SP15_GEN-APND',
    },
    'NYISO': {
        'price_file': os.path.join(DATA_DIR, 'NYISO', 'processed', 'actual_price_hourly.csv'),
        'location': 'CENTRL',
    },
}


def load_and_ffill_to_daily(csv_path, date_col, value_col, daily_range):
    """加载时序数据，reindex 到日频并 forward-fill"""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.set_index(date_col)
    return df[value_col].reindex(daily_range).ffill().bfill()


# ═══ 加载 P0 + P1 协变量源数据 ═══
cov_sources = {
    'henry_hub_usd_per_mmbtu': {
        'file': os.path.join(GAS_DIR, 'henry_hub_daily.csv'),
        'date_col': 'date', 'value_col': 'henry_hub_usd_per_mmbtu',
    },
    'natgas_storage_bcf': {
        'file': os.path.join(GAS_DIR, 'natgas_storage_weekly.csv'),
        'date_col': 'date', 'value_col': 'natgas_storage_bcf',
    },
    'wti_usd_per_barrel': {
        'file': os.path.join(OIL_DIR, 'wti_crude_daily.csv'),
        'date_col': 'date', 'value_col': 'wti_usd_per_barrel',
    },
    'hrc_futures_usd_per_ton': {
        'file': os.path.join(STEEL_DIR, 'hrc_futures_daily.csv'),
        'date_col': 'date', 'value_col': 'hrc_futures_usd_per_ton',
    },
}

# ═══ 对齐到每个市场的小时时间轴 ═══
print("=== Aligning all covariates to hourly ===\n")

for mkt_name, mkt_cfg in MARKETS.items():
    print(f"  Processing {mkt_name}...")

    # 获取该市场电价的时间轴
    df_price = pd.read_csv(mkt_cfg['price_file'], parse_dates=['timestamp_utc'])
    df_price = df_price[df_price['location'] == mkt_cfg['location']]
    df_price['timestamp_utc'] = pd.to_datetime(df_price['timestamp_utc'], utc=True)
    df_price = df_price.sort_values('timestamp_utc')

    ts_min = df_price['timestamp_utc'].min().tz_localize(None)
    ts_max = df_price['timestamp_utc'].max().tz_localize(None)

    # 创建完整小时索引
    hourly_idx = pd.date_range(start=ts_min, end=ts_max, freq='h')
    aligned = pd.DataFrame(index=hourly_idx)
    aligned.index.name = 'timestamp_utc'

    # 日频范围
    daily_range = pd.date_range(
        start=aligned.index.min().date(),
        end=aligned.index.max().date(),
        freq='D'
    )

    # 对齐每个协变量
    for cov_name, cov_cfg in cov_sources.items():
        daily_series = load_and_ffill_to_daily(
            cov_cfg['file'], cov_cfg['date_col'],
            cov_cfg['value_col'], daily_range
        )
        date_to_val = daily_series.to_dict()
        aligned[cov_name] = [
            date_to_val.get(pd.Timestamp(d), np.nan)
            for d in aligned.index.date
        ]
        aligned[cov_name] = aligned[cov_name].ffill().bfill()

    # 验证
    n = len(aligned)
    for col in aligned.columns:
        n_valid = aligned[col].notna().sum()
        pct = 100 * n_valid / n
        print(f"    {col}: {n_valid}/{n} ({pct:.1f}%)")

    # 保存
    out_path = os.path.join(MERGED_DIR, f'{mkt_name.lower()}_covariates_hourly.csv')
    aligned.to_csv(out_path)
    print(f"    Saved: {out_path}\n")

print("=== Alignment Complete ===")
