"""
P0 协变量下载：Henry Hub 天然气日频价格 + 天然气地下库存周报
数据源：FRED (Henry Hub) + EIA (Storage)
输出：data/covariates/gas/raw/ 下的原始文件 + gas/cleaned/ 下的清洗文件
"""
import os
import requests
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'covariates', 'gas', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'covariates', 'gas', 'cleaned')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def download_fred_csv(series_id, start='2024-01-01', end='2026-07-14'):
    """从 FRED 下载 CSV（无需 API Key）"""
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&cosd={start}&coed={end}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


# ═══ 1. Henry Hub 天然气现货价（日频）═══
# FRED Series: DHHNGSP
# 含义：Henry Hub 天然气现货价格，单位 $/MMBtu
# Henry Hub 是路易斯安那州的天然气管道交汇枢纽，其价格是全美天然气基准价
# 只有交易日有数据（周末/节假日缺失）
print("=== 1. Henry Hub Natural Gas Spot Price (FRED: DHHNGSP) ===")
raw_text = download_fred_csv('DHHNGSP')
raw_path = os.path.join(RAW_DIR, 'henry_hub_raw.csv')
with open(raw_path, 'w') as f:
    f.write(raw_text)
print(f"  Raw saved: {raw_path}")

df_hh = pd.read_csv(raw_path, parse_dates=['observation_date'])
df_hh = df_hh.rename(columns={
    'observation_date': 'date',
    'DHHNGSP': 'henry_hub_usd_per_mmbtu'
})
df_hh['henry_hub_usd_per_mmbtu'] = pd.to_numeric(
    df_hh['henry_hub_usd_per_mmbtu'], errors='coerce'
)
df_hh = df_hh.dropna(subset=['henry_hub_usd_per_mmbtu'])
df_hh = df_hh.sort_values('date').reset_index(drop=True)

proc_path = os.path.join(PROC_DIR, 'henry_hub_daily.csv')
df_hh.to_csv(proc_path, index=False)
print(f"  Processed saved: {proc_path}")
print(f"  Rows: {len(df_hh)}")
print(f"  Date range: {df_hh['date'].min().date()} to {df_hh['date'].max().date()}")
print(f"  Price: ${df_hh['henry_hub_usd_per_mmbtu'].min():.2f}"
      f" - ${df_hh['henry_hub_usd_per_mmbtu'].max():.2f}"
      f", Mean: ${df_hh['henry_hub_usd_per_mmbtu'].mean():.2f}")


# ═══ 2. 天然气地下库存（周频）═══
# EIA: Weekly Lower 48 States Natural Gas Working Underground Storage
# 含义：美国本土 48 州天然气地下储气量，单位 Bcf（十亿立方英尺）
# EIA 每周四发布上一周数据
# 反映供给侧状况：库存低 → 供应紧张 → 气价/电价上行
print("\n=== 2. Natural Gas Underground Storage (EIA Weekly) ===")
xls_url = 'https://www.eia.gov/dnav/ng/hist_xls/nw2_epg0_swo_r48_bcfw.xls'
r = requests.get(xls_url, timeout=30)
r.raise_for_status()

raw_path = os.path.join(RAW_DIR, 'natgas_storage_raw.xls')
with open(raw_path, 'wb') as f:
    f.write(r.content)
print(f"  Raw saved: {raw_path}")

df_st = pd.read_excel(raw_path, sheet_name='Data 1', header=2)
df_st.columns = ['date', 'natgas_storage_bcf']
df_st['date'] = pd.to_datetime(df_st['date'])
df_st['natgas_storage_bcf'] = pd.to_numeric(
    df_st['natgas_storage_bcf'], errors='coerce'
)
df_st = df_st.dropna().sort_values('date').reset_index(drop=True)

proc_path = os.path.join(PROC_DIR, 'natgas_storage_weekly.csv')
df_st.to_csv(proc_path, index=False)
print(f"  Processed saved: {proc_path}")
print(f"  Rows: {len(df_st)}")
print(f"  Date range: {df_st['date'].min().date()} to {df_st['date'].max().date()}")
print(f"  Storage: {df_st['natgas_storage_bcf'].min():.0f}"
      f" - {df_st['natgas_storage_bcf'].max():.0f} Bcf")

print("\n=== P0 Download Complete ===")
