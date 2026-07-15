"""
P1 协变量：WTI 原油日频价格 + HRC 钢卷期货日频价格
- WTI：FRED DCOILWTICO（自动下载，无 key）
- HRC 钢卷期货：investing.com 手动下载的 raw（中文表头/倒序/千分位/BOM）→ 本脚本清洗成日频
  （已替换原先的月度 PPI/产量指数——粒度太粗、与电价近零相关；日频 HRC 是 xlsx 原本想要的钢价）
输出：
  WTI  → data/covariates/oil/{raw,cleaned}/
  钢铁 → data/covariates/steel/{raw,cleaned}/  (hrc_futures_daily.csv)
"""
import os
import requests
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OIL_RAW = os.path.join(ROOT, 'data', 'covariates', 'oil', 'raw')
OIL_CLEANED = os.path.join(ROOT, 'data', 'covariates', 'oil', 'cleaned')
STEEL_RAW = os.path.join(ROOT, 'data', 'covariates', 'steel', 'raw')
STEEL_CLEANED = os.path.join(ROOT, 'data', 'covariates', 'steel', 'cleaned')
for d in (OIL_RAW, OIL_CLEANED, STEEL_RAW, STEEL_CLEANED):
    os.makedirs(d, exist_ok=True)


def download_fred_csv(series_id, start='2024-01-01', end='2026-07-14'):
    """从 FRED 下载 CSV（无需 API Key）"""
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&cosd={start}&coed={end}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


# ═══ 1. WTI 原油日频价格 ═══
# FRED Series: DCOILWTICO
# 含义：West Texas Intermediate 原油现货价格，单位 $/barrel
# WTI 是北美原油基准价，对德州(ERCOT)经济影响最大
# 只有交易日有数据
print("=== 1. WTI Crude Oil Daily Price (FRED: DCOILWTICO) ===")
raw_text = download_fred_csv('DCOILWTICO')
raw_path = os.path.join(OIL_RAW, 'wti_crude_raw.csv')
with open(raw_path, 'w') as f:
    f.write(raw_text)
print(f"  Raw saved: {raw_path}")

df_wti = pd.read_csv(raw_path, parse_dates=['observation_date'])
df_wti = df_wti.rename(columns={
    'observation_date': 'date',
    'DCOILWTICO': 'wti_usd_per_barrel'
})
df_wti['wti_usd_per_barrel'] = pd.to_numeric(
    df_wti['wti_usd_per_barrel'], errors='coerce'
)
df_wti = df_wti.dropna().sort_values('date').reset_index(drop=True)

proc_path = os.path.join(OIL_CLEANED, 'wti_crude_daily.csv')
df_wti.to_csv(proc_path, index=False)
print(f"  Processed saved: {proc_path}")
print(f"  Rows: {len(df_wti)}")
print(f"  Date range: {df_wti['date'].min().date()} to {df_wti['date'].max().date()}")
print(f"  Price: ${df_wti['wti_usd_per_barrel'].min():.2f}"
      f" - ${df_wti['wti_usd_per_barrel'].max():.2f}"
      f", Mean: ${df_wti['wti_usd_per_barrel'].mean():.2f}")


# ═══ 2. HRC 钢卷期货日频价格（清洗 investing.com 手动下载的 raw）═══
# raw 文件格式：中文表头(日期/收盘/...)、倒序(最新在前)、千分位逗号("1,158.00")、UTF-8 BOM
# 含义：Hot-Rolled Coil 钢卷期货结算价，单位 $/吨，日频(交易日)
# 钢价高 → 钢厂(电弧炉)开工多 → 工业负荷升 → 电价上行；主要与 PJM(Ohio/PA 钢铁带)相关
# raw 需手动从 investing.com 下载历史数据放到 STEEL_RAW/hrc_futures_raw.csv（脚本无法自动下载）
print("\n=== 2. HRC Steel Coil Futures Daily (investing.com manual raw) ===")
hrc_raw = os.path.join(STEEL_RAW, 'hrc_futures_raw.csv')
if not os.path.exists(hrc_raw):
    print(f"  ⚠ 找不到 {hrc_raw}")
    print("  请先从 investing.com 手动下载 HRC 期货历史数据 CSV 放到该路径，再重跑本脚本。")
else:
    df_hrc = pd.read_csv(hrc_raw, encoding='utf-8-sig')   # utf-8-sig 处理 BOM
    df_hrc.columns = [c.strip().strip('"') for c in df_hrc.columns]
    # 取"收盘"列，去千分位逗号 → 数值
    close = df_hrc['收盘'].astype(str).str.replace(',', '', regex=False)
    df_hrc = pd.DataFrame({
        'date': pd.to_datetime(df_hrc['日期']),
        'hrc_futures_usd_per_ton': pd.to_numeric(close, errors='coerce'),
    })
    df_hrc = df_hrc.dropna().sort_values('date').reset_index(drop=True)
    proc_path = os.path.join(STEEL_CLEANED, 'hrc_futures_daily.csv')
    df_hrc.to_csv(proc_path, index=False)
    print(f"  Processed saved: {proc_path}")
    print(f"  Rows: {len(df_hrc)}")
    print(f"  Date range: {df_hrc['date'].min().date()} to {df_hrc['date'].max().date()}")
    print(f"  Price: ${df_hrc['hrc_futures_usd_per_ton'].min():.0f}"
          f" - ${df_hrc['hrc_futures_usd_per_ton'].max():.0f}"
          f", Mean: ${df_hrc['hrc_futures_usd_per_ton'].mean():.0f}")

print("\n=== P1 Download Complete ===")
