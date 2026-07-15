"""
协变量稳健性分析（汇报材料补充）

针对 04 同步相关性结果的三类质疑，产出三张表：

1. 差分/去趋势相关（硬伤A）
   - 在原始价格水平上的 Pearson/Spearman 可能被共同趋势/季节性撑高
   - 这里报 first-difference 后的 Pearson/Spearman，看信号在去掉趋势后是否还在
   输出: analysis/robust_correlation_results.csv

2. 滞后峰值表
   - 04 的 lagged crosscorr 只画了图、峰值只进图标题没存表
   - 这里把每市场每协变量的峰值 lag 与 r 落表
   - 正 lag = 协变量领先电价（可作预测协变量的依据）
   输出: analysis/lag_peak_results.csv

3. 协变量冗余/共线性（硬伤B 的数据侧）
   - gas_share = gas_gen_mwh / total_gen_mwh（完全共线）
   - 多个 storm 严重度列相互共线
   - 输出每市场的相关矩阵高共线对(|r|>0.9) + 各列 VIF
   输出: analysis/covariate_collinearity.csv
        analysis/covariate_collinearity_high.csv

用法: python scripts/covariates/09_robustness_analysis.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MERGED_DIR = os.path.join(ROOT, 'data', 'covariates', 'merged')
ANALYSIS_DIR = os.path.join(ROOT, 'data', 'covariates', 'analysis')
DATA_DIR = os.path.join(ROOT, 'data', 'raw')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MARKETS = {
    'ERCOT': {'file': os.path.join(DATA_DIR, 'ERCOT', 'processed', 'actual_price_hourly.csv'), 'loc': 'HB_HUBAVG'},
    'PJM':   {'file': os.path.join(DATA_DIR, 'PJM', 'processed', 'actual_price_hourly.csv'),   'loc': 'HUB:WESTERN HUB'},
    'CAISO': {'file': os.path.join(DATA_DIR, 'CAISO', 'processed', 'actual_price_hourly.csv'),  'loc': 'TH_SP15_GEN-APND'},
    'NYISO': {'file': os.path.join(DATA_DIR, 'NYISO', 'processed', 'actual_price_hourly.csv'),  'loc': 'CENTRL'},
}

# 与 04 保持一致的协变量清单
P0_COV = {'henry_hub_usd_per_mmbtu': 'Henry Hub Gas Price',
          'natgas_storage_bcf': 'Gas Storage'}
P1_COV = {'wti_usd_per_barrel': 'WTI Crude Oil',
          'hrc_futures_usd_per_ton': 'HRC Steel Futures'}
ALL_COV = {**P0_COV, **P1_COV}


def load_market(mkt_name):
    """加载电价 + 协变量并合并（复用 04 的逻辑）"""
    cfg = MARKETS[mkt_name]
    df_p = pd.read_csv(cfg['file'], parse_dates=['timestamp_utc'])
    df_p = df_p[df_p['location'] == cfg['loc']][['timestamp_utc', 'value']].rename(columns={'value': 'price'})
    df_p['timestamp_utc'] = pd.to_datetime(df_p['timestamp_utc'], utc=True)
    df_p = df_p.set_index('timestamp_utc').sort_index()
    df_p['price'] = pd.to_numeric(df_p['price'], errors='coerce')

    cov_path = os.path.join(MERGED_DIR, f'{mkt_name.lower()}_covariates_hourly.csv')
    df_c = pd.read_csv(cov_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
    df_c.index = df_c.index.tz_localize('UTC')

    merged = df_p.join(df_c, how='inner').dropna(subset=['price'])
    daily = merged.resample('D').mean().dropna(subset=['price'])
    return merged, daily


def lagged_crosscorr(daily, cov_col, max_lag=7):
    """滞后交叉相关（正 lag = 协变量领先电价）。复用 04 的实现。"""
    x_full = daily[cov_col].values
    y_full = daily['price'].values
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = x_full[:len(x_full) - lag] if lag > 0 else x_full
            y = y_full[lag:] if lag > 0 else y_full
        else:
            x = x_full[-lag:]
            y = y_full[:len(y_full) + lag]
        n = min(len(x), len(y))
        if n > 10:
            r, _ = stats.pearsonr(x[:n], y[:n])
            results.append((lag, r))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 1. 差分/去趋势相关
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("1. First-difference correlation (硬伤A: 去趋势后信号是否还在)")
print("=" * 64)

robust_rows = []
for mkt in MARKETS:
    _, daily = load_market(mkt)
    d_price = daily['price'].diff().dropna()
    for cov_col, cov_label in ALL_COV.items():
        d_cov = daily[cov_col].diff().dropna()
        idx = d_price.index.intersection(d_cov.index)
        x = d_cov.loc[idx]
        y = d_price.loc[idx]
        if len(idx) > 10:
            r, p = stats.pearsonr(x, y)
            s, _ = stats.spearmanr(x, y)
        else:
            r = s = p = np.nan
        # 同时取原始水平相关作对照
        r_lvl, _ = stats.pearsonr(daily[cov_col], daily['price'])
        robust_rows.append({
            'market': mkt, 'covariate': cov_label,
            'level_pearson_r': round(r_lvl, 4),
            'diff_pearson_r': round(r, 4), 'diff_pearson_p': p,
            'diff_spearman_r': round(s, 4),
            'n': len(idx),
        })
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {mkt:5s} x {cov_label:25s} | level r={r_lvl:+.4f} -> diff r={r:+.4f}{sig}")

robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(os.path.join(ANALYSIS_DIR, 'robust_correlation_results.csv'), index=False)
print(f"  -> saved: processed/robust_correlation_results.csv\n")

# ═════════════════════════════════════════════════════════════════════════════
# 2. 滞后峰值表
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("2. Lag peak table (正 lag = 协变量领先电价)")
print("=" * 64)

lag_rows = []
for mkt in MARKETS:
    _, daily = load_market(mkt)
    for cov_col, cov_label in ALL_COV.items():
        lc = lagged_crosscorr(daily, cov_col)
        # max r（与 04 图标题一致）与 max|r|
        best_max = max(lc, key=lambda t: t[1])
        best_abs = max(lc, key=lambda t: abs(t[1]))
        r_lag0 = dict(lc).get(0, np.nan)
        lag_rows.append({
            'market': mkt, 'covariate': cov_label,
            'lag_at_max_r': best_max[0], 'r_at_max_r': round(best_max[1], 4),
            'lag_at_max_abs_r': best_abs[0], 'r_at_max_abs_r': round(best_abs[1], 4),
            'r_at_lag0': round(r_lag0, 4),
            'leads_price': 'yes' if best_max[0] > 0 and best_max[1] > r_lag0 else ('no' if best_max[0] < 0 else 'same'),
        })
        print(f"  {mkt:5s} x {cov_label:25s} | best lag={best_max[0]:+d}d r={best_max[1]:+.4f} | lag0 r={r_lag0:+.4f}")

lag_df = pd.DataFrame(lag_rows)
lag_df.to_csv(os.path.join(ANALYSIS_DIR, 'lag_peak_results.csv'), index=False)
print(f"  -> saved: processed/lag_peak_results.csv\n")

# ═════════════════════════════════════════════════════════════════════════════
# 3. 协变量冗余/共线性
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("3. Covariate collinearity (VIF + 高相关对 |r|>0.9)")
print("=" * 64)

# 只看连续型协变量（排除二值标志），二值标志单列
CONTINUOUS = [
    'henry_hub_usd_per_mmbtu', 'natgas_storage_bcf', 'wti_usd_per_barrel',
    'steel_ppi_index', 'steel_production_index',
    'storm_event_count', 'storm_high_impact_count', 'storm_damage_usd', 'storm_injuries',
    'total_gen_mwh', 'gas_gen_mwh', 'wind_gen_mwh', 'solar_gen_mwh',
    'gas_share', 'renewable_share', 'gas_share_diff',
]
BINARY = ['storm_has_extreme_temp', 'storm_has_wind', 'storm_has_winter', 'renewable_shock']

vif_rows = []
high_pairs = []
for mkt in MARKETS:
    # 共线必须在模型实际消费的【小时】层检查，且用 _all_ 文件（含 storm/gen/gas_share）
    all_path = os.path.join(MERGED_DIR, f'{mkt.lower()}_all_covariates_hourly.csv')
    df_all = pd.read_csv(all_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
    avail = [c for c in CONTINUOUS if c in df_all.columns]
    sub = df_all[avail].dropna()
    # 相关矩阵高共线对（阈值 0.85，抓 storm 计数对与 gas_share~renewable_share）
    corr = sub.corr()
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            r = corr.iloc[i, j]
            if not np.isnan(r) and abs(r) > 0.85:
                high_pairs.append({'market': mkt, 'var_a': avail[i], 'var_b': avail[j], 'pearson_r': round(r, 4)})
    # VIF（逐列；线性共线会出大值/inf）
    for i, col in enumerate(avail):
        try:
            vif = variance_inflation_factor(sub.values, i)
        except Exception:
            vif = np.inf
        vif_rows.append({'market': mkt, 'variable': col, 'VIF': round(vif, 2) if np.isfinite(vif) else 'inf'})

vif_df = pd.DataFrame(vif_rows)
vif_df.to_csv(os.path.join(ANALYSIS_DIR, 'covariate_collinearity.csv'), index=False)
high_df = pd.DataFrame(high_pairs)
high_df.to_csv(os.path.join(ANALYSIS_DIR, 'covariate_collinearity_high.csv'), index=False)

print("  高共线对 (|r|>0.85):")
if len(high_df):
    for _, r in high_df.iterrows():
        print(f"    {r['market']:5s} {r['var_a']:24s} ~ {r['var_b']:24s} r={r['pearson_r']:+.4f}")
else:
    print("    (无 |r|>0.85 的连续协变量对)")
print("  VIF>10（线性共线，多为月度慢变量同趋势）:")
for _, r in vif_df.iterrows():
    v = r['VIF']
    if (isinstance(v, str) and v == 'inf') or (isinstance(v, (int, float)) and v > 10):
        print(f"    {r['market']:5s} {r['variable']:24s} VIF={v}")
# 功能性依赖（比值关系，VIF/两两相关抓不到，靠定义）
print("  功能性依赖（非共线、靠定义）:")
print("    gas_share ≡ gas_gen_mwh / total_gen_mwh  -> 三者只留其二，或只留 gas_share")
print(f"\n  -> saved: processed/covariate_collinearity.csv")
print(f"  -> saved: processed/covariate_collinearity_high.csv")
print("\nDone.")
