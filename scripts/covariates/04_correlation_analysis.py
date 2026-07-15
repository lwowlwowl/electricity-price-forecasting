"""
协变量相关性分析 + 可视化

对所有 P0+P1 协变量 vs 四个市场电价进行：
1. 同步相关性（Pearson + Spearman，小时级和日均值）
2. 滞后交叉相关（日均值，lag -7 到 +7 天）
3. 散点图、时序叠加图、滞后相关图

输出：
- processed/p0_correlation_results.csv
- processed/p1_correlation_results.csv
- analysis/figures/covariates/*.png
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MERGED_DIR = os.path.join(ROOT, 'data', 'covariates', 'merged')
ANALYSIS_DIR = os.path.join(ROOT, 'data', 'covariates', 'analysis')
DATA_DIR = os.path.join(ROOT, 'data', 'raw')
FIG_DIR = os.path.join(ROOT, 'analysis', 'figures', 'covariates')
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MARKETS = {
    'ERCOT': {'file': os.path.join(DATA_DIR, 'ERCOT', 'processed', 'actual_price_hourly.csv'), 'loc': 'HB_HUBAVG'},
    'PJM':   {'file': os.path.join(DATA_DIR, 'PJM', 'processed', 'actual_price_hourly.csv'),   'loc': 'HUB:WESTERN HUB'},
    'CAISO': {'file': os.path.join(DATA_DIR, 'CAISO', 'processed', 'actual_price_hourly.csv'),  'loc': 'TH_SP15_GEN-APND'},
    'NYISO': {'file': os.path.join(DATA_DIR, 'NYISO', 'processed', 'actual_price_hourly.csv'),  'loc': 'CENTRL'},
}

COLORS = {'ERCOT': '#E63946', 'PJM': '#457B9D', 'CAISO': '#2A9D8F', 'NYISO': '#E9C46A'}

# P0 协变量（四个市场通用）
P0_COVARIATES = {
    'henry_hub_usd_per_mmbtu': 'Henry Hub Gas Price',
    'natgas_storage_bcf': 'Gas Storage',
}

# P1 协变量
P1_COVARIATES = {
    'wti_usd_per_barrel': 'WTI Crude Oil',
    'hrc_futures_usd_per_ton': 'HRC Steel Futures',
}


def load_market_data(mkt_name):
    """加载电价 + 协变量并合并"""
    cfg = MARKETS[mkt_name]
    df_p = pd.read_csv(cfg['file'], parse_dates=['timestamp_utc'])
    df_p = df_p[df_p['location'] == cfg['loc']][['timestamp_utc', 'value']]
    df_p = df_p.rename(columns={'value': 'price'})
    df_p['timestamp_utc'] = pd.to_datetime(df_p['timestamp_utc'], utc=True)
    df_p = df_p.set_index('timestamp_utc').sort_index()
    df_p['price'] = pd.to_numeric(df_p['price'], errors='coerce')

    cov_path = os.path.join(MERGED_DIR, f'{mkt_name.lower()}_covariates_hourly.csv')
    df_c = pd.read_csv(cov_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
    df_c.index = df_c.index.tz_localize('UTC')

    merged = df_p.join(df_c, how='inner').dropna()
    daily = merged.resample('D').mean().dropna()
    return merged, daily


def compute_correlations(merged, daily, cov_col):
    """计算 Pearson/Spearman 相关系数"""
    r_h, p_h = stats.pearsonr(merged[cov_col], merged['price'])
    s_h, _ = stats.spearmanr(merged[cov_col], merged['price'])
    r_d, p_d = stats.pearsonr(daily[cov_col], daily['price'])
    s_d, _ = stats.spearmanr(daily[cov_col], daily['price'])
    return {
        'hourly_pearson_r': round(r_h, 4), 'hourly_spearman_r': round(s_h, 4),
        'hourly_pearson_p': p_h,
        'daily_pearson_r': round(r_d, 4), 'daily_spearman_r': round(s_d, 4),
        'daily_pearson_p': p_d,
        'n_hourly': len(merged), 'n_daily': len(daily),
    }


def lagged_crosscorr(daily, cov_col, max_lag=7):
    """计算滞后交叉相关（正 lag = 协变量领先电价）"""
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


# ═══════════════════════════════════════════════
# P0 相关性分析
# ═══════════════════════════════════════════════
print("=" * 60)
print("P0 Correlation Analysis")
print("=" * 60)

p0_results = []
for mkt in MARKETS:
    merged, daily = load_market_data(mkt)
    for cov_col, cov_label in P0_COVARIATES.items():
        corr = compute_correlations(merged, daily, cov_col)
        corr.update({'market': mkt, 'covariate': cov_label})
        p0_results.append(corr)
        sig = '***' if corr['daily_pearson_p'] < 0.001 else '**' if corr['daily_pearson_p'] < 0.01 else '*' if corr['daily_pearson_p'] < 0.05 else 'ns'
        print(f"  {mkt:5s} x {cov_label:25s} | Daily r={corr['daily_pearson_r']:+.4f}{sig}")

pd.DataFrame(p0_results).to_csv(os.path.join(ANALYSIS_DIR, 'p0_correlation_results.csv'), index=False)

# ─── P0 Fig 1: Henry Hub vs Price scatter ───
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Henry Hub Natural Gas Price vs. Electricity Price (Daily Mean)', fontsize=16, fontweight='bold')
for idx, mkt in enumerate(MARKETS):
    ax = axes[idx // 2][idx % 2]
    _, daily = load_market_data(mkt)
    x, y = daily['henry_hub_usd_per_mmbtu'], daily['price']
    ax.scatter(x, y, alpha=0.5, s=15, color=COLORS[mkt])
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, label=f'r={r:.3f}, p={p:.1e}')
    ax.set_xlabel('Henry Hub ($/MMBtu)')
    ax.set_ylabel('Electricity Price ($/MWh)')
    ax.set_title(mkt, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p0_scatter_gas_vs_price.png'), dpi=150, bbox_inches='tight')
print(f"\n  Saved: p0_scatter_gas_vs_price.png")

# ─── P0 Fig 2: Storage vs Price scatter ───
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Natural Gas Storage vs. Electricity Price (Daily Mean)', fontsize=16, fontweight='bold')
for idx, mkt in enumerate(MARKETS):
    ax = axes[idx // 2][idx % 2]
    _, daily = load_market_data(mkt)
    x, y = daily['natgas_storage_bcf'], daily['price']
    ax.scatter(x, y, alpha=0.5, s=15, color=COLORS[mkt])
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, label=f'r={r:.3f}, p={p:.1e}')
    ax.set_xlabel('Gas Storage (Bcf)')
    ax.set_ylabel('Electricity Price ($/MWh)')
    ax.set_title(mkt, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p0_scatter_storage_vs_price.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p0_scatter_storage_vs_price.png")

# ─── P0 Fig 3: Time series overlay ───
fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
fig.suptitle('Electricity Price vs. Henry Hub Gas Price (Daily)', fontsize=16, fontweight='bold')
for idx, mkt in enumerate(MARKETS):
    ax = axes[idx]
    _, daily = load_market_data(mkt)
    ax.plot(daily.index, daily['price'], color=COLORS[mkt], lw=1, alpha=0.8, label=f'{mkt} Price')
    ax.set_ylabel('Elec. Price ($/MWh)', color=COLORS[mkt])
    ax.tick_params(axis='y', labelcolor=COLORS[mkt])
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily['henry_hub_usd_per_mmbtu'], color='black', lw=1, alpha=0.6, label='Henry Hub')
    ax2.set_ylabel('Gas Price ($/MMBtu)')
    ax.set_title(mkt, fontweight='bold')
    ax.grid(True, alpha=0.2)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p0_timeseries_overlay.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p0_timeseries_overlay.png")

# ─── P0 Fig 4: Lagged cross-correlation (Henry Hub) ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Lagged Cross-Correlation: Henry Hub Gas Price → Electricity Price (Daily)', fontsize=14, fontweight='bold')
for idx, mkt in enumerate(MARKETS):
    ax = axes[idx // 2][idx % 2]
    _, daily = load_market_data(mkt)
    lc = lagged_crosscorr(daily, 'henry_hub_usd_per_mmbtu')
    best = max(lc, key=lambda t: abs(t[1]))
    lags = [t[0] for t in lc]
    corrs = [t[1] for t in lc]
    bar_colors = ['#E63946' if l == best[0] else '#457B9D' for l in lags]
    ax.bar(lags, corrs, color=bar_colors, alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Lag (days, positive = gas leads)')
    ax.set_ylabel('Pearson r')
    ax.set_title(f"{mkt} (best: lag={best[0]}d, r={best[1]:+.3f})", fontweight='bold')
    ax.set_xticks(range(-7, 8))
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p0_lagged_crosscorr.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p0_lagged_crosscorr.png")


# ═══════════════════════════════════════════════
# P1 相关性分析
# ═══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("P1 Correlation Analysis")
print("=" * 60)

p1_results = []
for mkt in MARKETS:
    merged, daily = load_market_data(mkt)
    for cov_col, cov_label in P1_COVARIATES.items():
        corr = compute_correlations(merged, daily, cov_col)
        corr.update({'market': mkt, 'covariate': cov_label})
        p1_results.append(corr)
        sig = '***' if corr['daily_pearson_p'] < 0.001 else '**' if corr['daily_pearson_p'] < 0.01 else '*' if corr['daily_pearson_p'] < 0.05 else 'ns'
        print(f"  {mkt:5s} x {cov_label:25s} | Daily r={corr['daily_pearson_r']:+.4f}{sig}")

pd.DataFrame(p1_results).to_csv(os.path.join(ANALYSIS_DIR, 'p1_correlation_results.csv'), index=False)

# ─── P1 Fig 1: WTI vs Price scatter ───
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('WTI Crude Oil Price vs. Electricity Price (Daily Mean)', fontsize=16, fontweight='bold')
for idx, mkt in enumerate(MARKETS):
    ax = axes[idx // 2][idx % 2]
    _, daily = load_market_data(mkt)
    x, y = daily['wti_usd_per_barrel'], daily['price']
    ax.scatter(x, y, alpha=0.5, s=15, color=COLORS[mkt])
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, label=f'r={r:.3f}, p={p:.1e}')
    ax.set_xlabel('WTI Crude ($/barrel)')
    ax.set_ylabel('Electricity Price ($/MWh)')
    ax.set_title(mkt, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p1_scatter_wti_vs_price.png'), dpi=150, bbox_inches='tight')
print(f"\n  Saved: p1_scatter_wti_vs_price.png")

# ─── P1 Fig 2: Steel covariates vs PJM ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Steel Industry Covariates vs. PJM Electricity Price (Daily Mean)', fontsize=14, fontweight='bold')
_, daily_pjm = load_market_data('PJM')
for i, (cov, label) in enumerate([
    ('steel_ppi_index', 'Steel PPI (Iron & Steel)'),
    ('steel_production_index', 'Steel Production Index'),
]):
    ax = axes[i]
    x, y = daily_pjm[cov], daily_pjm['price']
    ax.scatter(x, y, alpha=0.5, s=15, color='#457B9D')
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', lw=1.5, label=f'r={r:.3f}, p={p:.1e}')
    ax.set_xlabel(f'{label} (Index)')
    ax.set_ylabel('PJM Price ($/MWh)')
    ax.set_title(label, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p1_scatter_steel_vs_pjm.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p1_scatter_steel_vs_pjm.png")

# ─── P1 Fig 3: PJM price + Steel PPI time series ───
fig, ax1 = plt.subplots(figsize=(16, 6))
fig.suptitle('PJM Electricity Price vs. Steel Industry Indicators (Daily)', fontsize=14, fontweight='bold')
ax1.plot(daily_pjm.index, daily_pjm['price'], color='#457B9D', lw=1, alpha=0.8, label='PJM Price')
ax1.set_ylabel('PJM Price ($/MWh)', color='#457B9D')
ax1.tick_params(axis='y', labelcolor='#457B9D')
ax2 = ax1.twinx()
ax2.plot(daily_pjm.index, daily_pjm['steel_ppi_index'], color='#E63946', lw=1.5, alpha=0.7, label='Steel PPI')
ax2.set_ylabel('Steel PPI Index', color='#E63946')
ax2.tick_params(axis='y', labelcolor='#E63946')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p1_timeseries_steel_pjm.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p1_timeseries_steel_pjm.png")

# ─── P1 Fig 4: Lagged cross-corr WTI → ERCOT ───
_, daily_ercot = load_market_data('ERCOT')
lc = lagged_crosscorr(daily_ercot, 'wti_usd_per_barrel')
best = max(lc, key=lambda t: abs(t[1]))
print(f"\n  WTI → ERCOT lagged cross-corr: best lag={best[0]}d, r={best[1]:+.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
lags = [t[0] for t in lc]
corrs = [t[1] for t in lc]
bar_colors = ['#E63946' if l == best[0] else '#457B9D' for l in lags]
ax.bar(lags, corrs, color=bar_colors, alpha=0.7)
ax.axhline(y=0, color='black', lw=0.5)
ax.set_xlabel('Lag (days, positive = WTI leads)')
ax.set_ylabel('Pearson r')
ax.set_title(f"Lagged Cross-Correlation: WTI → ERCOT Price (best: lag={best[0]}d, r={best[1]:+.3f})", fontweight='bold')
ax.set_xticks(range(-7, 8))
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'p1_lagged_wti_ercot.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: p1_lagged_wti_ercot.png")

print(f"\n{'=' * 60}")
print("All Done")
print("=" * 60)
