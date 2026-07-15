# 产业协变量数据处理脚本

按编号顺序运行即可复现全部结果。

## 运行顺序

```bash
cd school/

# 1. 下载 P0 原始数据（Henry Hub 天然气价格 + 天然气库存）
python scripts/covariates/01_download_p0.py

# 2. 下载 P1 原始数据（WTI 原油 + 钢铁 PPI + 钢铁生产指数）
python scripts/covariates/02_download_p1.py

# 3. 对齐 P0+P1 协变量到四个市场的小时频率电价时间轴
python scripts/covariates/03_align_to_hourly.py

# 4. P0+P1 相关性分析 + 出图
python scripts/covariates/04_correlation_analysis.py

# 5. 下载 NOAA Storm Events 并处理成市场级日频/小时频特征 (P2)
python scripts/covariates/05_download_noaa_storm.py

# 6. 从 EIA generation_by_fuel 提取供给侧燃料结构特征 (P2)
python scripts/covariates/06_eia_fuel_mix_features.py

# 7. 合并所有协变量 (P0+P1+P2) 为统一小时级文件
python scripts/covariates/07_merge_all_covariates.py

# 8. 下载 GDELT 能源新闻（P2，半成品：1/96 抽样、用 tone 非 LLM）
python scripts/covariates/08_download_gdelt_news.py

# 9. 稳健性分析：差分相关 + 滞后峰值表 + 共线性报告
python scripts/covariates/09_robustness_analysis.py

# 10. 按领域逻辑筛市场专属特征（钢→PJM、WTI→ERCOT、其余→全部）→ model_ready
python scripts/covariates/10_build_model_ready.py
```

## 依赖

```
pip install pandas requests scipy matplotlib xlrd numpy statsmodels
```

## 输出文件

协变量数据统一放在 `data/covariates/`，按**协变量类型**分目录，每类下按**处理阶段**（raw / cleaned / hourly）细分。与原始数据 `data/raw/` 平级、分开。

### data/covariates/gas/（天然气，P0，四市场通用）
- `raw/henry_hub_raw.csv` — FRED Henry Hub 日频价格
- `raw/natgas_storage_raw.xls` — EIA 天然气库存周报
- `cleaned/henry_hub_daily.csv` — 清洗后天然气日频价格
- `cleaned/natgas_storage_weekly.csv` — 清洗后天然气库存周频

### data/covariates/oil/（原油，P1，ERCOT 专用）
- `raw/wti_crude_raw.csv` — FRED WTI 日频价格
- `cleaned/wti_crude_daily.csv` — 清洗后 WTI 日频

### data/covariates/steel/（钢铁，P1，PJM 专用）
- `raw/hrc_futures_raw.csv` — investing.com 手动下载的 HRC 钢卷期货（中文表头/倒序/千分位）
- `cleaned/hrc_futures_daily.csv` — 清洗后日频钢价（主用，已替换月度 PPI/产量）
- ~~`raw/steel_ppi_raw.csv`、`raw/steel_ip_raw.csv`~~（legacy 已弃用，粒度太粗）
- ~~`cleaned/steel_ppi_monthly.csv`、`cleaned/steel_production_monthly.csv`~~（legacy 已弃用）

### data/covariates/storm/（NOAA 风暴事件，P2）
- `raw/noaa_storm/StormEvents_details_d{year}.csv.gz` — 原始归档
- `cleaned/noaa_storm_daily.csv` — 四市场日频特征
- `hourly/{market}_storm_hourly.csv` — 小时频特征 (forward-fill)

### data/covariates/generation_mix/（EIA 燃料结构，P2；源在 data/raw/EIA）
- `hourly/{market}_fuel_mix_hourly.csv` — 发电量/份额/可再生出力小时频特征

### data/covariates/news/（GDELT 新闻，P2，半成品）
- `raw/gdelt_news_raw.csv`、`raw/gdelt_progress.txt` — 逐条明细 + 断点续传
- `cleaned/gdelt_news_daily.csv` — 按市场日频聚合
- `hourly/{market}_news_hourly.csv` — 小时频 (次日 forward-fill 防泄漏)

### data/covariates/merged/（跨类型合并，对齐到小时）
- `{market}_covariates_hourly.csv` — P0+P1 经济协变量（gas+oil+steel）
- `{market}_all_covariates_hourly.csv` — 全部 20 列合并，喂模型用

### data/covariates/analysis/（相关性 + 稳健性）
- `p0_correlation_results.csv`、`p1_correlation_results.csv` — Pearson/Spearman 相关
- `robust_correlation_results.csv` — first-difference 后的相关（去趋势，回答"水平相关是否伪信号"）
- `lag_peak_results.csv` — 每市场每协变量滞后峰值（lag, r），正 lag=协变量领先电价
- `covariate_collinearity.csv` — 各协变量 VIF（小时层）
- `covariate_collinearity_high.csv` — 高共线对（|r|>0.85）

### analysis/figures/covariates/（可视化）
- `p0_scatter_gas_vs_price.png` — 天然气价格 vs 电价散点图
- `p0_scatter_storage_vs_price.png` — 天然气库存 vs 电价散点图
- `p0_timeseries_overlay.png` — 电价与气价时序叠加图
- `p0_lagged_crosscorr.png` — 天然气价格滞后交叉相关图
- `p1_scatter_wti_vs_price.png` — WTI 原油 vs 电价散点图
- `p1_scatter_steel_vs_pjm.png` — 钢铁指标 vs PJM 电价散点图
- `p1_timeseries_steel_pjm.png` — PJM 电价与钢铁 PPI 时序叠加图
- `p1_lagged_wti_ercot.png` — WTI 原油滞后交叉相关图（ERCOT）

> 注：温度协变量不在本流水线，模型 loader 直接读 `data/raw/weather/processed/by_ba/{BA}_weather_hourly.csv`（原始数据，未并入 `data/covariates/`）。

## 协变量完整列表（20列）

| 列名 | 来源 | 频率→对齐 | 含义 |
|------|------|-----------|------|
| henry_hub_usd_per_mmbtu | FRED DHHNGSP | 日→小时 ffill | Henry Hub 天然气现货价 |
| natgas_storage_bcf | EIA 周报 XLS | 周→小时 ffill | 美国天然气地下储量 |
| wti_usd_per_barrel | FRED DCOILWTICO | 日→小时 ffill | WTI 原油现货价 |
| hrc_futures_usd_per_ton | investing.com HRC 期货 | 日→小时 ffill | HRC 钢卷期货日频价（已替换月度 PPI/产量，粒度更细） |
| storm_event_count | NOAA Storm Events | 日→小时 ffill | 当日该市场区域内总风暴事件数 |
| storm_high_impact_count | NOAA Storm Events | 日→小时 ffill | 高影响事件数（对电力系统有直接影响） |
| storm_damage_usd | NOAA Storm Events | 日→小时 ffill | 当日总财产损失（美元） |
| storm_injuries | NOAA Storm Events | 日→小时 ffill | 当日总伤亡人数 |
| storm_has_extreme_temp | NOAA Storm Events | 日→小时 ffill | 是否有极端温度事件 (0/1) |
| storm_has_wind | NOAA Storm Events | 日→小时 ffill | 是否有大风/龙卷风事件 (0/1) |
| storm_has_winter | NOAA Storm Events | 日→小时 ffill | 是否有冬季风暴事件 (0/1) |
| total_gen_mwh | EIA EBA generation | 小时 | 总发电量 |
| gas_gen_mwh | EIA EBA generation | 小时 | 天然气发电量 |
| wind_gen_mwh | EIA EBA generation | 小时 | 风力发电量 |
| solar_gen_mwh | EIA EBA generation | 小时 | 太阳能发电量 |
| gas_share | EIA EBA generation | 小时 | 天然气发电占比（边际定价燃料） |
| renewable_share | EIA EBA generation | 小时 | 可再生能源（风+光+水）发电占比 |
| gas_share_diff | EIA EBA generation | 小时 | gas_share 逐时变化量 |
| renewable_shock | EIA EBA generation | 小时 | 可再生能源出力 24h 滚动 z-score |
