# 协变量数据目录说明

> 本目录与 `data/raw/`（原始市场/EIA/天气数据）平级、分开。协变量均由 `scripts/covariates/01–15` 生成，可随时复现。
> 本文件随数据一起被 git 忽略（与 `data/raw/README.md` 同惯例），仅作本地参考。

## 目录结构（按协变量类型 + 处理阶段）

```
data/covariates/
├── gas/            天然气（P0，四市场通用）
│   ├── raw/          henry_hub_raw.csv, natgas_storage_raw.xls
│   └── cleaned/      henry_hub_daily.csv, natgas_storage_weekly.csv
├── oil/            原油 WTI（P1，ERCOT 专用）
│   ├── raw/          wti_crude_raw.csv
│   └── cleaned/      wti_crude_daily.csv
├── steel/         钢铁（P1，PJM 专用）
│   ├── raw/          hrc_futures_raw.csv (主用，investing.com 手动下载); steel_ppi_raw.csv, steel_ip_raw.csv (legacy 已弃用)
│   └── cleaned/      hrc_futures_daily.csv (主用，日频); steel_ppi_monthly.csv, steel_production_monthly.csv (legacy 已弃用)
├── storm/         NOAA 风暴事件（P2）
│   ├── raw/noaa_storm/   StormEvents_details_d{year}.csv.gz
│   ├── cleaned/           noaa_storm_daily.csv
│   └── hourly/            {market}_storm_hourly.csv
├── generation_mix/ EIA 发电燃料结构（P2；源在 data/raw/EIA）
│   └── hourly/           {market}_fuel_mix_hourly.csv
├── news/          GDELT 能源新闻（P2，半成品：1/96 抽样、用 tone 非 LLM）
│   ├── raw/          gdelt_news_raw.csv, gdelt_progress.txt
│   ├── cleaned/      gdelt_news_daily.csv
│   └── hourly/       {market}_news_hourly.csv
├── merged/        跨类型合并（对齐到小时）
│   ├── {market}_covariates_hourly.csv     P0+P1 经济（gas+oil+steel）
│   └── {market}_all_covariates_hourly.csv 全部 20 列，喂模型用
├── analysis/      相关性 + 稳健性结果（跨类型）
│   ├── p0_correlation_results.csv, p1_correlation_results.csv
│   ├── robust_correlation_results.csv      差分（去趋势）相关
│   ├── lag_peak_results.csv               滞后峰值（正 lag=协变量领先电价）
│   ├── covariate_collinearity.csv         各列 VIF（小时层）
│   └── covariate_collinearity_high.csv    高共线对（|r|>0.85）
└── model_ready/   按领域逻辑筛过的市场专属特征集（脚本10，喂模型起点，政策B 全滞后无泄漏）
    ├── {market}_features_hourly.csv           ← 滞后版（shift(24)，实测值滞后1天）
    ├── {market}_features_forecast_hourly.csv  ← 日前预报版（load/temp 用真实预报，wind/solar 用代理变量）
    └── {market}_features.txt                  保留理由 + 滞后明细 + 缺失说明
```

## 预报版 model_ready 文件

`{market}_features_forecast_hourly.csv` 是为解决"协变量泄漏"问题而生成的日前预报版本。原始预报数据存放在 `data/raw/forecasts/`：

```
data/raw/forecasts/
├── temperature/   Open-Meteo Historical Forecast API 日前温度预报
│   ├── {BA}_temp_forecast_hourly.csv
│   └── all_markets_temp_forecast_hourly.csv
├── load/          EIA Grid Monitor Day-Ahead Demand Forecast
│   ├── {BA}_load_forecast_hourly.csv
│   └── all_markets_load_forecast_hourly.csv
└── wind_solar/    Open-Meteo Historical Forecast（wind_speed_10m + shortwave_radiation，代理变量）
    ├── {BA}_wind_solar_forecast_hourly.csv
    └── all_markets_wind_solar_forecast_hourly.csv
```

预报版与滞后版的对应关系：load 列替换为 EIA 日前负荷预报，temperature 列替换为 Open-Meteo 日前温度预报，wind/solar 列替换为 Open-Meteo 风速/辐射代理变量。经济类协变量（气价/WTI/钢价/库存/风暴/发电结构）两版完全一致，均保持 shift(24) 或 shift(168)。

### 三种协变量使用场景

| 场景 | load/temp/wind/solar 来源 | 问题 |
|------|--------------------------|------|
| ① Oracle（泄漏） | 未来实测值 | 模型偷看未来，结果虚高 |
| ② 滞后实测（悲观） | shift(24) 昨日实测 | 昨日负荷/天气对明日电价信号极弱 |
| ③ 日前预报（公平） | EIA/Open-Meteo 日前预报 | 起报时已知的预报值，无泄漏且有预测信号 |

`{market}_features_hourly.csv` = 场景②；`{market}_features_forecast_hourly.csv` = 场景③。

### wind/solar 代理变量说明

wind/solar 目前使用 Open-Meteo 的 wind_speed_10m (m/s) 和 shortwave_radiation (W/m²) 作为代理变量，而非 ISO 官方 MW 预报。solar 代理相对可靠（辐射与光伏出力近似线性），wind 代理存在问题（风速与风电出力为非线性立方关系，且空间代表性不足）。

获取 ISO 官方风光 MW 预报需在外网机器（非公司网络）运行 `scripts/covariates/download_iso_wind_solar.py`（使用 gridstatus 库下载 ERCOT/CAISO 风光预报）。拿到后替换代理变量列即可。详见 `model_ready/FORECAST_README.md`。

## 数据源与抓取方式

| 类型 | 协变量 | 数据源 | 抓取方式 / URL | 免费? | 脚本 |
|---|---|---|---|---|---|
| gas | Henry Hub 气价 | FRED 序列 `DHHNGSP` | `fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP&cosd=...&coed=...`，`requests.get` 直拉 CSV，无 key | ✅ | 01 |
| gas | 天然气库存 | EIA 周报 | `https://www.eia.gov/dnav/ng/hist_xls/nw2_epg0_swo_r48_bcfw.xls`（直接下 XLS） | ✅ | 01 |
| oil | WTI 原油 | FRED 序列 `DCOILWTICO` | 同上 FRED CSV 接口，换 `id=DCOILWTICO` | ✅ | 02 |
| steel | ~~钢铁 PPI~~（legacy 已弃用） | FRED 序列 `PCU33113311` | 同上 FRED CSV 接口 | ✅ | 02（已不接管线） |
| steel | ~~钢铁生产指数~~（legacy 已弃用） | FRED 序列 `IPG3311A2S` | 同上 FRED CSV 接口 | ✅ | 02（已不接管线） |
| steel | **HRC 钢卷期货价（主用）** | CME HRC Futures (investing.com) | 手动下载历史数据 CSV（英为财经/investing.com），`02` 清洗中文表头/倒序/千分位 | ✅ | 手动下载 + 脚本02清洗 |
| storm | 风暴事件 | NOAA NCEI Storm Events | `ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/` 按年 gzip CSV，无 key | ✅ | 05 |
| generation_mix | 发电/燃料结构 | EIA U.S. Electric System Operating Data (EBA) | 已在 `data/raw/EIA/`（EBA.zip 解析）；源见 `data/raw/EIA/README.md` | ✅ | 06（读本地） |
| news | 能源新闻 | GDELT GKG v2 | `data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.gkg.csv.zip` 静态 zip，无 key、不限流 | ✅ | 08 |
| **forecast** | 日前温度预报 | Open-Meteo Historical Forecast API | `historical-forecast-api.open-meteo.com/v1/forecast`，无 key，归档 2022+ 起的预报 | ✅ | 11 |
| **forecast** | 日前负荷预报 | EIA Grid Monitor | `https://www.eia.gov/electricity/gridmonitor/knownissues/xls/{BA}.xlsx`，Excel 中 "Demand Forecast" 列 | ✅ | 12 |
| **forecast** | 风速/辐射代理 | Open-Meteo Historical Forecast API | 同温度 API，取 wind_speed_10m + shortwave_radiation | ✅ | 14 |

## 各协变量原始频率与对齐方式

| 协变量 | 原始频率 | 发布/更新时间 | 对齐到小时的方式 | 喂模型是否需要滞后 |
|--------|----------|---------------|------------------|-------------------|
| Henry Hub 天然气现货价 | 日频（工作日） | EIA 当日傍晚发布当日现货 | 日→小时 forward-fill | 需要 lag 1 天；**model_ready(脚本10) 已 shift(24) 应用** |
| 天然气库存 | 周频（按周末周五对齐） | 下周四上午 10:30 EST 才发 | 周→小时 forward-fill | ⚠ 按周末对齐会偷看 6 天（值到下周四才发布）；**model_ready 已 shift(168) 修复** |
| WTI 原油现货价 | 日频（工作日） | EIA 当日发布当日现货 | 日→小时 forward-fill | 需要 lag 1 天；**model_ready(脚本10) 已 shift(24) 应用** |
| ~~钢铁 PPI 指数~~（legacy 已弃用） | 月频（按月初对齐） | 次月中旬发布上月数据 | 月→小时 forward-fill | 粒度太粗、与电价近零相关 → **已用日频 HRC 替换** |
| ~~钢铁工业生产指数~~（legacy 已弃用） | 月频（按月初对齐） | 次月中旬发布上月数据 | 月→小时 forward-fill | 同上 → **已弃用** |
| **HRC 钢卷期货价（主用）** | 日频（交易日） | CME 当日收盘后 | 日→小时 forward-fill | 需要 lag 1 天；**model_ready(脚本10) 已 shift(24) 应用** |
| NOAA 风暴事件 | 日频（按事件日期聚合） | 事后归档（数周延迟） | 日→小时 forward-fill | ⚠ 事后记录，forecast 窗口内事件起报时未知；**model_ready 已 shift(24) 修复（用昨日风暴）** |
| EIA 燃料结构（发电量/份额） | 小时频 | 次日发布（T+1） | 原生小时，无需对齐 | ⚠ 实际发电 T+1，forecast 窗口用未来实际值=泄漏；**model_ready 已 shift(24) 修复（用 T-1 实际发电，日提前标准做法）**；严格无泄漏需发电预报 |
| GDELT 新闻 | 日频（按日聚合） | 当日可获取 | 日→小时 forward-fill（次日填充防泄漏） | 已在脚本(08)中做了次日填充，无泄漏 |
| **日前温度预报** | 小时频 | T-1 发布 T 日 24h 预报 | 原生小时，无需对齐 | 不需要滞后；**forecast版直接使用预报值，无泄漏** |
| **日前负荷预报** | 小时频 | T-1 发布 T 日 24h 预报 | 原生小时，无需对齐 | 不需要滞后；**forecast版直接使用预报值，无泄漏** |
| **风速/辐射代理预报** | 小时频 | T-1 发布 T 日 24h 预报 | 原生小时，无需对齐 | 不需要滞后；**forecast版直接使用预报值（代理变量，待替换为 ISO 官方 MW）** |

> 全部免费、无需 API key。各数据源说明：
> - **FRED** = Federal Reserve Economic Data，美国圣路易斯联储维护的免费经济时间序列数据库（气价/油价/钢价/工业指数等数千序列），拼序列号+日期即返回 CSV。
> - **NOAA** = National Oceanic and Atmospheric Administration，美国国家海洋和大气管理局，风暴/气象事件的权威来源。
> - **EIA** = U.S. Energy Information Administration，美国能源信息署，电力/天然气/能源数据的权威来源。
> - **GDELT** = Global Database of Events, Language and Tone，全球新闻事件+情感数据库，静态 zip 公开下载。
> - **Open-Meteo** = 开放气象数据平台，Historical Forecast API 归档 2022 年起的日前天气预报，无需 API key。
>
> 温度协变量见下方"注"，不在本目录抓取（滞后版）；预报版温度由脚本 11 从 Open-Meteo 下载至 `data/raw/forecasts/temperature/`。

## 模型使用方式

消融实验（`configs/parameter_ablation/ablation_A_covariates.yaml`）从 model_ready 读取协变量，当前配置 `covariates_source: model_ready` 指向滞后版（场景②）。如需使用日前预报版（场景③），将 `covariates_source` 改为 `model_ready_forecast` 或在 loader 中指向 `{market}_features_forecast_hourly.csv`。

消融实验使用的 8 个协变量（5 档递进）：

| 档位 | 协变量 | 数据来源（forecast 版） |
|------|--------|------------------------|
| 1 | 无协变量 | — |
| 2 | +load | EIA 日前负荷预报 ✅ |
| 3 | +temperature | Open-Meteo 日前温度预报 ✅ |
| 4 | +wind, +solar | Open-Meteo 代理变量 ⚠️（待 ISO 官方 MW 替换） |
| 5 | +henry_hub, +natgas_storage, +wti, +storm_event | shift(24)/shift(168) 滞后实测 ✅ |

## 优先级

P0（最高，已做但差分后信号下调）/ P1（钢铁/原油，已做但假设基本不成立）/ P2（风暴/燃料/新闻/interchange）。优先级体现在文件名注释，不体现在目录。详见 `docs/covariate_datasets_recommendation.xlsx` 与 `docs/todo2.md`。

## 复现

一键跑全流程（推荐）：

```bash
bash scripts/covariates/run_all.sh     # 下载→清洗→对齐→合并→相关→稳健性→model_ready
```

或分步：

```bash
# ── 原始协变量（01-09）──
python scripts/covariates/01_download_p0.py           # → gas/
python scripts/covariates/02_download_p1.py           # → oil/ + steel/
python scripts/covariates/03_align_to_hourly.py        # → merged/{market}_covariates_hourly.csv
python scripts/covariates/04_correlation_analysis.py   # → analysis/p0|p1_correlation_results.csv
python scripts/covariates/05_download_noaa_storm.py     # → storm/
python scripts/covariates/06_eia_fuel_mix_features.py  # → generation_mix/
python scripts/covariates/07_merge_all_covariates.py   # → merged/{market}_all_covariates_hourly.csv
python scripts/covariates/08_download_gdelt_news.py    # → news/
python scripts/covariates/09_robustness_analysis.py    # → analysis/robust|lag|collinearity

# ── model_ready 滞后版（10）──
python scripts/covariates/10_build_model_ready.py      # → model_ready/{market}_features_hourly.csv

# ── 日前预报版（11-15）──
python scripts/covariates/11_download_temp_forecast.py      # → data/raw/forecasts/temperature/
python scripts/covariates/12_download_load_forecast.py      # → data/raw/forecasts/load/
python scripts/covariates/14_download_wind_solar_forecast_proxy.py  # → data/raw/forecasts/wind_solar/
python scripts/covariates/15_build_forecast_model_ready.py  # → model_ready/{market}_features_forecast_hourly.csv

# ── ISO 官方风光预报（需外网机器）──
python scripts/covariates/download_iso_wind_solar.py       # → data/raw/forecasts/wind_solar/ (ERCOT+CAISO MW 预报)
```

> 注：脚本 13 (`13_download_ercot_wind_solar_forecast.py`) 因 ERCOT Incapsula 反爬虫保护无法运行，已被 `download_iso_wind_solar.py`（使用 gridstatus 库）替代。
> 脚本 `11_scrape_article_text.py` 为 GDELT 新闻文章正文抓取（非协变量主流程），与 `11_download_temp_forecast.py` 编号冲突但功能独立。

## 注

- 温度协变量（滞后版）：模型 loader 直接读 `data/raw/weather/processed/by_ba/{BA}_weather_hourly.csv`（原始数据，未并入本目录）。
- 温度协变量（forecast 版）：由脚本 11 从 Open-Meteo 下载至 `data/raw/forecasts/temperature/`。
- `{market}` ∈ {ercot, pjm, caiso, nyiso}；`{BA}` ∈ {ERCO, PJM, CISO, NYIS}（EIA Grid Monitor 代码）。
- 相关性图在 `analysis/figures/covariates/`（不在本目录）。
- 预报版详细说明见 `model_ready/FORECAST_README.md`。
