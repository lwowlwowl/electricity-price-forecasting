# todo2 — 协变量进度盘点（按 covariate_datasets_recommendation.xlsx 分类）

> 日期：2026-07-14 初版 / 2026-07-15 稳健性修订
> 依据：`docs/covariate_datasets_recommendation.xlsx`（三张表）+ 文件系统核查 + 相关性数值 + 稳健性分析（`scripts/covariates/09_robustness_analysis.py`）
> 产物：`data/covariates/analysis/` 下 `p0/p1_correlation_results.csv`、`robust_correlation_results.csv`、`lag_peak_results.csv`、`covariate_collinearity.csv`、`covariate_collinearity_high.csv`；协变量数据已重构到 `data/covariates/{gas,oil,steel,storm,generation_mix,news,merged,analysis}/`（按类型分，与 `data/raw/` 分开）
> 与 `docs/TODO.md` 同级，跟踪协变量这条线。

---

## 〇、汇报口径（今天直接用）

**主结论**：建好并对齐了全部 P0/P1 经济协变量（气价/库存/WTI/钢铁），并做了稳健性检验。**关键发现是反直觉但重要的**：经济协变量与电价的水平相关（气价 0.29–0.62）**在差分（去趋势）后几乎全部归零、不显著**——说明那些相关是 2025–2026 共同趋势/季节性撑出来的，不是日频上的真实联动。因此日频预测意义上的有效协变量仍是高频的负荷/温度/风光（即现有消融 `ablation_A_covariates` 已在测的那套），经济协变量的**模型级增益待 Step3 验证，预期有限**。

**可以说**：气价在水平上与四市场电价正相关、且领先 1–2 天；钢铁对 PJM 近零；做了差分/共线性稳健性检验；冗余已识别并给出精简集。

**别说过头**：不要说"气价四市场通用且验证有效"（ERCOT Spearman 仅 0.02、差分后全 ns）；不要把钢铁结论讲死（用的是月度代理，粒度限制）；不要把水平相关当预测能力。

---

## 一、按 xlsx 协变量优先级（Sheet 1）

### P0 — 已做，但稳健性后结论下调 ⚠️

| 协变量 | 下载+对齐 | 水平相关（日 Pearson / Spearman） | 差分后（去趋势） |
|---|---|---|---|
| Henry Hub 气价 | ✅ | ERCOT 0.44/**0.02**；PJM 0.54/0.34；CAISO 0.29/0.36；NYISO 0.62/0.57 | **全部 ns、≈0**（ERCOT -0.03、PJM 0.01、CAISO 0.01、NYISO 0.03） |
| 天然气库存 | ✅ | CAISO 0.46/0.51 稳健；ERCOT 0.12/0.25；PJM 0.01/**ns**；NYISO 0.07/0.13 | **全部 ns、≈0** |

> ERCOT 气价 Pearson 0.44 vs Spearman 0.02 的巨大缺口 = 水平相关被异常值/趋势主导，不稳健。差分后所有市场归零，坐实"趋势伪相关"判断。

### P1 — 已做，假设基本不成立，且差分后无信号 ⚠️

| 协变量 | 实现 | 水平相关（日） | 差分后 |
|---|---|---|---|
| 钢价（FRED PPI 月度，非 CME 期货） | ⚠️ 替代 | PJM 0.003/**ns**；CAISO -0.45；NYISO -0.17 | ns、≈0（仅 ERCOT PPI diff +0.10*） |
| 粗钢产量（FRED 工业生产指数月度，非 AISI 周度） | ⚠️ 替代 | PJM 0.021/**ns** | ns、≈0 |
| WTI 原油 | ✅ | 弱且负：CAISO -0.42；余 -0.08~-0.18 | ns、≈0 |

> **钢铁粒度 caveat**：用 FRED 月度 PPI/IP 作代理，粒度太粗；近零相关可能部分源于此，AISI 周度需注册未取。**论文里当"已知限制"写，别讲成"钢对电价无影响"的定论。**

### P2 — 新闻半成品（降权，见第八节）；interchange 已有

| 协变量 | 状态 |
|---|---|
| LLM 新闻情绪 | 🟡 GDELT raw 半（到 2025-08-03，目标 2026-06-02；1/96 抽样；用 tone 非 LLM；未对齐未进模型） |
| RGGI 碳价 | ❌ 未做（可选） |
| 区域间输电潮流 | ✅ 已有（EIA csv） |

### 额外做的（xlsx 未列，已并入小时表）

- NOAA 风暴事件 ✅（`storm_*` 列，脚本 05）
- EIA 燃料构成/发电份额 ✅（`gas_share`/`renewable_share`/`renewable_shock`，脚本 06）
- 四市场小时合并表 ✅ `*_all_covariates_hourly.csv`

---

## 二、稳健性三块硬货（本次新增，汇报材料核心）

### 1. 差分/去趋势相关（硬伤 A 的回答）— `robust_correlation_results.csv`

去趋势（first-difference）后，**P0/P1 全部市场全部协变量的日频相关几乎归零、不显著**。结论：水平相关是共同趋势伪信号，**日频上经济协变量与电价无真实联动**。
> 注意 caveat：日电价差分极噪，该检验偏严，可能低估慢频信号；但举证责任现在反过来——要证明它们有用。

### 2. 滞后峰值表 — `lag_peak_results.csv`

正 lag = 协变量领先电价。**气价在四市场一致领先 1–2 天**（ERCOT +2d r0.58、PJM +2d r0.68、CAISO +1d r0.31、NYISO +1d r0.72），库存领先 6–7 天（慢变量），WTI/钢铁无有意义领先结构。
> 注意：这些是**水平**滞后相关，同样受趋势通胀，配合第 1 条谨慎读。

### 3. 共线性/冗余 — `covariate_collinearity.csv` + `covariate_collinearity_high.csv`（小时层）

| 冗余类型 | 证据 | 处理 |
|---|---|---|
| 风暴计数对 | `storm_event_count ~ storm_high_impact_count` r=0.94–0.99 | 留 1 个 |
| 份额互补 | `gas_share ~ renewable_share` r=-0.97 | 留 1 个 |
| 发电求和约束 | `gas_gen+wind+solar ≈ total_gen`（NYISO solar VIF=inf） | 留 total 或留各燃料，勿全留 |
| 比值依赖 | `gas_share ≡ gas_gen/total_gen`（定义层，VIF 抓不到） | 留份额或留绝对量，勿全留 |
| 慢变量同趋势 | steel/storage/wti 互 VIF>25 | 与差分结果互证：都是趋势 |

**建议精简集（喂模型前）**：`henry_hub`、`storm_event_count`（或 damage）、`total_gen_mwh`、`gas_share`、`wind_gen`、`solar_gen`、`renewable_shock`（+ load/temp/天气那套）。

---

## 三、按 xlsx 执行计划（Sheet 2）

| 步骤 | xlsx | 实际 |
|---|---|---|
| 第1步 气价+库存 | 待开始 | ✅ 完成（含相关性 + 差分稳健性） |
| 第2步 钢铁+WTI | 待开始 | ✅ 完成（AISI→FRED 月度替代） |
| 第3步 **喂进模型+消融** | 待开始 | ❌ **未做**——`ablation_A_covariates` 只测 `load/temp/wind/solar` 且只 ERCOT；经济协变量没接进 `loader.py load_slice` |
| 第4步 新闻 | 待开始 | 🟡 raw 半成品 |

---

## 四、方法硬伤 B：泄漏（✅ 已全修，model_ready 无泄漏）

`03_align_to_hourly.py` 原按"参考期"对齐：日频现货同日 ffill、周度库存按周末周五对齐（下周四才发布）、月度 PPI 按月初对齐（次月中旬才发）、NOAA 风暴事后记录、实际发电 T+1——**forecast 窗口内起报时刻未发布值 = 泄漏**（做相关性描述无妨，喂模型预测必泄漏）。

**已修（`scripts/covariates/10_build_model_ready.py`）**：model_ready 全部特征按发布频率滞后——日频现货/风暴/实际发电及派生 `shift(24)`、周度库存 `shift(168)`、月度 PPI `shift(1440)`。forecast 窗口只用起报时刻已发布值。各档滞后已验证（`new[i]==old[i-lag]`）。残留与泄漏无关的 EIA 发电源数据缺口（~29h outage ≈0.24%），Step3 时按缺失行跳过/插补。

> ⚠ **weather（温度）未审**：在 `data/raw/weather/`，由 `loader.py` 直接读，不在协变量流水线。若喂的是实测温度，forecast 窗口同泄漏，需改用温度预报或滞后——Step3 接 loader 时一并查。

---

## 五、Step3 评估与路径（报告后第一件事）

- **当前断点**：消融协变量加载在 `src/data_processing/loader.py` 的 `load_slice`，**只认 `load/wind/solar/temperature`**；经济协变量在 `*_all_covariates_hourly.csv` 里但**没接进加载路径**。
- **要做的**：① 改 `load_slice`（或加 model_ready 分支）让协变量消融从 `data/covariates/model_ready/{market}_features_hourly.csv` 读**全部**协变量（含 **lagged load/temperature/wind/solar**，政策B），而非从市场+天气直接读未滞后实测——否则 load/temp 仍是 oracle、和经济协变量不自洽；② `ablation_A_covariates.yaml` 的 `ablate.values` 加经济协变量组合档；③ 重跑（ERCOT，Chronos2，12 个月连续回测，数小时）。
- **已就绪**：model_ready 已是统一无泄漏全集（经济/风暴/发电按频率滞后 + load/temp/wind/solar 全 shift(24)，PJM/NYISO 跳过 wind/solar），政策B 数据层已落地验证。
- **判断**：今天报告前不动 loader（怕弄坏能跑的消融）；**差分结果已预示增益有限**，Step3 更像"确认性负结果"。作为报告后第一个工程任务。

---

## 六、冗余清理建议（喂模型前必做）

1. `storm_event_count`/`storm_high_impact_count`/`storm_damage_usd`/`storm_injuries` → 留 **count + damage** 两列。
2. `gas_share` 与 `gas_gen/total_gen` 三者 → 留 **gas_share**（份额）或留绝对量，二选一。
3. `gas_share`+`renewable_share` → 留一（近乎互补）。
4. `total_gen` 与各燃料 → 留 total 或留各燃料，勿全留。

---

## 七、待办（按优先级）

1. **Step3**：扩 `loader.load_slice` 接经济协变量 + 气价/WTI 滞后 1 天 + 加消融组合 + 重跑 ERCOT。（报告后）
2. **冗余清理**：按第六节产出精简集，喂模型用精简版。
3. **钢铁**：若要强论断，取 AISI 周度（需注册）；否则按"月度代理、粒度限制"写。
4. **新闻**（最低优先级，见下）。

---

## 八、新闻（降权块）

W7 要 (b)（LLM 读新闻正文），但国内网络卡 `infini-news-corpus`（`ruggsea/infini-news-corpus`，有正文、按年分片、覆盖 2025-2026，但 `vblagoje/cc_news` 排除因仅 2017-2019）。GDELT raw 半成品有 1/96 抽样 + 用 tone 非 LLM 两个问题。**报告前不投入**，作为后续 (b) 路线，需 AutoDL 或海外节点。
