# 实验 TODO — 正式评估流程改造

> 最后更新：2026-07-04
>
> 依据：Lago et al. (2021) 13 条 checklist + Weron (2014) EPF 评估准则
>
> 原则：先改底层共用组件，再改各模块配置和流程，最后重跑实验

---

## 整体现状

实验分三大模块（参数消融、结构消融、模型融合），共享底层数据加载（`loader.py`）、回测引擎（`backtest.py`）和指标体系（`metrics.py`）。

当前问题：三个模块都围绕"ERCOT 单市场、W1/W2/W3 三个独立月份、SMAPE 为主指标"设计，而 Lago/Weron 要求"多市场、连续测试期 ≥ 1 年、rMAE 为主指标 + DM/GW 统计检验"。

数据现状：四个市场（ERCOT 15 节点、PJM 35 节点、CAISO 3 节点、NYISO 11 节点）均有 2025-01-01 ~ 2026-06-02 共 17 个月小时级数据。

时间划分方案：
- train（spike head）: 2025-01-01 ~ 2025-05-31（5 个月）
- validation: 2025-06-01 ~ 2025-06-30（1 个月）
- test: 2025-07-01 ~ 2026-06-01（12 个月连续）

---

## 一、底层共用组件改动

### [x] 1.1 metrics.py：新增 rMAE + 统一 naive 基线口径

**对应 Lago checklist #6**

目前有 MAE、RMSE、SMAPE、MASE、pinball、coverage、spike_f1。缺少 rMAE = MAE_model / MAE_naive7（naive7 = 7 天前同时刻价格 p_{d-7,h}，即 period=168）。

⚠️ **naive 基线口径问题**：当前 backtest.py 的 MASE 分母来自 SeasonalNaive（period=24，日周期），而 Lago 的 rMAE 分母是 naive2（period=168，周周期）。这不仅是新增一个函数的问题，需要确保整个评估管线的 naive 基线口径一致：
- rMAE 分母：naive2（period=168，即 p_{d-7,h}）← Lago 定义
- MASE 如果保留为辅助指标：明确标注用的是日周期（period=24）还是周周期（period=168）

改动：新增 `rmae()` 函数 + 在 backtest.py 中额外计算 naive2（period=168）的 MAE 作为 rMAE 分母，在 `all_point_prob_metrics()` 里添加 `rmae` 输出。

⚠️ **数值稳定性**：当某些时段电价极低（如 CAISO 负电价占 14.7%），naive7 的 MAE 可能趋近于零，导致 rMAE 爆炸。需要对分母设 floor（如 max(MAE_naive7, 1.0)），或在论文中说明"负价格市场的 rMAE 解读需谨慎"。

### [x] 1.2 新建 stat_tests.py：DM test 和 GW test

**对应 Lago checklist #7 + Weron Section 4.5.2**

完全没有统计检验。需要实现：

(a) **单变量 DM test**：对每个小时 h=1..24 独立做 t 检验，输入是两个模型的逐日 loss 差序列 d_h(t) = |e_A(t,h)| - |e_B(t,h)|，用 Newey-West HAC 标准误

(b) **多变量 GW test**：日度 L1 loss differential Δd = Σ_h |e_A(d,h)| - Σ_h |e_B(d,h)|，用 HAC 协方差估计做检验

输出 p-value 矩阵 + 热力图（参考 Lago Fig.5 格式）。

⚠️ **DM vs GW 的适用场景区分**：
- **参数消融 / 结构消融**（消融配置之间对比）：用 **DM test**。因为模型参数在整个测试期固定不变（zero-shot），不满足 GW test 的条件假设（要求模型在每个时点使用 t-1 信息 recalibrate）
- **ElecFM vs LEAR**：用 **GW test**。两者都有某种程度的 recalibration（LEAR daily recalibration，ElecFM 如做 4g 也有更新）
- 论文方法部分需加一句说明 GW 的条件假设及为何 zero-shot 实验退而使用 DM

### [x] 1.3 backtest.py：支持 naive7 MAE 计算

当前 `_summarize` 只算 SeasonalNaive（period=24）MAE 作为 MASE 分母。rMAE 需要不同的分母：naive2（period=168），即对每个起报点每个时刻，naive7 预测 = 168 步前的实际价格。

这不只是"回查数据"——需要确保 backtest 引擎在每个 origin 都正确计算 naive2 的预测和 MAE，并传递给 metrics.py 的 rMAE 计算。具体需要在 `_predict_all_origins` 中额外收集 naive2 预测序列。

### [x] 1.5 多市场数据质量审计

在跑多市场实验前，先对 PJM/CAISO/NYISO 做一轮数据质量检查：
- 缺失率和缺失模式（是否有整天缺失）
- 负价格比例和分布
- spike 频率和幅度
- 协变量覆盖（PJM/NYISO 只有 load，无 wind/solar）
- 市场结构差异（PJM 有 hub+zone 两类节点，CAISO 只有 3 个 trading hub）

输出：每市场一份数据概况表，确认节点选择合理性。

### [x] 1.4 nodes.yaml：扩展到多市场

**对应 Lago checklist #5**

目前只有 ERCOT 节点分组。扩展方案按用途区分：

**消融实验（参数消融 + 结构消融）**：每市场选 2-3 个代表节点，控制算力

| 市场 | 代表节点（建议） |
|------|-----------------|
| ERCOT | HB_HOUSTON, LZ_WEST, HB_HUBAVG（已有） |
| PJM | HUB:WESTERN HUB, HUB:CHICAGO HUB, ZONE:PJM-RTO |
| CAISO | TH_SP15_GEN-APND, TH_NP15_GEN-APND |
| NYISO | N.Y.C., WEST, NORTH |

需要先跑一次各市场价格统计，确认代表性节点选择合理（高波动/多负价格/稳定各一个）。

**融合模型（ElecFM）训练 + 测试**：每市场全部节点

| 市场 | 节点数 | 用途 |
|------|--------|------|
| ERCOT | 15 | 全部用于训练和测试 |
| PJM | 35 | 全部用于训练和测试 |
| CAISO | 3 | 全部用于训练和测试 |
| NYISO | 11 | 全部用于训练和测试 |

节点越多训练数据越多，spike head 学得越好；全节点测试也更有说服力，不存在挑节点嫌疑。

---

## 二、参数消融模块改动

**现状**：ERCOT volatility 节点，W1(8月)/W2(3月)/W3(1月) 三窗口，max_origins=30，消融 A(协变量)/B(context)/C(多变量)/D(horizon)/F(频率)。

### [x] 2a. 测试期改为连续 12 个月

**对应 Lago checklist #1**

当前每个消融 3 个 YAML（baseline.yaml / _w2 / _w3），每个只测一个月。改为：
- 合并成 1 个配置文件
- test_start=2025-07-01, test_end=2026-06-01
- data_start 前移到 2025-01-01（给树模型留够 train 窗口）
- 去掉 max_origins 限制（跑满 ~335 个起报点）

⚠️ 所有参数消融需要重跑。好消息是消融不需要训练（zero-shot 或 per-origin 重训），只是推理时间更长。

### [x] 2b. 消融配置支持多市场

每个消融配置需要市场参数化。最简做法：在 `run_experiment.py` 加 `--market` 参数循环遍历，或写一个顶层 shell 脚本串联调用。

### [x] 2c. 输出报告加 rMAE 列 + 消融配置间 DM test

summary.csv 当前有 mae_mean、rmse_mean、smape_mean、mase_mean。底层 metrics.py 改好后，上层自动增加 rmae_mean。

另外，消融配置之间（如 baseline vs 去掉协变量、baseline vs 缩短 context）同样需要 DM test 验证差异显著性，与结构消融 3c 标准保持一致。

### [x] 2d. 加 LEAR baseline

**对应 Lago checklist #2 + #13**

models 列表加上 LEAR（来自 `epftoolbox`）。需要在 `forecasters.py` 包装 `LEARForecaster`。LEAR 每天用 LASSO 自动 recalibrate（前 N 天滚动窗口），跑得很快（1-10s/天），完美符合 Lago daily recalibration 要求。

⚠️ **LEAR 在三种实验中的定位不同**：
- **参数消融 / 结构消融**：LEAR 是“外部定标参照”，不是竞争对手。帮助说明“截断 5 层后精度是否仍比 LEAR 好”。论文表格中 LEAR 单独一行标注“external reference (daily recalibration)”，不与消融配置并列排名
- **ElecFM**：LEAR 是必须超越的基准，公平竞争、DM/GW test 对比

### 不需要改的

消融 A/B/C/D/F 的实验逻辑本身不用改，只改配置文件的 test window 和市场。backtest.py 的 walk-forward 设计无泄漏，不需要改。

---

## 三、结构消融模块改动

**现状**：ERCOT volatility 节点、W1(8月) 单窗口，对 TimesFM/Chronos2/Toto2 做 11 种结构消融 + perlayer 逐层分析。

### [x] 3a. 测试期改为连续

full_timesfm.yaml 等配置的 test_start/test_end 改为连续测试期（同参数消融）。

### [x] 3b. 多市场验证

核心发现（如"TimesFM 移除 5 层 SMAPE 基本不变"）需在其他市场验证。至少在 PJM 上重复完整结构消融，确认跨市场普适性。若发现仅在 ERCOT 成立，论文中诚实说明。

### [x] 3c. 加 rMAE + DM test

结果表加 rMAE 列。关键对比（baseline vs truncate_front_half 等）做 DM test 验证显著性。目前只有 MAE 数值差和百分比变化，审稿人会问"差异统计显著吗？"

### 不需要改的

skip_attention、halve_heads、truncate_front/back、skip_layer 等消融类型设计合理（论文核心贡献之一），只需在更严格评估框架下重跑。

---

## 四、模型融合（ElecFM）模块改动

**现状**：ERCOT 15 节点训练 spike head 各版本，W1/W2/W3 三窗口评估，输出 SMAPE + spike-F1。

**这是改动最大的模块。**

### [x] 4a. dataset.py 时间划分重构

当前围绕 W1/W2/W3 设计——排除 3 个测试月 + buffer，剩余做训练。改为简单三段切割：

- train: 2025-01-01 ~ 2025-05-31
- validation: 2025-06-01 ~ 2025-06-30
- test: 2025-07-01 ~ 2026-06-01

删掉 `EXCLUDED_RANGES`，不再需要"哪些月份排除"的复杂逻辑。

### [x] 4b. evaluate.py 改为连续测试期

`TEST_WINDOWS = {"w1_stable": ..., "w2_negative": ..., "w3_extreme": ...}` 改为 `TEST_PERIOD = ("2025-07-01", "2026-06-01")`。

W1/W2/W3 保留为子分析（论文展示不同市场状态下的表现差异），但主结果必须基于连续 12 个月汇总。

### [x] 4c. 评估指标扩展

evaluate.py 目前输出 SMAPE/pinball/spike-F1。改为：
- **主指标**：rMAE（价格预测）+ spike-F1(head)（尖峰检测）
- **辅助指标**：MAE、SMAPE、pinball（兼容旧结果）

### [x] 4d. 加 LEAR baseline 对比

**对应 Lago checklist #2**

ElecFM 评估结果需和 LEAR baseline（daily recalibration）做对比。在 evaluate.py 或新对比脚本里，LEAR 在同一连续测试期上的 rMAE/MAE/spike-F1 一并报告，并做 DM test。

### [ ] 4e. 多市场评估

ElecFM spike head 在 ERCOT 训练。对其他市场两种做法：

**(a) zero-shot transfer**：ERCOT 训练的 spike head 直接在 PJM/CAISO/NYISO 上测，展示迁移能力

**(b) per-market retrain**：每个市场各训练一个 spike head，展示上限

建议两种都做。

### [x] 4f. 记录计算成本

**对应 Lago checklist #3**

在 backtest 和 evaluate 中添加计时，统计 ElecFM 的 per-day inference time，与 LEAR/DNN 对比：

| 模型 | Lago 基准 |
|------|-----------|
| LEAR | 1-10s/天 |
| DNN | 2-5min/天 |
| ElecFM | 待测 |

### [ ] 4g. 每日 recalibration 的 spike head（需论证）

**对应 Lago checklist #8**

Lago 原文要求 "models should be recalibrated on a daily basis"，Weron Section 3.8.4 也提到 ARX 等常用模型通常配有 daily recalibration。当前 ElecFM 在 12 个月连续测试期内完全不更新，这需要在论文中**明确讨论并提供实证**，不能简单跳过。

**两种方案**：

**(a) 实现 daily recalibration**：每天用截至前一天的全部历史对 spike head 做 1-2 步在线更新（backbone 冻结不动）。工作量 ~4h。

**(b) 不做 recalibration，但提供实证论证**：
- 在 12 个月测试期上，对比前 6 个月 vs 后 6 个月的 rMAE 和 spike-F1
- 如果后 6 个月无显著退化 → argue "fixed spike head 足够泛化，因为 backbone 的表征是 pretrained 的、稳定的"
- 如果后 6 个月有退化 → 必须实现方案 (a)
- 工作量 ~1h（纯分析）

建议先做 (b) 的分析，根据结果决定是否需要 (a)。

### 不需要改的

train.py 两阶段训练逻辑（Stage 1 冻结底层训 spike head → Stage 2 解冻精调）合理。spike_head_only / cross_node_only 设计没问题。只要 dataset.py 时间划分改好，训练代码基本不动。

---

## 五、Lago 2021 Checklist 对照

| # | 要求 | 当前状态 | 改动项 |
|---|------|---------|--------|
| 1 | 测试期 ≥ 1 年 | ❌ 3 个独立月 | 2a/3a/4b |
| 2 | 对比开源 SOTA 模型 | ❌ 无 LEAR/DNN | 2d/4d |
| 3 | 评估计算成本 | ❌ 未记录 | 4f |
| 4 | 数据集开源 | ⚠️ ERCOT 公开数据 | 论文说明 |
| 5 | 多市场 | ❌ 仅 ERCOT | 1.4/2b/3b/4e |
| 6 | 使用 rMAE 指标 | ❌ 无 rMAE | 1.1/2c/3c/4c |
| 7 | 统计检验 | ❌ 无 DM/GW | 1.2/3c |
| 8 | 每日 recalibration | ⚠️ 骨干零样本 | 4g（需论证） |
| 9 | 验证集 ≠ 测试集 | ✅ 已分离 | — |
| 10 | 明确数据划分和日期 | ❌ dataset.py 逻辑需重构 | 4a（代码改动）+ 论文说明 |
| 11 | 明确所有输入 | ⚠️ 需补充 | 论文说明 |
| 12 | 测试期 = 数据最后一段，无重叠 | ❌ W1/W2/W3 交叉 | 4a/4b |
| 13 | 使用开源工具做 baseline | ❌ 无 epftoolbox | 2d |

---

## 六、工作量与优先级

### P0（方法论合规，审稿人必问）

| 改动 | 涉及文件 | 工作量 |
|------|---------|--------|
| 1.1 metrics.py 加 rMAE | 1 文件 | 0.5h |
| 1.2 stat_tests.py 实现 DM/GW | 新建 1 文件 | 2-3h |
| 2a/3a/4b 测试期改连续 12 个月 | ~30 YAML + dataset.py | 2h 改配置 + 重跑 |

### P1（比较公平性 + 多市场）

| 改动 | 涉及文件 | 工作量 |
|------|---------|--------|
| 1.4 nodes.yaml 扩展多市场 | 1 文件 | 0.5h |
| 2d/4d 加 LEAR baseline | forecasters.py + 1-2 文件 | 2h |
| 4a dataset.py 时间划分重构 | 1 文件 | 1h |

### P2（结果完整性）

| 改动 | 涉及文件 | 工作量 |
|------|---------|--------|
| 1.3 backtest.py 支持 naive7 | 1 文件 | 1h |
| 1.5 多市场数据质量审计 | 新建脚本 | 1h |
| 2b/3b 多市场重跑消融 | 配置 + 算力 | 算力为主 |
| 4f 记录计算成本 | backtest/evaluate | 0.5h |

### P3（创新点增强）

| 改动 | 涉及文件 | 工作量 |
|------|---------|--------|
| 4e 多市场 transfer 评估 | evaluate.py 扩展 | 3h |
| 4g-b spike head 退化分析（前6月 vs 后6月） | evaluate.py | 1h |
| 4g-a spike head daily recalibration（如退化显著） | train.py 扩展 | 4h |

---

## 七、关键文件路径

| 内容 | 路径 |
|------|------|
| 指标体系 | `src/evaluation/metrics.py` |
| 回测引擎 | `src/evaluation/backtest.py` |
| 数据加载 | `src/data_processing/loader.py` |
| 参数消融入口 | `src/parameter_ablation/run_experiment.py` |
| 结构消融入口 | `src/structural_ablation/run_structural_ablation.py` |
| 融合模型入口 | `src/fusion_model/run_fusion.py` |
| 融合训练 | `src/fusion_model/train.py` |
| 融合评估 | `src/fusion_model/evaluate.py` |
| 融合数据集 | `src/fusion_model/dataset.py` |
| 节点配置 | `configs/nodes.yaml` |
| 参数消融配置 | `configs/parameter_ablation/` |
| 结构消融配置 | `configs/structural_ablation/` |
| 融合模型配置 | `configs/fusion/` |
| 实验结果 | `data/results/` |
| Checkpoint | `data/checkpoints/` |
| 参考论文 | `docs/reference/main.pdf`（Lago 2021）+ `docs/reference/old_main.pdf`（Weron 2014）|
