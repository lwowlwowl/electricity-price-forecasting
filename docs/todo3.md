# 模型搭建决策清单（todo3）

> 日期：2026-07-22
> 状态：待逐项确认
> 关联文档：w9/模型相关.md、w9/decision_aware_multimodal_tsfm_modeling.pdf、w8/研究方案.md

---

## 一、训练回路层面（端到端可微）

### 1. 策略 π 的实现方式 ⭐ 最高优先级

端到端可微的核心瓶颈：策略 π 把预测价格 p̂ 变成充放电动作 u，这一步涉及离散判断（电价高于阈值就放电），导数为零或不存在。∂u/∂p̂ 不可微，导致 L_bus 的梯度无法回传到 f_θ。

| 选项 | 方案 | 优点 | 缺点 | 依赖 |
|------|------|------|------|------|
| A | Greedy 规则 + 代理梯度 g=2Δt(u_ref-u*) | 实现简单，几十行代码 | 近似梯度，无理论收敛保证 | 无 |
| B | 凸优化层(LP/QP) + 扰动法 (Perturbed DFL) | 理论干净，有凸性和收敛证明 | 每 step 多一次 LP 求解 | cvxpy/Gurobi |
| C | SPO+ (Smart Predict-then-Optimize) | 类似 B，梯度计算方式不同 | 需要凸优化 oracle | Gurobi |

决策：先行版用 A，正式实验版用 B，A→B 消融对比写进论文。
参考论文：Perturbed Decision-Focused Learning (Columbia, 2024, IEEE Trans. Smart Grid, arXiv:2406.17085)

### 2. Loss 配比 α 和 β

L = αL_pred + βL_bus

| 选项 | 方案 | 说明 |
|------|------|------|
| A | 固定比例 α=0.8, β=0.2 | 简单，但可能不是最优 |
| B | 退火策略：前期 α=1,β=0 → 逐步 α→0,β→1 | 先热启动再切业务损失 |
| C | 纯 L_bus (α=0, β=1) | 老师可能的意思，但需要 π 可微 |

决策：用退火策略（B）。需跟老师确认"不要 MAE"的确切含义——是最终不要，还是完全不要。

### 3. L_pred 的具体损失函数

| 选项 | 说明 |
|------|------|
| MAE (L1) | 最简单 |
| Huber Loss | 对电价尖峰更鲁棒 |
| Quantile / Pinball Loss | 输出分位数预测，量化不确定性 |

决策：先行版用 Huber，正式版可换 Quantile Loss。如果最终 α→0，此项影响不大。

---

## 二、Encoder 层面

### 4. 各 Encoder 的基础结构 ⭐ 高优先级

老师原话(w9)："可以引入 GRU 来提高训练效率"，未指定加在哪个 Encoder。

| 选项 | 方案 | 参数量 | 适用场景 |
|------|------|--------|----------|
| A | 全部 Transformer (Self-Attn + FFN + RoPE) | 最大 | 表达力最强 |
| B | 全部 GRU | 约 A 的 1/4~1/3 | 快速验证 |
| C | 混合：Price/Load 用 Transformer，Weather/System 用 GRU，Calendar/News 用 Embedding/MLP | 中等 | 平衡精度和效率 |

决策：先行版全部 GRU（B），正式版混合（C），消融对比。
【待确认】问老师：GRU 只在某个 Encoder 上用，还是所有 Encoder 都可以先 GRU 起步？

### 5. Attention 机制：MHA vs MQA

老师提到 MQA (Multi-Query Attention)，目前标 [待确认]。

MHA：每个 head 独立 Q/K/V 投影，标准做法。
MQA：所有 head 共享 K/V，只 Q 独立。参数减少约 30%，推理更快。

决策：先用 MHA。模型才 ~100M 参数，MQA 省的参数和速度意义不大。优先级低。

### 6. 位置编码

| 选项 | 说明 |
|------|------|
| RoPE | TimesFM 使用，对相对位置和周期性建模好 |
| Sinusoidal | 原始 Transformer 做法 |
| Learnable PE | 可学习 |

决策：用 RoPE，跟 TimesFM 对齐。

---

## 三、Decoder 层面

### 7. Learnable Queries 数量

取决于预测窗口和数据粒度：
- 小时级数据 → 24 queries（24h 日前预测）
- 15 分钟级数据 → 96 queries

决策：跟数据粒度对齐。先确认数据是小时级还是 15 分钟级。

### 8. Decoder 结构

| 选项 | 方案 | 参数量 | 说明 |
|------|------|--------|------|
| A | DETR-like (Learnable Queries + Cross Attention + FFN) | 较大 | 能建模时间点之间的依赖 |
| B | 直接线性投影 p̂ = Linear(h_i) | 极小 | 简单但丢失时间交互 |

决策：先行版用 B（线性投影），正式版用 A（DETR-like），消融对比。

---

## 四、输出头层面

### 9. Head_DA 和 Head_RT 是否共享 Decoder

| 选项 | 方案 | 说明 |
|------|------|------|
| A | 共享 Decoder + 分叉 Head | w9 PDF 的设计，参数量小 |
| B | 两个独立 Decoder | 参数量翻倍，DA/RT 可学不同模式 |

决策：用 A（共享 Decoder + 分叉 Head），跟 w9 一致。

### 10. 额外输出头（Load / BESS 等）

老师(w9)："Decoder 的输出不止电价一种，可以扩展预测 BESS、Load 等。"

决策：第一轮只做电价预测（Head_DA + Head_RT）。跑通后再加 Load Head 做多任务学习。代码留好接口。

---

## 五、数据与训练层面

### 11. 上下文窗口长度

老师指定 7 天历史 (168h)。可尝试 3 天 / 14 天作为超参数调。
决策：先按 7 天。

### 12. 先跑哪个市场

可选：CAISO / ERCOT / NYISO / PJM。
建议：ERCOT（波动最大，decision-aware 优势更明显）。

### 13. 参数配置

| 版本 | d_model | n_layers | 参数量 | 训练时长(M3 Pro) |
|------|---------|----------|--------|-----------------|
| 先行版 | 256 | 2 | ~15M | 几小时 |
| 正式版 | 512 | 4 | ~100M | 数天或云 GPU |

决策：先行版跑通验证，再上正式版。

---

## 六、需要问老师确认的清单

1. GRU 具体用在哪些 Encoder？还是所有都可以先 GRU 起步？
2. 日内任务是"复用架构自己训练"还是"复用日前模型只微调 Head"？
3. "不要 MAE"的确切含义——最终以 L_bus 为主，还是完全不用任何预测损失？
4. MQA 是否需要在第一版就引入？
5. 数据粒度确认：小时级还是 15 分钟级？
6. 对比实验的具体基线清单：TimesFM / Chronos / Moirai / Toto，还有别的吗？
