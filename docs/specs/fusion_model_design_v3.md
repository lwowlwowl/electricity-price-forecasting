# 融合模型设计文档 v3.0

> 版本：v3.0 ｜ 日期：2026-06-30
> 前置文档：`experiment_manual.md`（v1.0 参数消融）、`experiment_manual_v2.md`（v2.0 结构消融）
> 状态：**设计完成，待实现**

---

## 0. 为什么现在可以写这份文档

v2.0 结构消融手册在 Section 4.2 留了一个占位符：

> "融合架构的具体方案需要等消融结果出来后再确定。"

现在消融全部完成（36 次组件级 + 32 次逐层），结论足够给每一个融合决策点提供定量依据。本文档把消融结论转化为**具体的、可实现的模型架构**。

---

## 1. 消融结论 → 设计决策映射

| v2.0 提出的决策点 | 消融依据 | 本次决策 |
|---|---|---|
| **骨干选择** | TimesFM 基准 SMAPE 最低（27.67 vs 29.92/30.06）| ✅ TimesFM-2.5 为骨干，迁移学习微调 |
| **FFN 类型** | 三模型移除 FFN 后退化最大（TimesFM +486%，Toto +419%）| ✅ 保留 TimesFM 原有 Swish MLP（不替换 SwiGLU，保留预训练权重）|
| **位置编码** | xPos 衰减完全冗余（Toto ΔF1≈0%）；TimesFM RoPE 轻度依赖（+6%）| ✅ 保留 TimesFM 原有标准 RoPE；xPos 不引入 |
| **跨变量机制** | 变量注意力贡献存在但非决定性（+7%~+22%）；TimesFM 本无此机制 | ✅ **不引入**；Q2 决策 |
| **层深度** | 20 层中 8 层严格安全（Δ < 3%）；L17–L19 较敏感（+6%–9%）| ✅ 20→12 层，移除 8 个冗余层 |
| **输出头** | 简化输出头后 TimesFM SMAPE +88%（复杂头必要）| ✅ 保留 quantile head；新增 spike head |
| **精度 vs 尖峰通路** | L7 是纯尖峰检测层（ΔSMAPE +1.6%，ΔSpike-F1 −6.7%）| ✅ spike head 从 L7 对应位置引 skip connection |
| **参数归一化** | 移除 LayerNorm 后 CRASH（TimesFM 数值稳定的生死线）| ✅ 保留 TimesFM 原有 pre+post LayerNorm |

---

## 2. 架构总览

### 命名

**ElecFM**（Electricity Foundation Model）——基于 TimesFM-2.5 的电价预测专用模型。

### 数据流

```
原始电价序列（单节点，单变量）
  → Tokenizer（ResidualBlock，32 时间步 → 1280 维 patch 嵌入）
    → L0（原 TimesFM L0，CRITICAL：SMAPE +71.6%）
    → L1（原 L1，CRITICAL：SMAPE +40.7%）
    → L2（原 L2）
    → L3（原 L3，spike 敏感：ΔF1 −5.5%）
    → L4（原 L5）
    → L5（原 L7，"尖峰检测层"：ΔF1 −6.7%，ΔSMAPE +1.6%）
         │
         ├──► [SPIKE HEAD BRANCH]
         │         Linear(1280→24) → sigmoid
         │         → spike_prob[0..23]（24 步各时刻尖峰概率）
         │
    → L6（原 L10）
    → L7（原 L11）
    → L8（原 L14）
    → L9（原 L17）
    → L10（原 L18）
    → L11（原 L19）
         │
         └──► [PRECISION HEAD]
                  output_projection_quantiles（原 TimesFM quantile head）
                  → q[0.1..0.9] × 24 步
```

### Step 1 验证结果与实际剪枝决策

**原计划（8 层）验证结果**：同时移除 {L4,L6,L8,L9,L12,L13,L15,L16} 后，SMAPE 退化 **+18%**（超过 10% 阈值）。单层独立消融实验的累积效应远大于预期，8 层方案不可用。

**实际采用：5 层保守方案**，仅移除独立测试时 |ΔSMAPE| < 1% 的最安全层：

| 原 TimesFM 层号 | ΔSMAPE（单层移除） | ΔSpike-F1 | 移除理由 |
|---|---|---|---|
| L6 | +0.6% | +0.4% | 最安全，双指标均安全 |
| L8 | ≈0.0% | +0.6% | 最安全，双指标均安全 |
| L9 | +0.8% | −1.0% | 双指标均安全 |
| L13 | +0.3% | +0.2% | 最安全 |
| L15 | +0.8% | +3.0% | 双指标均安全（ΔF1 正向） |

**Step 1 零样本验证（5 层方案）**：SMAPE = 28.90 vs baseline 27.67 → 退化 **+4.4%** ✅ 通过

### 保留的 15 层（新编号 → 原 TimesFM 编号）

| 新层号 | 原层号 | 保留理由 |
|---|---|---|
| L0 | L0 | CRITICAL：SMAPE +71.6%，Spike-F1 −7.7% |
| L1 | L1 | CRITICAL：SMAPE +40.7% |
| L2 | L2 | SMAPE +3.3% |
| L3 | L3 | spike 敏感：ΔF1 −5.5% |
| L4 | L4 | SMAPE +0.9%，ΔF1 −2.9% |
| L5 | L5 | SMAPE −0.7% |
| **L6** | **L7** | **spike 关键层**：ΔF1 −6.7%；spike head 从此处分叉 |
| L7 | L10 | 保留（|ΔSMAPE|<1% 但累积效应未验证） |
| L8 | L11 | SMAPE +1.0%，ΔF1 +4.8% |
| L9 | L12 | SMAPE +1.1% |
| L10 | L14 | SMAPE +3.3% |
| L11 | L16 | SMAPE +2.5% |
| L12 | L17 | SMAPE +9.0%（不可移除） |
| L13 | L18 | SMAPE +6.5%（不可移除） |
| L14 | L19 | SMAPE +6.4%（不可移除） |

---

## 3. Spike Head 设计详述

### 3.1 理论依据

逐层消融揭示的"精度通路 vs 尖峰检测通路"分离现象：

- **原 L7（新 L5）**：ΔSMAPE +1.6%（精度影响极小），但 ΔSpike-F1 **−6.7%**（移除后尖峰检测能力显著下降）→ 该层是尖峰信号的关键编码节点。
- **原 L17–L19（新 L9–L11）**：SMAPE 灵敏（+6–9%，不可移除），同时对 Spike-F1 影响偏负（移除后 F1 略有回升）→ 深层对精度有重要贡献，但对尖峰检测有轻度"压制"效果。

**结论**：在精度"压制"层之前（L9–L11 之前）引出尖峰信号，可以让 spike head 拿到未被精度优化"覆盖"的纯尖峰表征。

### 3.2 Spike Head 结构

提供两个版本，默认用 V2，Step 6 敏感性分析中对比两版：

```python
class SpikeHeadV1(nn.Module):
    """单层版（基线）：参数量 ~31K。
    理论上 Transformer 输出已高度非线性，线性头可能够用，但表达力有限。"""

    def __init__(self, d_model: int = 1280, horizon: int = 24):
        super().__init__()
        self.proj = nn.Linear(d_model, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # [batch, horizon]


class SpikeHeadV2(nn.Module):
    """双层版（默认）：参数量 ~340K，仍可忽略不计。
    隐藏层 256 维，SiLU 激活，与 TimesFM 的 Swish 风格一致。"""

    def __init__(self, d_model: int = 1280, hidden: int = 256, horizon: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),           # 轻度正则，防止 47K 样本上的过拟合
            nn.Linear(hidden, horizon),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)             # [batch, horizon]
```

> 推理时：`spike_prob = torch.sigmoid(spike_head(h_layer5))`；阈值 0.5 判断为尖峰。  
> 训练时：使用 `nn.BCEWithLogitsLoss(pos_weight=...)`（见 Section 4.2）。

**⚠️ `hidden[:, -1, :]` 的语义——实现时必须验证**

上述代码中 `h_layer5 = hidden[:, -1, :]` 假设最后一个序列位置对应预测起点。TimesFM 是因果 decoder，output_projection 作用于最后一个 token 的输出——从 TimesFM 的 `output_projection_point` 调用位置可以确认这一点。

实现时请对照 `timesfm/src/timesfm/` 的 `forward()` 方法，确认：
1. `hidden` 的 shape 是 `[batch, n_patches, d_model]`（n_patches = context_len / patch_size）
2. `hidden[:, -1, :]` 对应最后一个 input patch 的 hidden state，即 TimesFM 用于输出预测的 token
3. 若 TimesFM 在 decode 时使用不同的索引（如专用 `[CLS]` token），需相应调整

### 3.3 Spike Head 的训练标签构造

Spike 标签 = 该未来时刻价格是否超过 P95 滚动阈值，与评估协议保持一致：

```python
def make_spike_labels(y_true: np.ndarray, threshold: float) -> torch.Tensor:
    """
    Args:
        y_true: [batch, horizon]  真实未来价格
        threshold: float  P95 尖峰阈值（来自回测框架的 thresholds.json）
    Returns:
        labels: [batch, horizon]  float tensor，1.0 / 0.0
    """
    return (y_true > threshold).astype(np.float32)
```

---

## 4. 损失函数

```
L_total = 0.8 × L_pinball + 0.2 × L_spike
```

### 4.1 Pinball Loss（分位数损失）

标准 Pinball Loss，9 个分位数（q0.1, …, q0.9），对 24 步求平均：

```
L_pinball = (1/9) × Σ_q Σ_t pinball(y_t, q_hat_t, q)
```

与原 TimesFM 训练目标完全一致，保证分位数预测的校准性。

### 4.2 BCE Spike Loss

**⚠️ 类别不平衡处理（关键）**：P95 阈值意味着正样本（尖峰）占比约 5%，即 95:5 的不平衡。不加权重的 `BCEWithLogitsLoss` 会退化为"全部预测非尖峰"（此时 95% 准确率但 Spike-F1 = 0），spike head 静默失效。

**必须加 `pos_weight`**：

```python
# pos_weight = neg_count / pos_count ≈ 95 / 5 = 19
# 等价地告诉模型：漏报一个尖峰的代价是误报的 19 倍
spike_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([19.0]).to(device)  # P95 不平衡比
)

L_spike = spike_criterion(spike_logits, spike_labels)
```

> **pos_weight 的计算**：从训练集统计实际正样本率后计算 `neg_count / pos_count`。P95 定义下理论值为 19，但不同时间段的实际率可能略有不同，建议训练前从数据统计一次。

24 步的所有时刻一起参与计算（不只看最极端时刻）。

### 4.3 权重选取理由

- 0.8 : 0.2 是初始值，基于"精度是主目标、尖峰是辅助目标"的设定。
- 若训练后尖峰 F1 < 0.3（远低于 Chronos-2 的 0.33），应将 `λ_spike` 上调至 0.3–0.4 并重训。
- **不推荐 `λ_spike` > 0.5**：即使有了 pos_weight，过大的 spike 权重仍可能破坏 Pinball 的分位数校准，导致覆盖率（coverage）偏离目标。

---

## 5. 参数规模

| 组件 | 参数量（估算） | 备注 |
|---|---|---|
| Tokenizer（ResidualBlock，64→1280） | ~3.3M | 原 TimesFM，保留 |
| **15 × Transformer 层**（attn + FFN + norms，d_model=1280） | **~150M** | 含 PerDimScale、QK-Norm（原 TimesFM 设计，全部保留）；Step 1 验证后由 8 层降为 5 层剪枝 |
| Quantile Head（output_projection_quantiles，原 TimesFM） | ~10M | 保留 |
| **Spike Head V2（新增，1280→256→24）** | **~340K** | 新增 |
| **总计** | **~163M** | |

> 原 TimesFM-2.5 约 200M，剪去 5 层后约 150M 主干 + 13M 头部 ≈ **163M**（原计划 133M，因 Step 1 退化超标改用保守方案）。  
> Spike Head V2 参数量（340K）相对 163M 仍可忽略。  
> **关于 PerDimScale 和 QK-Norm**：这两个 TimesFM 原有组件（每层 attention 内的可学习维度缩放和 Q/K RMSNorm）全部原样保留，未做替换，视为骨干整体组成部分。

---

## 6. 训练计划

### 6.1 数据准备

**训练集**：ERCOT 实时电价，2020-01-01 至 2025-07-31，但**显式排除三个测试窗口及其上下文缓冲期**

- 市场：ERCOT  
- 节点：`volatility` 组的 3 个高波动节点（与 v1.0/v2.0 实验一致）  
- 频率：1h  
- 上下文长度：168h（1 周）  
- 预测步长：24h  
- 构造方式：滑窗，stride = 1h → 约 **46,000 条训练样本**（排除窗口后估算）

**⚠️ 数据隔离规则（防止泄露）**：W2（2025-03）落在训练集时间范围内，必须显式排除。凡是满足以下条件的滑窗样本一律丢弃：

```
target 区间 [t+1h, t+24h] ∩ 测试窗口 ≠ ∅
                   OR
context 区间 [t-168h, t] ∩ 测试窗口 ≠ ∅
```

对应实际排除段（含 168h buffer）：

| 测试窗口 | 排除段（含 context buffer） | 落在训练范围内？ |
|---|---|---|
| W1 2025-08 | 2025-07-25 ~ 2025-08-31 | 仅末尾 buffer（2025-07-25~31）需排除 |
| **W2 2025-03** | **2025-02-21 ~ 2025-03-31** | **⚠️ 全部落在训练集内，必须排除** |
| W3 2026-01 | 2025-12-24 ~ 2026-01-31 | 完全在 2025-07-31 之后，自然隔离 |

```python
# 数据集构造时的过滤逻辑（伪代码）
EXCLUDED_RANGES = [
    ("2025-02-21", "2025-03-31"),  # W2 + buffer
    ("2025-07-25", "2025-07-31"),  # W1 context buffer
]

def is_excluded(context_start, target_end):
    for excl_start, excl_end in EXCLUDED_RANGES:
        if not (target_end < excl_start or context_start > excl_end):
            return True
    return False
```

**测试集**（与 v1.0/v2.0 基准完全一致，**不参与训练**）：

| 窗口 | 区间 | 特征 |
|---|---|---|
| W1 | 2025-08-01 ~ 08-31 | 夏季稳定期（基准） |
| W2 | 2025-03-01 ~ 03-31 | 春季负电价 |
| W3 | 2026-01-01 ~ 01-31 | 冬季极端尖峰 |

**完整时间划分一览**（基于实际可用数据：2025-01-01 ~ 2026-06-02）：

| 用途 | 时间段 | 样本数（3 节点合计） |
|---|---|---|
| 训练 Seg1 | 2025-01-01 ~ 2025-02-20 | ~3,099 条 |
| 排除（W2 + 168h buffer） | 2025-02-21 ~ 2025-03-31 | — |
| 训练 Seg2 | 2025-04-01 ~ 2025-07-24 | ~7,707 条 |
| 排除（W1 + 168h buffer） | 2025-07-25 ~ 2025-08-31 | — |
| 训练 Seg3 | 2025-09-01 ~ 2025-11-30 | ~5,979 条 |
| **验证集** | **2025-12-01 ~ 2025-12-24** | **~1,155 条** |
| 排除（W3 + 168h buffer） | 2025-12-25 ~ 2026-01-31 | — |
| 训练 Seg4 | 2026-02-01 ~ 2026-06-02 | ~8,211 条 |
| **训练合计** | **4 段不连续** | **~24,996 条** |
| 测试 W1 | 2025-08-01 ~ 2025-08-31 | 30 起报点 |
| 测试 W2 | 2025-03-01 ~ 2025-03-31 | 30 起报点 |
| 测试 W3 | 2026-01-01 ~ 2026-01-31 | 30 起报点 |

> 实际数据从 2025-01-01 起，无 2020-2024 历史。训练集由 4 个不连续段构成，dataset.py 的滑窗过滤逻辑确保没有任何窗口的 context 或 target 与测试窗口/验证集重叠。验证集固定为 Dec 2025 前半段（2025-12-01 ~ 2025-12-24），位于 W1 之后、W3 之前，时序上是真正未见数据。

### 6.2 分阶段微调策略

#### Stage 1：冻结低层，热身顶层（约 10 epoch）

**目标**：让顶层和两个输出头适应电价分布，同时保护底层预训练表征（含 spike 关键层）不被污染。

> **实际架构（Step 1 验证后）**：模型共 15 层（L0–L14），spike layer = 新 L6（原 L7）。
> Stage 1 冻结 tokenizer + L0–L6（7 层），训练 L7–L14（8 层）+ 两个 head。

```python
# 冻结：tokenizer + L0–L6（含 spike 关键层新 L6 = 原 L7，避免扰动尖峰表征）
for module in [model.tokenizer, *model.layers[:7]]:
    for p in module.parameters():
        p.requires_grad = False

# 训练：L7–L14 + quantile_head + spike_head
trainable_params = (
    list(model.layers[7:].parameters()) +
    list(model.quantile_head.parameters()) +
    list(model.spike_head.parameters())
)
# 可训练参数：约 8×10M + 10M + 0.34M ≈ 90M
```

- 学习率：`1e-4`  
- 优化器：AdamW（`weight_decay=0.01`）  
- LR scheduler：constant（Stage 1 不 decay，让顶层快速适应）

#### Stage 2：全层微调（约 40 epoch）

**目标**：端到端精调整个网络，让底层表征也向电价特征适配。

```python
# 解冻全部参数
for p in model.parameters():
    p.requires_grad = True
```

- 学习率：`5e-6`（比 Stage 1 低 20×，避免灾难性遗忘）  
- 优化器：AdamW（`weight_decay=0.01`）  
- LR scheduler：Cosine decay → `5e-7`  
- 梯度裁剪：`max_norm = 1.0`

**⚠️ Stage 2 过拟合退路（若验证集 Pinball 在 5 epoch 内开始回升）**：

全参数解冻 133M 参数在 46K 样本上存在过拟合风险。如果 Stage 2 中验证集 Pinball 无法继续收敛或开始反弹，改用 **LoRA 微调底层**：

```python
# 退路方案：保持底层 L0–L6 冻结，仅对底层 attn 的 q_proj/v_proj 加 LoRA adapter
# （需要 peft 库：pip install peft）
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,                     # LoRA rank，8 或 16
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],   # 仅注意力 QV 投影
    lora_dropout=0.05,
    bias="none",
)
# 应用到底层 L0–L6（仍冻结其余参数）
# 实际可训练参数：LoRA ~2M + 顶层 90M ≈ 92M，与 Stage 1 相当
```

> **执行顺序**：先跑默认的全参数 Stage 2，观察 5 epoch 内的验证集曲线。若过拟合，切换到 LoRA 退路，将两个结果作为消融比较（全参 vs LoRA）写入实验报告。

#### 通用设置

| 超参数 | 值 |
|---|---|
| batch_size | 32 |
| max_epochs | Stage1: 10，Stage2: 40，共 50 |
| early_stopping | patience = 10 epoch，监控验证集 Pinball Loss |
| 精度 | float16（amp），避免 d_model=1280 下的 OOM |
| checkpoint | 保存验证集 Pinball 最优的 checkpoint |

### 6.3 数据增强（可选，若过拟合）

- **滑窗 stride 扩展**：将 stride 从 1h 降为随机 {1h, 2h, 3h}，引入随机性
- **价格 jitter**：对训练序列加入 N(0, 0.01×σ) 的高斯噪声（σ 为该节点历史标准差）
- **节点 mixup**：随机在 3 个节点间混合上下文序列（仅对价格信号做，不跨节点混标签）

---

## 7. 评估协议

与 v1.0/v2.0 完全一致，确保对比公平：

```
市场: ERCOT | 节点: volatility(3个) | 频率: 1h | 上下文: 168h | 步长: 24h
起报点: ≥30 个滚动起报点（等间隔）| 指标: MAE / RMSE / MASE / SMAPE / Pinball / Spike-F1
```

ElecFM 需要在三个测试窗口（W1/W2/W3）上全部跑，以验证在不同市场状态下的鲁棒性。

---

## 8. 与三个基础模型的预期对比

基于消融逻辑的预期：

| 指标 | TimesFM-2.5（基准） | ElecFM（预期） | 依据 |
|---|---|---|---|
| SMAPE（W1） | 27.67 | ≤26（目标）| 移除冗余层减少过拟合；电价专用微调 |
| Spike-F1（W1） | 0.380 | ≥0.45（目标）| 专用 spike head + 0.2×BCE 训练信号 |
| 模型大小 | 200M | 133M | 移除 8 层 |
| 推理速度 | 基准 | ~40% 提速 | 层数从 20→12 |

> **注意**：SMAPE 提升是"可能"而非"保证"——如果电价数据量不足以充分微调 133M 参数，可能出现 SMAPE 持平甚至略差但 Spike-F1 显著改善的结果。这两种结果都是有价值的：前者说明融合+微调有效，后者说明大模型在小数据上仍有提升空间但尖峰目标可以专项改善。

---

## 9. 实现路径

### 9.1 文件结构

```
src/
└── fusion_model/
    ├── __init__.py
    ├── model.py              # ElecFM 主模型（继承/包装 TimesFM 模块）
    ├── spike_head.py         # SpikeHead 类
    ├── loss.py               # combined_loss(pinball, spike_bce, lambda_spike=0.2)
    ├── train.py              # 两阶段训练主循环
    ├── data.py               # 滑窗数据集构造（含 spike label 生成）
    └── evaluate.py           # 对接回测框架，生成与 v1.0/v2.0 格式一致的 summary.csv
```

### 9.2 关键实现细节

**① 层剪枝（加载 TimesFM 权重后就地删除）**

```python
import timesfm

tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
module = tfm.model  # TimesFM_2p5_200M_torch_module

# 要删除的层号（原 TimesFM 索引）
LAYERS_TO_REMOVE = {4, 6, 8, 9, 12, 13, 15, 16}
kept_layers = [layer for i, layer in enumerate(module.stacked_xf)
               if i not in LAYERS_TO_REMOVE]
module.stacked_xf = nn.ModuleList(kept_layers)
# 剪枝后 module.stacked_xf 有 12 层，新 L5 = 原 L7
```

**② Spike Head 接入点**

新 L5 = 原 L7（在删除 {L4, L6} 后，原 L7 是第 6 个被保留的层，0-indexed 为 5）。

```python
# 在 forward 中，保存新 L5 的输出用于 spike head
hidden = tokenizer_output
for i, layer in enumerate(stacked_xf):
    hidden, _ = layer(hidden, atten_mask, decode_cache=None)
    if i == 5:  # 新 L5 = 原 L7，尖峰关键层
        h_spike = hidden[:, -1, :]  # 取最后一个 patch 位置的表征，[batch, d_model]

# 双头输出
spike_logits = spike_head(h_spike)                          # [batch, 24]
quantile_output = quantile_head(hidden[:, -1, :])           # [batch, 24×9]
```

**③ 验证剪枝正确性（实现后第一件事）**

```python
# 验证原 L7 确实是新 L5
original_l7_id = id(original_module.stacked_xf[7])
pruned_l5_id   = id(pruned_module.stacked_xf[5])
assert original_l7_id == pruned_l5_id, "层对应关系验证失败"
```

**④ 尖峰阈值**

沿用回测框架已有的 `thresholds.json`（P95 rolling 窗口），不重新计算，保证与 v1.0/v2.0 评估一致。

### 9.3 先决条件检查

在开始实现前，确认以下前提：

- [ ] TimesFM-2.5 的 `.venv` 中 PyTorch 版本支持 `autocast`（fp16 训练）
- [ ] 训练机器有足够显存（133M + 优化器状态 + 梯度 ≈ 至少 12GB VRAM 在 fp16 下）
- [ ] ERCOT 数据已有 2020–2025-07 的完整 hourly 长表（`loader.load_slice` 可覆盖此范围）
- [ ] `thresholds.json`（P95 阈值）已存在并可读取
- [ ] **TimesFM `forward()` 支持反向传播**（⚠️ 关键，实现后第一件事就验证）

  结构消融阶段只用了 TimesFM 的推理模式（子进程 + `no_grad` 环境）。训练模式下需要确认：

  ```python
  # 验证脚本（放入先决条件检查，实现后立即跑）
  module = tfm.model
  module.train()  # 确保 train 模式

  # 构造一个最小 batch
  dummy_input = torch.randn(2, 6, 1280)  # [batch, n_patches, d_model]，粗略尺寸
  dummy_mask  = torch.ones(2, 6, dtype=torch.bool)

  # 走一遍 forward（需要 hook 住中间层输出）
  output = module.stacked_xf[0](dummy_input, dummy_mask)
  loss = output[0].sum()
  loss.backward()

  # 检查梯度是否正常流通
  for name, p in module.stacked_xf[0].named_parameters():
      assert p.grad is not None, f"梯度未到达参数 {name}，检查是否有 no_grad/detach"
  print("✅ 梯度流通正常")
  ```

  **常见问题及处理**：
  - 若 `tfm.forecast()` 内部有 `@torch.no_grad()` 或 `with torch.no_grad()` 包裹 → 不要调用 `forecast()`，直接调用 `module.forward()`
  - 若某层内有 `tensor.detach()` 调用 → 需要在 fork 的模型代码里移除该 detach，保留梯度通路
  - 若 TimesFM 使用了 JAX/Flax（部分版本）而非 PyTorch → 需切换到 PyTorch 版本（`timesfm_2p5_base.py` 的 `TimesFM_2p5_200M_torch`）

---

## 10. 实验结论预期与"so what"说明

本设计将结构消融最重要的**方法论发现**——精度通路与尖峰检测通路的功能分离——转化为具体的架构改进：

1. **Skip connection 位置**（新 L5 = 原 L7）不是拍脑袋选的，而是逐层消融实验的直接输出：该层 ΔSpike-F1 = −6.7%（移除后尖峰检测能力显著下降）而 ΔSMAPE 仅 +1.6%（精度几乎不受影响），正是"纯尖峰检测层"的精确定位。

2. **Spike head 在 L5 之后、L9 之前分叉**的设计，对应了另一个发现：深层（新 L9–L11 = 原 L17–L19）对精度有重要贡献但对尖峰有轻度压制（移除这些层后 ΔF1 为 −0.6% ~ −5.7%）。精度头走到底，spike head 在压制开始前提前离场。

3. 双头权重 0.8 : 0.2 的选取基于尖峰标签的稀疏性（P95 正样本率 < 5%），过大的 spike 权重会破坏分位数校准。

如果最终实验结果显示 Spike-F1 提升 > 10%，这个方法论链条（消融 → 发现 → 架构决策 → 验证）就构成了一个完整的科研贡献。

---

## 附录 A：不引入的设计及理由

| 设计 | 为何不引入 |
|---|---|
| SwiGLU FFN（Toto-2.0 风格） | 替换 FFN 需重新初始化权重，在 47K 样本上从头训 FFN 等于放弃预训练知识；TimesFM 的 Swish MLP 已经是最强基准 |
| xPos 衰减 | 结构消融明确结论：Toto xPos Δ ≈ 0%，完全冗余，不引入 |
| 跨变量注意力（Variate/Group） | Q2 明确决策；TimesFM 骨干无此机制，引入需大量修改且 47K 样本不足以学习跨节点模式 |
| 双分支架构（方案 C） | 47K 样本训两条分支等于稀释一半；设计自由度过大，难以归因 |
| 多节点联合训练（multivariate=True） | Q2 决策；与 TimesFM 单变量骨干兼容性问题，留到后续版本 |
| 从头训练 | 47K 样本 vs 133M 参数严重不对等，必须依赖 TimesFM 预训练的跨域时序知识 |

---

## 附录 B：待做实验的执行顺序

```
Step 1：【验证】12 层剪枝基准（零样本，不训练）
  → 对比原 20 层 TimesFM 的 SMAPE 和 Spike-F1
  → 通过标准：SMAPE 退化 < 5%（验证 8 层同时移除的累积效应可接受）
  → 若 5% ≤ SMAPE 退化 < 10%：警告，进入 Step 2 但在 Step 5 额外报告基准差距
  → 若 SMAPE 退化 ≥ 10%：层间依赖强，切换到 5 层保守方案：
      仅移除 |ΔSMAPE| < 1% 的最安全层 {L6(+0.6%), L8(≈0%), L9(+0.8%), L13(+0.3%), L15(+0.8%)}
      → 重跑 Step 1 验证 15 层模型，通过后继续

Step 2：【实现】SpikeHead + 双头 forward pass（不训练，随机初始化 spike head）
  → 验证 spike head 的 skip connection 能正确接到新 L5 的输出
  → 确认 TimesFM forward() 在训练模式下梯度可正常流通（见 9.3 先决条件第 5 条）

Step 3：【训练 Stage 1】冻结底层，只训顶层 + 两个 head，10 epoch
  → 监控：训练 loss 下降，验证集 Pinball 不发散

Step 4：【训练 Stage 2】解冻全部，低 LR 微调，40 epoch + early stopping
  → 若验证集 Pinball 在 5 epoch 内反弹，切换至 LoRA 退路（见 6.2）

Step 5：【评估】在 W1/W2/W3 上跑滚动回测，生成 summary.csv
  → 与 v1.0/v2.0 中 TimesFM 基准对比
  → ⚠️ Spike head 推理阈值搜索（不能直接用 0.5）：
      先在验证集上枚举阈值 {0.05, 0.10, ..., 0.90}，
      找到使验证集 Spike-F1 最大的最优阈值 τ*，
      再以 τ* 在三个测试窗口上输出最终 Spike-F1
      （原因：pos_weight=19 训练后模型输出分布已偏移，0.5 不再是最优边界）

Step 6：（可选）消融对比
  → λ_spike 敏感性：{0.1, 0.2, 0.3, 0.4}，找 Pinball vs Spike-F1 的 Pareto 前沿
  → SpikeHead V1 vs V2：对比单层 Linear 和 Linear-SiLU-Linear 的效果差异
```
