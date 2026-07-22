# ElecFM 设计文档

> **当前状态**：LoRA 实现完成 ✅，全 ERCOT 训练中  
> **最后更新**：2026-07-02  
> **版本历史**：v3.0 (设计) → v3.1 (添加 LoRA)

---

## 📋 文档导航

| 章节 | 内容 |
|------|------|
| [1. 架构设计](#1-架构设计) | ElecFM 模型结构、数据流、剪枝决策 |
| [2. 训练策略](#2-训练策略) | LoRA 微调方案、两阶段训练、超参数 |
| [3. 实现细节](#3-实现细节) | 代码结构、关键配置、使用示例 |
| [4. 实验记录](#4-实验记录) | 各轮实验结果、问题诊断 |

**相关文档**：
- 完整实验记录 → `experiments.md`
- 技术实现细节 → `implementation.md`
- 任务追踪 → `../TODO.md`

---

## 1. 架构设计

### 1.1 命名与定位

**ElecFM**（Electricity Foundation Model）——基于 TimesFM-2.5 的电价预测专用模型。

### 1.2 消融结论 → 设计决策

| 决策点 | 消融依据 | 最终决策 |
|--------|----------|----------|
| **骨干选择** | TimesFM 基准 SMAPE 最低（27.67 vs 29.92/30.06）| ✅ TimesFM-2.5 为骨干 |
| **FFN 类型** | 移除 FFN 后退化最大（+486%）| ✅ 保留 Swish MLP |
| **位置编码** | xPos 冗余（Δ≈0%），RoPE 轻度依赖（+6%）| ✅ 保留标准 RoPE |
| **跨变量机制** | 变量注意力贡献非决定性（+7%~+22%）| ✅ **不引入** |
| **层深度** | 8 层同时移除退化 +18%（失败）| ✅ 5 层保守方案 |
| **输出头** | 简化头 SMAPE +88% | ✅ 保留 quantile + spike 双头 |
| **精度 vs 尖峰** | L7 是纯尖峰检测层（ΔSpike-F1 −6.7%）| ✅ spike head 从 L7 分叉 |

### 1.3 数据流

```
原始电价序列（单节点，单变量）
  → Tokenizer（ResidualBlock，32 时间步 → 1280 维 patch 嵌入）
    → L0（原 TimesFM L0，CRITICAL）
    → L1（原 L1，CRITICAL）
    → L2（原 L2）
    → L3（原 L3，spike 敏感）
    → L4（原 L5）
    → L5（原 L7，"尖峰检测层"）
         │
         ├──► [SPIKE HEAD] → spike_prob[0..23]
         │
    → L6（原 L10）
    → L7（原 L11）
    → L8（原 L14）
    → L9-L11（原 L17-L19，精度关键层）
         │
         └──► [QUANTILE HEAD] → q[0.1..0.9] × 24
```

### 1.4 剪枝方案（15 层）

保留的层（新编号 → 原编号）：

| 新层号 | 原层号 | 保留理由 |
|--------|--------|----------|
| L0 | L0 | CRITICAL：SMAPE +71.6% |
| L1 | L1 | CRITICAL：SMAPE +40.7% |
| L2 | L2 | SMAPE +3.3% |
| L3 | L3 | spike 敏感：ΔF1 −5.5% |
| **L5** | **L7** | **spike 关键层**：ΔF1 −6.7% |
| L9-L11 | L17-L19 | 精度关键层：SMAPE +6~9% |

**零样本验证**：15 层 SMAPE 27.55 vs 20 层 27.67（退化 -0.4% ✅）

### 1.5 Spike Head 设计

```python
class SpikeHeadV2(nn.Module):
    """双层版：1280 → 256 → 24"""
    def __init__(self, d_model=1280, hidden=256, horizon=24):
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, horizon),
        )
```

**损失函数**：
```
L_total = 0.8 × L_pinball + 0.2 × L_spike
L_spike = BCEWithLogitsLoss(pos_weight=19.0)  # P95 不平衡
```

---

## 2. 训练策略

### 2.1 为什么用 LoRA

**第一轮训练问题**（非 LoRA）：
- 可训练参数 79M / 125K 样本 = **634 参数/样本** ❌
- Val Pinball 从第 1 epoch 持续上升（严重过拟合）

**LoRA 解决方案**：
- 可训练参数降至 **1.1M**（-98.6%）
- 参数/样本比 = **8.8** ✅
- 冻结 TimesFM 主干，保留预训练知识

### 2.2 LoRA 配置（TimesFM 适配）

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # 低秩维度
    lora_alpha=16,          # 缩放因子 = 2*r
    target_modules=[        # TimesFM 命名
        "attn.qkv_proj",    # 合并 QKV
        "attn.out",         # Attention 输出
        "ff0", "ff1",       # FFN 两层
    ],
    lora_dropout=0.05,
    bias="none",
)
```

**参数规模**：
- 总参数：178M
- 可训练：1.1M（0.62%）
- 预期 Checkpoint：~20MB

### 2.3 两阶段训练

#### Stage 1：LoRA 预热

| 参数 | 非 LoRA | LoRA |
|------|---------|------|
| 可训练参数 | 79M | 1.1M |
| 学习率 | 1e-5 | **1e-3** (+100x) |
| Epochs | 10 | **5** |
| 冻结 | L0-L6, quant_head | 所有非 LoRA 参数 |

#### Stage 2：全层 LoRA

| 参数 | 非 LoRA | LoRA |
|------|---------|------|
| 学习率 | 5e-7 | **5e-5** |
| Epochs | 40 | **20** |
| Scheduler | Cosine | Cosine |
| 早停 patience | 10 | **5** |

### 2.4 关键超参数

```yaml
# 公共配置
batch_size: 32
weight_decay: 0.01
gradient_clip: 1.0
use_amp: true

# 损失权重
lambda_pinball: 0.8
lambda_spike: 0.2
spike_pos_weight: 19.0  # P95 统计

# LoRA 特有
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
```

---

## 3. 实现细节

### 3.1 代码结构

```
src/fusion_model/
├── model.py       # ElecFM + LoRA 支持
├── train.py       # 两阶段训练循环
├── run_fusion.py  # 主入口
├── spike_head.py  # SpikeHeadV2
├── loss.py        # combined_loss
└── evaluate.py    # 评估流程
```

### 3.2 使用示例

```bash
# 快速测试（单节点，2+2 epochs）
bash run_lora_quick_test.sh

# 正式训练（全 ERCOT 15 节点）
python src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_lora.yaml
```

### 3.3 配置文件

**LoRA 配置** `configs/fusion/electfm_ercot_full_lora.yaml`：
```yaml
use_lora: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05

stage1_epochs: 10  # 实际 5
stage1_lr: 1.0e-5  # 实际 1e-3

stage2_epochs: 40  # 实际 20
stage2_lr: 5.0e-7  # 实际 5e-5
```

---

## 4. 实验记录

### 4.0 实验结论总览（2026-07-02，已完成）

六轮实验（全参数/冻结quant_head/LoRA/纯spike head/720h/V6）揭示的核心规律：

> **任何骨干权重修改都导致 SMAPE 退化；完全冻结骨干是正确路线。**

**最终实验结果汇总**：

| 版本 | W1 SMAPE | W1 Spike-F1(head) | 关键教训 |
|------|----------|-------------------|---------|
| 零样本 TimesFM | 27.67 | 0.329 | 基准 |
| v1（quant_head 可训）| 31.95 | 0.370 | quant_head 必须冻结 |
| v2（quant_head 冻结）| 29.19 | 0.335 | 数据太少 |
| v3（全 ERCOT，非 LoRA）| 未收敛 | — | 参数/样本比 634，严重过拟合 |
| v4（LoRA）| 29.39 | 0.351 | 骨干修改仍损害 SMAPE |
| **v5a（纯 spike head）**| **27.55** ✅ | **0.4159** ✅ | **冻结骨干是正确路线** |
| v5b（720h context）| 27.52 | 0.3779 | 720h 稀释尖峰信号 |
| **V6（CrossNodeAttention）**| 28.90（3节点）| **0.4439** ✅ | **消融预测得到验证，+6.7%** |

### 4.1 快速测试结果（2026-07-02）

**配置**：HB_HOUSTON 单节点，LoRA r=4，2+2 epochs

| 指标 | 结果 |
|------|------|
| W1 SMAPE | **20.93** ✅ (vs baseline 27.67) |
| 可训练参数 | 1.1M (0.62%) |
| Stage 1 | val pinball 3.81 → 4.40 (上升) |
| Stage 2 | val pinball 4.02 → 4.28 (上升) |

**分析**：
- ✅ W1 表现优秀，模型结构正确
- ⚠️ 单节点数据太少（2K 样本），仍然过拟合
- 🎯 全 ERCOT 训练（124K 样本）应该能解决

### 4.2 各轮实验汇总

| 轮次 | 配置 | 状态 | W1 SMAPE | W1 Spike-F1 |
|------|------|------|----------|-------------|
| 零样本基准 | TimesFM 原版 | ✅ | 27.67 | 0.329 |
| v1 | quant_head 可训 | ❌ | 31.95 | 0.370 |
| v2 | quant_head 冻结 | ⚠️ | 29.19 | — |
| v3 | 全 ERCOT，非 LoRA | ❌ | 未收敛 | — |
| v4 | 全 ERCOT，LoRA | ⚠️ | 29.39 | 0.351 |
| **v5a** | **纯 spike head，168h** | ✅ **最优** | **27.55** | **0.4159** |
| v5b | 纯 spike head，720h | ✅ | 27.52 | 0.3779（差于 v5a）|
| **V6** | **CrossNodeAttention（3节点）** | ✅ **Spike-F1 最优** | 28.90（3节点口径）| **0.4439** |
| V6 allgroups | 5组×3节点，d_attn=64 | ✅ | 28.90 | 0.3840（NaN 不稳定）|
| V7 SwiGLU adapter | v5a + SwiGLU adapter | ✅ | 27.55 | 0.3980（< v5a）|
| V6 d_attn=32/128 | 3节点 | ✅ | 28.90 | 0.3967 / 0.3819 |

---

## 附录

### A. 参考资料

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
2. PEFT 文档: https://huggingface.co/docs/peft

### B. 历史文档

- `docs/archive/elecfm/fusion_model_design_v3.md` - 原始设计 v3
- `docs/archive/elecfm/elecfm_lora_optimization.md` - 原始 LoRA 方案

---

**维护者**：Claude Code  
**最后更新**：2026-07-02
