# ElecFM LoRA 优化方案

> 版本：v1.0 ｜ 日期：2026-07-02
> 前置文档：`fusion_model_design_v3.md`
> 状态：**设计完成，待实现**

---

## 1. 问题背景

### 1.1 第一轮全 ERCOT 训练结果

**执行时间**：2026-07-01

**零样本基准验证**：
```
15层 ElecFM（零样本）SMAPE = 27.55
原始20层 TimesFM SMAPE = 27.67
退化 = -0.4% （实际略微提升！）
✅ 剪枝基准通过
```

**两阶段训练趋势**：

| Stage | Epoch | Train Pinball | Val Pinball | 状态 |
|-------|-------|---------------|-------------|------|
| S1 | 1 | 3.87 | **8.29** | ✅ 最优 |
| S1 | 5 | 1.93 | 8.70 | 📈 过拟合 |
| S1 | 10 | 1.47 | 8.95 | 📈📈 严重过拟合 |
| S2 | 1 | 2.93 | 8.44 | 📈 继续恶化 |

**Spike 任务同样过拟合**：
- Train: 1.09 → 0.55（下降）
- Val: 1.44 → 1.58（上升）

### 1.2 问题诊断

```
Stage 1 可训练参数: 79,020,184
训练样本: 124,605
参数/样本比: 79M / 125K ≈ 634 参数/样本 ❌ 严重失衡

Stage 2 可训练参数: 149,676,584
参数/样本比: 149M / 125K ≈ 1,197 参数/样本 ❌❌ 极度失衡
```

**根本原因**：模型容量远大于数据量，模型在**记忆训练数据**而非学习泛化模式。

### 1.3 为什么传统方法不够

| 方法 | 局限性 |
|------|--------|
| 早停 | 只能延缓过拟合，无法解决根本问题（最优 checkpoint 在 epoch 1） |
| Weight Decay | 对 79M 参数效果有限，正则化力度难以平衡 |
| Dropout | Transformer 已有内置 dropout，进一步增加可能损害预训练知识 |
| 数据增强 | 时序数据增强需谨慎设计，盲目添加可能破坏时间依赖性 |

---

## 2. LoRA 方案设计

### 2.1 为什么选择 LoRA

**核心优势**：
1. **参数效率**：可训练参数从 79M 降至 **~3-5M**（减少 93-95%）
2. **参数/样本比**：634 → **24-40**（进入合理范围）
3. **保留预训练知识**：冻结 TimesFM 主干，只训练低秩适配器
4. **快速收敛**：更少的参数意味着更快的训练和更稳定的优化

**与 ElecFM 的契合点**：
- 零样本表现已经很好（SMAPE 27.55），不需要大幅改变预训练权重
- 电价域适应是"轻量微调"任务，LoRA 的低秩假设成立
- 两阶段训练策略可以自然扩展到 LoRA：Stage 1 训练高层 LoRA，Stage 2 扩展到全层

### 2.2 核心配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # 低秩维度 (4-16 范围)
    lora_alpha=16,          # 缩放因子，通常 = 2*r
    target_modules=[        # 目标模块（TimesFM 的 attention 和 FFN）
        "q_proj",           # Query 投影
        "k_proj",           # Key 投影
        "v_proj",           # Value 投影
        "gate_proj",        # FFN gate（如果存在）
        "up_proj",          # FFN up
        "down_proj",        # FFN down
    ],
    lora_dropout=0.05,      # 轻度 dropout 防过拟合
    bias="none",            # 不训练 bias
    task_type="SEQ_2_SEQ_LM",  # 任务类型
)

# 应用 LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 预期输出: trainable params: ~4,000,000 || all params: ~163,000,000 || trainable%: ~2.5%
```

### 2.3 两阶段 LoRA 训练策略

#### Stage 1：高层 LoRA 预热

```python
# 冻结所有非 LoRA 参数（包括 TimesFM 主干和 quant_head）
for name, param in model.named_parameters():
    if "lora_" not in name:  # 只训练 LoRA 参数
        param.requires_grad = False

# Stage 1 只训练特定层的 LoRA（可选优化）
# 默认训练所有层的 LoRA，但 spike_head 始终可训练

# 优化器设置
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,                # LoRA 可以使用更高学习率
    weight_decay=0.01,
)

# Scheduler: Constant（Stage 1 不衰减）
```

**Stage 1 特点**：
- 可训练参数：~4M (LoRA) + 0.34M (spike_head) ≈ **4.3M**
- 学习率：**1e-3**（比非 LoRA 的 1e-5 高 100 倍，LoRA 更稳定）
- Epochs：**5**（减少 from 10，LoRA 收敛更快）

#### Stage 2：全层 LoRA 微调

```python
# 所有 LoRA 参数保持可训练
# 可选择性解冻部分 TimesFM 层（如果需要）

# 优化器设置
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4,                # 比 Stage 1 略低
    weight_decay=0.01,
)

# Scheduler: Cosine decay → 5e-5
```

**Stage 2 特点**：
- 可训练参数：~4M (LoRA) + 0.34M (spike_head) ≈ **4.3M**（与 Stage 1 相同）
- 学习率：**5e-4** → **5e-5**（Cosine decay）
- Epochs：**20**（减少 from 40）
- 早停：patience=5（更敏感）

---

## 3. 完整配置对比

### 3.1 参数规模对比

| 配置 | 非 LoRA 方案 | LoRA 方案 | 变化 |
|------|-------------|-----------|------|
| **Stage 1 可训练参数** | 79M | **4.3M** | -95% |
| **Stage 2 可训练参数** | 149M | **4.3M** | -97% |
| **参数/样本比** | 634 / 1197 | **34** | -95% |
| **模型总大小** | 163M | **163M** | 相同 |
| **Checkpoint 大小** | ~650MB | **~20MB** | -97% |

### 3.2 训练超参数对比

| 参数 | 非 LoRA | LoRA | 理由 |
|------|---------|------|------|
| **Stage 1 LR** | 1e-5 | **1e-3** | LoRA 更稳定，可用更高 LR |
| **Stage 2 LR** | 5e-7 | **5e-4** | 保持 20:1 比例 |
| **Stage 1 Epochs** | 10 | **5** | 更快收敛，早停更早 |
| **Stage 2 Epochs** | 40 | **20** | 减少过拟合风险 |
| **Batch Size** | 32 | **32** | 相同 |
| **Weight Decay** | 0.01 | **0.01** | 相同 |
| **LoRA Dropout** | N/A | **0.05** | 新增 |
| **LoRA Rank (r)** | N/A | **8** | 新增 |
| **LoRA Alpha** | N/A | **16** | = 2*r |
| **早停 Patience** | 10 | **5** | 更敏感 |

### 3.3 预期训练时间

| 阶段 | 非 LoRA | LoRA | 加速比 |
|------|---------|------|--------|
| Stage 1 | ~3.5h (10 epochs) | **~1h** (5 epochs) | 3.5x |
| Stage 2 | ~24h (40 epochs) | **~6h** (20 epochs) | 4x |
| **总计** | ~27.5h | **~7h** | **4x** |

---

## 4. 实现路径

### 4.1 依赖安装

```bash
# 在 TimesFM 虚拟环境中安装 peft
external/timesfm/.venv/bin/pip install peft
```

### 4.2 代码修改清单

#### 修改 1: `src/fusion_model/model.py`

添加 LoRA 支持：

```python
from peft import LoraConfig, get_peft_model, TaskType

def create_elecfm_model(use_lora=True, lora_config=None):
    """创建 ElecFM 模型，可选 LoRA"""
    # 加载基础 TimesFM 模型（含剪枝）
    base_model = load_pruned_timesfm(...)
    
    if use_lora:
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=get_target_modules(base_model),
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model
    
    return model

def get_target_modules(model):
    """自动识别目标模块名称"""
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 匹配 attention 和 FFN 的投影层
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]):
                target_modules.append(name)
    return target_modules
```

#### 修改 2: `src/fusion_model/train.py`

适配 LoRA 训练流程：

```python
def train_epoch(model, dataloader, optimizer, use_lora=True):
    """训练一个 epoch"""
    model.train()
    
    # LoRA 模式下确保只有 LoRA 参数可训练
    if use_lora:
        for name, param in model.named_parameters():
            if "lora_" in name or "spike_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    for batch in dataloader:
        # 正常训练流程
        ...
```

#### 修改 3: `configs/fusion/electfm_ercot_full.yaml`

添加 LoRA 配置：

```yaml
# LoRA 配置
use_lora: true
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

# 调整后的训练参数（LoRA 适用）
training:
  stage1:
    epochs: 5
    lr: 1e-3
    weight_decay: 0.01
  stage2:
    epochs: 20
    lr: 5e-4
    min_lr: 5e-5
    weight_decay: 0.01
  early_stopping:
    patience: 5
    monitor: "val_pinball"
```

#### 修改 4: Checkpoint 保存/加载

```python
def save_checkpoint(model, path, use_lora=True):
    """保存 checkpoint"""
    if use_lora:
        # 只保存 LoRA 参数（小得多）
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), path)

def load_checkpoint(model, path, use_lora=True):
    """加载 checkpoint"""
    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path)
    else:
        model.load_state_dict(torch.load(path))
    return model
```

### 4.3 验证清单

- [ ] peft 安装成功：`external/timesfm/.venv/bin/python -c "from peft import LoraConfig; print('OK')"`
- [ ] LoRA 参数正确识别：`print_trainable_parameters()` 显示 ~4M 可训练参数
- [ ] 梯度流通正常：LoRA 参数在 backward 后有梯度
- [ ] Checkpoint 保存/加载正常：文件大小 ~20MB
- [ ] 训练速度提升：Stage 1 每个 epoch < 15 分钟

---

## 5. 预期结果

### 5.1 成功标准

| 指标 | 非 LoRA 结果 | LoRA 目标 | 理由 |
|------|-------------|-----------|------|
| **Val Pinball 趋势** | Epoch 1 后持续上升 | **至少 5 个 epoch 下降或持平** | 过拟合缓解 |
| **最优 Epoch** | 1 | **> 3** | 模型真正学习而非记忆 |
| **W1 SMAPE** | 未收敛（训练发散） | **≤ 28.0** | 接近零样本 27.55 |
| **Coverage** | 未测量 | **≥ 0.75** | 保持分位数校准 |
| **Spike-F1** | 未测量 | **≥ 0.40** | 超过基准 0.38 |

### 5.2 失败预案

**如果 LoRA (r=8) 仍有过拟合迹象**：
1. 降低 rank: r=4（参数再减半）
2. 增加 dropout: 0.05 → 0.1
3. 增加数据增强：价格 jitter、节点 mixup
4. 减少 Stage 1 epochs: 5 → 3

**如果 LoRA 欠拟合（收敛太慢）**：
1. 提高 rank: r=8 → r=16
2. 提高学习率: 1e-3 → 2e-3
3. 增加 Stage 1 epochs: 5 → 10

---

## 6. 后续计划

### 阶段 1：LoRA 实现与验证

1. 实现 LoRA 支持（预计 2-3 小时）
2. 小规模验证（lsp 单个节点，1 小时）
3. 修复问题，确保梯度流通

### 阶段 2：全 ERCOT LoRA 训练

```bash
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full.yaml \
    --use_lora \
    2>&1 | tee run_ercot_lora.log
```

预期时间：~7 小时（夜间运行）

### 阶段 3：结果分析与调参

**如果成功**：
- 补做 lambda sweep（λ_spike = 0.1, 0.2, 0.3）
- 对比 LoRA vs 非 LoRA 的最终指标
- 更新所有文档

**如果失败**：
- 尝试更低 rank 或更多正则化
- 或考虑其他方案（如 Adapter、Prompt Tuning）

---

## 附录：LoRA 原理简述

### 核心思想

对于预训练权重 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 冻结 $W_0$ 并添加低秩分解：

$$W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

### 为什么有效

1. **预训练权重已经很好**：$W_0$ 包含大量跨域时序知识，不应大幅修改
2. **微调是低秩过程**：Aghajanyan et al. (2020) 证明微调主要发生在低维子空间
3. **参数效率**：参数量从 $d \times k$ 降至 $r \times (d + k)$，当 $r=8, d=k=1280$ 时，减少 ~98%

### 在 ElecFM 中的应用

```
TimesFM Attention:
  Q = X @ W_q          # W_q: [1280, 1280]
  
With LoRA:
  Q = X @ (W_q + B_q @ A_q)   # B_q: [1280, 8], A_q: [8, 1280]
  
参数节省: 1280*1280 = 1.6M → 1280*8*2 = 20K (-98.7%)
```

---

## 参考

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
2. Aghajanyan et al. (2020). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.
3. PEFT 文档: https://huggingface.co/docs/peft
