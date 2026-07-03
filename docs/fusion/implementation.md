# ElecFM 实现细节

> 技术实现指南，包含代码结构、配置说明和调试技巧  
> 最后更新：2026-07-02

---

## 📁 代码结构

```
src/fusion_model/
├── __init__.py
├── model.py              # ElecFM 模型 + LoRA 支持
├── train.py              # 两阶段训练循环
├── run_fusion.py         # 主入口（CLI）
├── spike_head.py         # SpikeHead V1/V2
├── loss.py               # 组合损失函数
├── evaluate.py           # 评估流程
└── dataset.py            # 数据集构造
```

---

## 1. 模型实现

### 1.1 ElecFM 类

核心组件：
- `tokenizer`: TimesFM ResidualBlock
- `layers`: 15 层 Transformer（ModuleList）
- `quant_head`: 原始分位数头（**冻结**）
- `spike_head`: 新增尖峰检测头

### 1.2 LoRA 集成

**关键函数**：`create_elecfm_with_lora()`

```python
def create_elecfm_with_lora(
    horizon=24,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
) -> ElecFM:
    # 1. 创建基础模型
    base_model = ElecFM(...)
    
    # 2. 配置 LoRA（TimesFM 适配）
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "attn.qkv_proj",  # 注意：TimesFM 是合并 QKV
            "attn.out",
            "ff0", "ff1",
        ],
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # 3. 应用 LoRA
    base_model.layers = get_peft_model(base_model.layers, config)
    
    return base_model
```

**TimesFM 命名适配**：
- 标准 Transformer: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **TimesFM**: `attn.qkv_proj`, `attn.out`, `ff0`, `ff1`

### 1.3 PeftModel 兼容

`forward()` 中需要特殊处理：

```python
def forward(self, x):
    # ... tokenizer ...
    
    # 兼容 PeftModel 和普通 ModuleList
    layers = self.layers
    if hasattr(layers, 'base_model'):
        layers = layers.base_model.model
    
    for i, layer in enumerate(layers):
        h, _ = layer(h, patch_mask, None)
        # ...
```

---

## 2. 训练实现

### 2.1 两阶段策略

**Stage 1：LoRA 预热**
```python
_setup_lora_training(model, train_lora=True)
# 只训练 LoRA 参数 + spike_head
# 学习率: 1e-3 (比非 LoRA 高 100x)
```

**Stage 2：全层 LoRA**
```python
# 保持相同的冻结策略
# 学习率: 5e-5 (Cosine decay)
```

### 2.2 参数冻结逻辑

```python
def _setup_lora_training(model, train_lora=True):
    for name, p in model.named_parameters():
        if train_lora:
            # 只训练 LoRA 参数和 spike_head
            if "lora_" in name or "spike_head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
```

### 2.3 学习率自动调整

| 模式 | Stage 1 LR | Stage 2 LR |
|------|------------|------------|
| 非 LoRA | 1e-5 | 5e-7 |
| LoRA | **1e-3** | **5e-5** |

---

## 3. 配置说明

### 3.1 完整配置示例

`configs/fusion/electfm_ercot_full_lora.yaml`:

```yaml
# 数据配置
market: ERCOT
nodes: [HB_BUSAVG, HB_HOUSTON, ...]  # 15 节点
freq: 1h
context_len: 168
horizon: 24

# LoRA 配置
use_lora: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05

# 训练配置（LoRA 模式下会自动调整）
stage1_epochs: 10  # 实际使用 5
stage1_lr: 1.0e-5  # 实际使用 1e-3

stage2_epochs: 40  # 实际使用 20
stage2_lr: 5.0e-7  # 实际使用 5e-5
stage2_lr_min: 5.0e-8

# 优化配置
batch_size: 32
gradient_clip: 1.0
weight_decay: 0.01
early_stop_patience: 10
use_amp: true

# 损失配置
lambda_pinball: 0.8
lambda_spike: 0.2

# 评估配置
stride_hours: 24
max_origins: 30
spike_quantile: 0.95
```

### 3.2 配置覆盖规则

LoRA 模式下，以下参数会自动覆盖：
- `stage1_epochs` = max(5, cfg.stage1_epochs // 2)
- `stage1_lr` = cfg.stage1_lr * 100
- `stage2_lr` = cfg.stage2_lr * 100
- `stage2_epochs` = max(20, cfg.stage2_epochs // 2)

---

## 4. 使用指南

### 4.1 快速测试

```bash
# 单节点快速验证（10-15 分钟）
bash run_lora_quick_test.sh
```

### 4.2 正式训练

```bash
# 全 ERCOT 训练（3-4 小时）
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_lora.yaml \
    2>&1 | tee run_ercot_lora.log
```

### 4.3 仅评估

```bash
# 使用已有 checkpoint，跳过训练
python src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_lora.yaml \
    --skip-train
```

### 4.4 仅剪枝验证

```bash
# 验证零样本性能，不训练
python src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_lora.yaml \
    --step1-only
```

---

## 5. 调试技巧

### 5.1 验证 LoRA 是否正确应用

```python
from model import create_elecfm_with_lora

model = create_elecfm_with_lora(horizon=24)

# 检查可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# 预期: ~1.1M / 178M (0.62%)

# 检查梯度流通
model.train()
x = torch.randn(2, 168)
q_pred, spike_logits = model(x)
loss = q_pred.sum() + spike_logits.sum()
loss.backward()

for name, p in model.named_parameters():
    if 'lora_' in name:
        assert p.grad is not None, f"{name} 没有梯度"
print("✅ LoRA 梯度流通正常")
```

### 5.2 监控过拟合

关键指标（每 epoch 记录）：
```
Train Pinball 应该下降
Val Pinball 应该跟随下降或持平（早期）
如果 Val Pinball 连续上升 → 过拟合
```

### 5.3 常见问题

**Q: Val Pinball 从 epoch 1 就上升？**
A: 学习率过高，降低 10x 再试。

**Q: 训练速度很慢？**
A: 检查 use_amp=true 且设备是 CUDA/MPS。

**Q: Checkpoint 文件很大？**
A: LoRA 应该只有 ~20MB，如果还是 650MB，检查是否正确应用 LoRA。

---

## 6. 依赖安装

```bash
# 核心依赖（TimesFM 虚拟环境）
external/timesfm/.venv/bin/python -m pip install peft

# 验证安装
external/timesfm/.venv/bin/python -c "from peft import LoraConfig; print('OK')"
```

---

**维护者**：Claude Code  
**最后更新**：2026-07-02
