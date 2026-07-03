# 实验 TODO

> 最后更新：2026-07-02

---

## 当前状态

| 阶段 | 状态 |
|---|---|
| v1.0 参数消融 | ✅ 完成 |
| v2.0 结构消融 | ✅ 完成 |
| ElecFM 第一轮（quant_head 可训，LR=1e-4）| ✅ 完成（失败，已分析原因）|
| ElecFM 第二轮（quant_head 冻结，LR=1e-5）| ✅ 完成（W1 SMAPE=29.19，Coverage=0.770）|
| lsp sweep（lambda=0.1/0.2/0.3，3 节点）| ❌ 已跳过 |
| 全 ERCOT 15 节点训练（非 LoRA）| ❌ 严重过拟合（分析见下方）|
| **LoRA 实现** | ✅ **已完成（可训练参数 1.87M）**|
| 全 ERCOT 15 节点训练（LoRA，v4）| ✅ **完成（SMAPE 仍退化，战略调整）**|
| ElecFM v5a（168h，纯 spike head）| ✅ **完成（W1 SMAPE=27.55，Spike-F1=0.4159）** |
| ElecFM v5b（720h，纯 spike head）| ✅ 完成（Spike-F1=0.3779，差于 v5a）|
| ElecFM V6（CrossNodeAttention）| ✅ 完成（W1 Spike-F1=0.4439，最优）|
| V6/V7 过夜实验（5个变体）| ✅ 完成（全部差于原始 V6，验证原始设计最优）|
| 最终结果分析与文档 | ✅ **完成** |

---

## 执行顺序

### [x] ~~1. lsp sweep~~（已决定跳过）

lsp sweep 未运行。3 节点 / 25K 样本下模型严重过拟合（val pinball epoch 1 就上升），在此状态下调 lambda 结论不可靠。先解决数据量问题，等全 ERCOT 跑完后如有需要再补做。

---

### [x] 1. 跑全 ERCOT 15 节点训练（第一轮完成）

**执行时间**：2026-07-01 夜间

**结果**：❌ **严重过拟合，优化方案见下方**

**关键观察指标**：
| 指标 | Stage 1 Epoch 1 | Stage 1 Epoch 10 | 趋势 |
|------|-----------------|------------------|------|
| Train Pinball | 3.87 | 1.47 | ✅ 持续下降 |
| Val Pinball | **8.29** | 8.95 | ❌ **持续上升** |
| Train Spike | 1.09 | 0.55 | ✅ 下降 |
| Val Spike | 1.44 | 1.58 | ❌ 上升 |

**零样本基准验证**：
- 15层 ElecFM（零样本）SMAPE = 27.55 vs 原始 20层 TimesFM = 27.67
- 退化 = **-0.4%**（实际略微提升）✅ 剪枝基准通过

**问题诊断**：
1. **参数/样本比例失衡**：Stage 1 可训练参数 79M / 训练样本 124K ≈ **637 参数/样本**
2. **训练轮数过多**：最优 checkpoint 出现在第 1 epoch，后续 9 个 epoch 都在过拟合
3. **Stage 2 同样恶化**：解冻 149M 参数后第 1 个 epoch val pinball 从 8.29→8.44

**根本原因**：即使 125K 样本，对于 79M 可训练参数仍然严重不足，模型在记忆而非学习泛化。

---

### [x] 2. 实施 LoRA 优化方案 ✅ 已完成（2026-07-02）

**决策**：采用 **LoRA + 早停 + Weight Decay** 组合策略

**实现结果**：
| 指标 | 非 LoRA | LoRA 实现 | 改进 |
|------|---------|-----------|------|
| 可训练参数 | 79M | **1.1M** | -98.6% |
| 参数占比 | 48.5% | **0.62%** | -98% |
| 参数/样本比 | 637 | **8.8** | -99% ✅ |
| 训练时间 | ~27h | **~7h**（预估）| 4x |

**TimesFM 适配的 LoRA 配置**：
```python
lora_config = LoraConfig(
    r=4,                                    # 低秩维度
    lora_alpha=8,                           # 缩放因子
    target_modules=["attn.qkv_proj", "attn.out", "ff0", "ff1"],  # TimesFM 命名
    lora_dropout=0.05,
    bias="none",
)
```

**修改文件**：
- ✅ `src/fusion_model/model.py` - 添加 `create_elecfm_with_lora()`
- ✅ `src/fusion_model/train.py` - 适配 LoRA 训练（冻结/解冻逻辑）
- ✅ `src/fusion_model/run_fusion.py` - 配置文件支持 `use_lora`
- ✅ `configs/fusion/electfm_ercot_full_lora.yaml` - LoRA 配置文件

**快速测试**：`bash test_lora_quick.sh`

---

### [x] 3. 运行全 ERCOT LoRA 训练（v4，已完成）

**结果**：⚠️ 改善但未达目标

- ✅ 最优 epoch 从 1 → 4，过拟合缓解
- ✅ W1 Spike-F1(mean)=0.388，首次超越零样本 0.329
- ❌ W1 SMAPE=29.39，仍比零样本 27.67 差
- ❌ Spike head τ*=0.05，退化严重

**诊断**：任何骨干微调都会导致 SMAPE 退化，这是结构性矛盾，非超参数问题。

---

### [x] 4. 纯 Spike Head 训练（v5，已就绪）

**实现完成**：
- ✅ `train.py` 新增 `spike_head_only` 模式，冻结全部骨干，监控 val spike loss
- ✅ `run_fusion.py` 透传 `spike_head_only` 参数
- ✅ 配置文件 `configs/fusion/electfm_ercot_full_v5.yaml`

**运行命令**：
```bash
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_v5.yaml \
    2>&1 | tee run_ercot_v5.log
```

**预期时间**：~2 小时

**成功标准**：
- W1 SMAPE ∈ [27.5, 27.8]（等于零样本）
- W1 Spike-F1(head) > 0.388（超过 mean 信号）

---

### [x] 5. v5 实验完成，最终最优模型确定

**v5a（168h，spike_head_only）为最终最优模型**：
- W1 SMAPE = 27.55 ✅（≈ 零样本水平）
- W1 Spike-F1(head) = 0.4159 ✅（> 目标 0.388，比零样本 0.329 提升 +26%）
- τ* = 0.30（合理）

**v5b（720h）对比结论**：
- W1 SMAPE 几乎相同（27.52 vs 27.55）
- W1 Spike-F1 反而更低（0.3779 < 0.4159）
- 720h 上下文过长导致尖峰信号被稀释，不如 168h

**lambda sweep 不适用于 v5**：
spike_head_only 模式下 pinball 梯度不经过 spike_head，改 λ 只等于改学习率，方向完全相同，结果无差别。lambda sweep 只在 pinball 和 spike 同时有梯度时（如 LoRA 版本）才有意义。

---

### [ ] 6. 最终横向对比与文档

- 填写最终数字到 `docs/结构消融汇报材料.md` 第七节
- 填写 `docs/fusion/design.md` Section 4 实验表格
- 运行 `python src/fusion_model/compare_sweep.py` 生成对比图

---

### [ ] 8. Commit & Push

```bash
git add -A
git commit -m "feat(ElecFM): 完成 v5 训练，填入最终结果"
git push
```

---

### [ ] 9. 整理论文/汇报材料

方法论和训练过程已写好（`docs/结构消融汇报材料.md` 第七节），只需填最终数字。

---

## 等训练时可以同时做

- [ ] 更新 README 实验进度
- [ ] 检查三份汇报材料格式是否一致

---

## 关键文件路径

| 内容 | 路径 |
|---|---|
| 主入口 | `src/fusion_model/run_fusion.py` |
| 全 ERCOT 配置 | `configs/fusion/electfm_ercot_full.yaml` |
| lsp sweep 脚本 | `run_fusion_sweep.sh` |
| 结果对比脚本 | `src/fusion_model/compare_sweep.py` |
| 设计文档 | `docs/specs/fusion_model_design_v3.md` |
| 汇报材料 | `docs/结构消融汇报材料.md`（含 ElecFM 第七节）|
| 实验结果 | `data/results/fusion_*/` |
| Checkpoint | `data/checkpoints/electfm*/`（已加入 .gitignore）|
