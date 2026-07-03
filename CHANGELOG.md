# 变更日志

> 按时间顺序记录 ElecFM 项目的关键变更

---

## 2026-07-02（续）

### 📊 ElecFM v4 结果：LoRA 有改善但未解决根本问题
- ✅ 最优 epoch 从 1 → 4，过拟合有所缓解
- ✅ W1 Spike-F1(mean)=0.388，首次超越零样本 0.329
- ❌ W1 SMAPE=29.39，仍比零样本 27.67 差
- ❌ τ*=0.05，spike head 退化

### ✅ 过夜实验完成（5个 V6/V7 变体，架构搜索）

结论：原始 V6（d_attn=64，3节点）依然最优。

- **V6 allgroups**（5组×3节点）：epoch 2 起 NaN，原因是跨价格区间节点混合训练数值不稳定；W1 Spike-F1=0.3840
- **V7 SwiGLU adapter**（v5a + 零初始化旁路）：SMAPE=27.55 ✅（零初始化验证），但 Spike-F1=0.3980 < v5a 0.4159，未提升
- **V6 d_attn=32/64/128 敏感性**：64 是最优，32=0.3967，128=0.3819；倒 U 型曲线
- **V6 allgroups+d32**：训练稳定（无 NaN），Spike-F1=0.3716，最差

论文价值：d_attn 敏感性表格、allgroups NaN 支撑节点选择合理性、V7 阴性结果可写入局限性章节。

### ✅ 文档全面更新（最终结果记录）

- `docs/结构消融汇报材料.md` 第七节：补全 7.5-7.9（v3/v4/v5a/v5b/V6 完整实验记录 + 研究贡献总结）
- `docs/fusion/design.md`：更新 4.0 节为全版本汇总表
- `docs/fusion/experiments.md`：完整记录所有版本结果
- `docs/TODO.md`：标记最终结果分析完成

### ✅ ElecFM V6 完成（CrossNodeAttention，W1 Spike-F1=0.4439）

**结果**：CrossNodeAttention 有效，W1 Spike-F1 从 0.4159 → **0.4439**（+6.7%），
与 Chronos skip_variate 消融退化幅度完全一致，实验闭环成立。
τ*=0.80（保守预测，精确率优先），W2 Spike-F1(head)=0.1905 优于 mean 信号 0.0893。
W3 受 OOD（Jan 2026 极端事件）影响，spike head 表现不佳（预期内）。

### ✅ ElecFM V6 实现完成（CrossNodeAttention）

- 新增 `CrossNodeAttention` 类（330K 参数，d_attn=64）
- 新增 `ElecFMV6` 类（共享骨干 + 跨节点注意力）
- 新增 `build_datasets_v6`（3节点同步滑窗，LZ_LCRA/LZ_WEST/LZ_RAYBN）
- 新增 `run_evaluation_v6`（3节点同时推理的评估流程）
- 更新 `train.py` 支持 `cross_node_only` 模式
- 新建配置 `electfm_ercot_full_v6.yaml`
- 验证：可训练参数 664K，参数/样本比 79.9，梯度流通正确

### ✅ ElecFM v5a 完成（168h context）

**结果**：核心目标全部达成

| 指标 | 目标 | 实际 |
|------|------|------|
| W1 SMAPE | ≈ 27.67 | **27.55** ✅ |
| W1 Spike-F1(head) | > 0.388 | **0.4159** ✅ |
| τ* | 合理 | **0.30** ✅ |

W1 Spike-F1: 0.329（零样本）→ **0.4159**（+26%）

### ✅ ElecFM v5b 完成（720h context）

W1 SMAPE=27.52（vs v5a 27.55，几乎无差），Spike-F1=0.3779（低于 v5a 0.4159）。
720h 上下文过长导致尖峰信号稀释，v5a（168h）更优。
**最终最优模型：v5a（168h，Spike-F1=0.4159）**

**注**：lambda sweep 对 spike_head_only 模式无意义——pinball 梯度不经过 spike_head，
改 λ 只等于改学习率，优化方向相同，结果无差别。

### 🔄 战略调整：转向纯 Spike Head 方案（v5）
- **发现**：四轮实验 SMAPE 始终退化，是结构性矛盾而非超参数问题
- **决策**：完全冻结骨干，只训练 spike head（~340K 参数）
- **核心 claim 调整**：SMAPE 等于零样本，spike head 增加尖峰检测能力
- **预期**：SMAPE ≈ 27.67，Spike-F1 > 0.388
- **代码变更**：train.py 新增 `spike_head_only` 模式，监控 val spike loss，自动跳过 Stage 2
- **配置文件**：`configs/fusion/electfm_ercot_full_v5.yaml`
- **注 1**：曾误引用 `halve_heads` 消融（+40%）作为反对 GroupSelfAttention 的依据，已更正
- **注 2**：曾误判"168h 已足够"，实为只看了文字摘要而非完整数值表。Ablation B 完整数据：W1 SMAPE 720h=26.84 vs 168h=27.67，720h 明确更优，v5 已更新为 720h context
- **内存验证**：MPS M3 Pro 上 720h + batch_size=32 无 OOM，前向 0.65s/batch

---

## 2026-07-02

### ✅ LoRA 实现完成
- **问题**：全 ERCOT 非 LoRA 训练严重过拟合（79M 参数 vs 125K 样本）
- **解决**：引入 LoRA 微调，可训练参数降至 1.1M（-98.6%）
- **实现**：
  - 安装 peft 库
  - 修改 `model.py` 添加 `create_elecfm_with_lora()`
  - 修改 `train.py` 适配 LoRA 训练逻辑
  - 创建 LoRA 配置文件
- **测试**：快速测试通过，W1 SMAPE 20.93（vs baseline 27.67）

### 📝 文档整理
- 新建 `docs/fusion/` 目录，合并分散的设计文档
- 归档旧文档到 `docs/archive/elecfm/`
- 创建归档索引 `docs/archive/elecfm/README.md`

---

## 2026-07-01

### ❌ 全 ERCOT 训练失败（非 LoRA）
- **配置**：15 节点，124K 样本，79M 可训练参数
- **问题**：严重过拟合
  - Val Pinball 从 epoch 1 持续上升（8.29 → 8.95）
  - 最优 checkpoint 在 epoch 1
- **诊断**：参数/样本比 = 634，严重失衡
- **决策**：改用 LoRA 方案

### ⚠️ ElecFM v2 部分成功
- **配置**：3 节点，quant_head 冻结
- **结果**：W1 SMAPE 29.19，Coverage 0.770
- **问题**：仍欠拟合，需要更多数据

### ❌ ElecFM v1 失败
- **配置**：3 节点，quant_head 可训练
- **问题**：预训练分位数校准被破坏
- **结果**：Coverage 0.775 → 0.582
- **修复**：quant_head 永久冻结

---

## 2026-06-30

### ✅ ElecFM v1 设计完成
- **架构**：基于 TimesFM-2.5，15 层 Transformer
- **创新点**：
  - 剪枝 5 个冗余层（20 → 15）
  - 双头输出：quantile head + spike head
  - spike head 从 L7（尖峰检测层）分叉
- **文档**：`fusion_model_design_v3.md`

---

## 2026-06-28

### ✅ v2.0 结构消融完成
- **实验**：36 次组件级 + 32 次逐层消融
- **关键发现**：
  - FFN 是核心引擎（移除退化 +419%~+486%）
  - 精度 vs 尖峰通路分离
  - TimesFM 40% 层可安全移除
- **决策**：采用 15 层保守方案

---

## 2026-06-25

### ✅ v1.0 参数消融完成
- **实验**：5 类消融 × 3 窗口 = 15 组
- **参与模型**：11 个（7 基线 + 4 基础模型）
- **结论**：确定 lambda_spike=0.2，batch_size=32 等默认参数

---

**维护者**：Claude Code
