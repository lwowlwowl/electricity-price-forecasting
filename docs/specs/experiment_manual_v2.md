# 电价预测：结构级消融与模型融合实验手册

> 版本：v2.0 ｜ 适用项目：`school/`（电价预测）
> 前置文档：`experiment_manual.md`（v1.0 参数级消融）
> 读者：负责模型方向的科研同学
> 目标：在 v1.0 "旋钮消融"（上下文长度、协变量、单/多变量、预测步长）的基础上，深入模型内部，做**结构级消融**——逐一移除或替换 Transformer 内部组件（注意力层、位置编码、输出头、Patch 机制等），定位每个模型在电价预测上的"强零件"，最终将三个基础模型的优秀组件**融合**成一个专用于电价预测的新模型。

---

## 0. 本手册与 v1.0 的关系

v1.0 手册定义了一套可控变量、可复现的实验框架，解决的是**参数级消融**——固定模型架构，只调整输入/配置旋钮（上下文长度、协变量集合、单/多变量、预测步长），观察哪些外部设置对各模型的性能影响最大。

v1.0 的消融矩阵（A–F）产出的结论形如"Chronos-2 在有协变量时提升最大""Toto 在多节点联合时表现最好"。这些结论告诉我们**什么场景下该选谁**，但没回答更深层的问题：**模型内部的哪些架构组件是性能的关键来源？**

本手册承接 v1.0 的结论和实验基础设施（统一框架、滚动回测、指标体系、配置驱动），进入下一阶段：

```
v1.0 参数级消融 → 识别"场景-模型"强项匹配
    ↓（本手册）
v2.0 结构级消融 → 识别"模型内部的关键组件"
    ↓
v2.0 模型融合  → 将各模型的强组件拼装成一个融合模型
    ↓
v2.0 微调训练  → 在电价数据上训练融合模型
```

**先决条件**：开始 v2.0 之前，v1.0 消融 A–D（零样本即可完成的那四组）的基准结果应已跑完。这些结果提供了每个模型在"完整架构"下的性能基准线，结构级消融的意义在于看"去掉某组件后性能下降多少"，需要和这条基准线做差值比较。

---

## 1. 三个基础模型的架构解剖

要做结构级消融，首先必须精确理解三个模型内部长什么样。以下是基于源码的完整解剖。

### 1.1 Toto-1.0（Datadog，约 150M 参数）

**源码位置**：`external/toto/toto/model/`

Toto 的核心思想是用交替的"时间注意力+空间注意力"同时建模时序依赖和变量间相关性。其数据流：

```
原始序列
  → CausalPatchStdMeanScaler（因果归一化，Welford 在线算法）
    → PatchEmbedding（线性投影，patch_size=64, stride=64）
      → Transformer（12 层，交替 TimeWise / SpaceWise 注意力）
        → Unembed（Linear: embed_dim → embed_dim × patch_size）
          → MixtureOfStudentTsOutput（24 分量 Student-t 混合分布）
```

**关键超参数**（来自 `config.json`）：

| 参数 | 值 | 说明 |
|---|---|---|
| embed_dim | 768 | 隐藏维度 |
| num_layers | 12 | Transformer 层数 |
| num_heads | 12 | 注意力头数 |
| mlp_hidden_dim | 3072 | FFN 隐藏层（SwiGLU） |
| patch_size / stride | 64 / 64 | 无重叠 Patch |
| spacewise_every_n_layers | 12 | 每 12 层 TimeWise 后接 1 层 SpaceWise |
| output_distribution | MixtureOfStudentTs | 24 分量混合 Student-t |
| scaler | CausalPatchStdMeanScaler | 因果 Patch 级归一化 |

**各组件详细说明**：

**1. CausalPatchStdMeanScaler**（`scaler.py`）：对输入序列做因果归一化——在每个 Patch 边界上用 Welford 在线算法计算到当前时刻为止的 running mean 和 std，再做 z-score。"因果"意味着不偷看未来数据。`stabilize_with_global=True` 会用全局统计量钳制极端值。

**2. PatchEmbedding**（`embedding.py`）：将归一化后的序列按 patch_size=64 切片，每片通过一个 `Linear(patch_size, embed_dim)` 投影到 768 维嵌入空间。stride=64 表示无重叠。

**3. Transformer**（`transformer.py`）：12 层 TransformerLayer，每层结构为 Pre-RMSNorm → Attention → 残差 → Pre-RMSNorm → SwiGLU FFN → 残差。注意力类型由 `_get_layer_types()` 决定：当前 `spacewise_every_n_layers=12` 意味着前 11 层都是 TimeWise，最后 1 层是 SpaceWise（`spacewise_first=False`）。

**4. TimeWiseMultiheadAttention**（`attention.py`）：沿时间轴做因果自注意力，使用 fused QKV 投影和 TimeAwareRotaryEmbedding（RoPE + xpos 缩放）。因果掩码确保每个 patch 只能看到自己及之前的 patch。

**5. SpaceWiseMultiheadAttention**（`attention.py`）：沿变量轴做双向注意力（无因果掩码），不使用 RoPE。用 `id_mask` 实现分组——同 group 的变量互相可见，不同 group 不可见。这是 Toto 多变量建模的核心机制。

**6. TimeAwareRotaryEmbedding**（`rope.py`）：在标准 RoPE 基础上加入 xpos 缩放（长距离位置衰减），帮助模型处理长序列。

**7. Unembed**：`Linear(embed_dim, embed_dim × patch_size)` 将 Transformer 输出映射回 patch 长度维度。

**8. MixtureOfStudentTsOutput**（`distribution.py`）：用 24 个 Student-t 分量的混合分布作为输出。每个分量有独立的自由度 df、位置 loc、尺度 scale，外加 softmax 混合权重。能捕捉电价的尖峰/厚尾特性。

**9. Fusion 模块**（`fusion.py`）：可选功能，在序列前端拼接"变量标签嵌入"以区分目标变量和外生变量。微调时通过 `enable_variate_labels()` 启用。

### 1.2 Chronos-2（Amazon，约 40M 参数）

**源码位置**：`external/chronos-forecasting/src/chronos/chronos2/`

Chronos-2 是 encoder-only 架构，核心特点是原生支持协变量和多序列联合预测（通过 GroupSelfAttention 实现跨序列信息共享）：

```
原始序列
  → InstanceNorm（逐实例归一化）
    → Patch（切片，input_patch_size / input_patch_stride）
      → InputPatchEmbedding（ResidualBlock 堆叠）
        → Encoder（6 层，每层：TimeSelfAttention → GroupSelfAttention → FFN）
          → OutputPatchEmbedding（映射到输出 patch 并展开）
            → 分位数输出（直接输出 9 个分位数，非参数化分布）
```

**关键超参数**（来自 `config.py`）：

| 参数 | 值 | 说明 |
|---|---|---|
| d_model | 512 | 隐藏维度 |
| num_layers | 6 | Encoder 层数 |
| num_heads | 8 | 注意力头数 |
| d_kv | 64 | 每头 Key/Value 维度 |
| d_ff | 2048 | FFN 隐藏层 |
| feed_forward_proj | relu | FFN 激活函数（非 gated） |
| dropout_rate | 0.1 | Dropout |
| rope_theta | 10000.0 | RoPE 基频 |

**各组件详细说明**：

**1. InstanceNorm**（继承自 Chronos-Bolt）：对每条输入序列独立做标准化（减均值除标准差），与 Toto 的因果 Patch 级 Scaler 不同，这里是全局 Instance 级。

**2. Patch + InputPatchEmbedding**：先按 `input_patch_size` / `input_patch_stride` 切片，然后通过 ResidualBlock（Linear → 激活 → Linear + 残差跳连）将每个 patch 映射到 d_model=512 维。

**3. Chronos2EncoderBlock**（`model.py`）：每个 block 严格按顺序执行三个子层：
   - **TimeSelfAttention**（`layers.py`）：沿时间轴的因果自注意力，使用 RoPE 位置编码。注意 Chronos-2 的注意力使用 T5-style RMSNorm（`Chronos2LayerNorm`），与 Toto 的标准 RMSNorm 略有不同。
   - **GroupSelfAttention**（`layers.py`）：**沿 batch 维度**做双向注意力——这是 Chronos-2 的独特设计。当多条序列标记为同一 `group_id` 时，它们的 patch 嵌入可以互相关注。这不同于 Toto 的 SpaceWise（沿变量维度），而是沿样本维度做跨序列注意力。不使用 RoPE。
   - **FeedForward**（`layers.py`）：Pre-RMSNorm → MLP（d_model → d_ff → d_model），激活函数为 ReLU。

**4. Encoder**（`model.py`）：6 个 EncoderBlock 堆叠 + 最终 RMSNorm + Dropout。

**5. OutputPatchEmbedding**：将编码器输出映射到 `output_patch_size × n_quantiles` 维，reshape 后直接输出各时刻的分位数预测。

**6. 分位数输出头**：Chronos-2 **不使用参数化概率分布**（不像 Toto 的 Student-t 混合），而是直接回归预设的 9 个分位数（如 q0.1, q0.2, ..., q0.9）。训练时用 Pinball Loss。

**7. 协变量处理**：Chronos-2 原生支持未来协变量（`future_covariates`），通过额外的 patch 通道与目标序列拼接后一起输入 Encoder。

### 1.3 TimesFM-2.5（Google，200M 参数）

**源码位置**：`external/timesfm/src/timesfm/`

TimesFM 是一个"纯 decoder"堆叠 Transformer，20 层深、1280 维宽，是三个模型中最大的：

```
原始序列
  → 输入归一化（可选）
    → Tokenizer（ResidualBlock：Linear → Swish → Linear + 残差）
      → 20 层 Stacked Transformer（每层：RMSNorm → MultiHeadAttention(RoPE) → 残差 → RMSNorm → FFN → 残差）
        → Output Projection Point（ResidualBlock → 点预测）
        → Output Projection Quantiles（ResidualBlock → 9 个分位数 × output_patch=128）
```

**关键超参数**（来自 `timesfm_2p5_base.py`）：

| 参数 | 值 | 说明 |
|---|---|---|
| model_dims | 1280 | 隐藏维度 |
| num_layers | 20 | Transformer 层数 |
| num_heads | 16 | 注意力头数 |
| hidden_dims（FFN） | 1280 | FFN 隐藏层（与 model_dims 相同） |
| patch_size（input） | 32 | 输入 Patch 大小 |
| output_patch | 128 | 输出 Patch 大小 |
| ff_activation | swish | FFN 激活函数 |
| qk_norm | rms | QK 归一化 |
| fuse_qkv | True | 融合 QKV 投影 |
| context_limit | 16384 | 最大上下文 token 数 |

**各组件详细说明**：

**1. Tokenizer**（ResidualBlock）：与 Chronos-2 的 InputPatchEmbedding 类似，是一个 `Linear(input_dims, hidden_dims) → Swish → Linear(hidden_dims, output_dims) + 残差跳连` 的结构，将每个 input patch（32 时间步）映射到 1280 维。

**2. Stacked Transformer**（`transformer.py`）：20 层完全相同的 Transformer 层，每层结构：

   - Pre-RMSNorm → MultiHeadAttention → 残差
   - Pre-RMSNorm → FFN → 残差

   TimesFM 的注意力有几个独特设计：
   - **PerDimScale**：在 Q 上对每个维度施加可学习的缩放因子（初始化为 1），替代标准的 `1/√d_k` 缩放。
   - **QK-Norm**：对 Q 和 K 分别做 RMSNorm 再点积，稳定深层训练。
   - **RoPE**（无 xpos）：标准旋转位置编码，不带 Toto 那样的 xpos 衰减。
   - **fuse_qkv=True**：Q/K/V 投影合并为一个大矩阵乘法。

**3. FFN**：`Linear(1280, 1280) → Swish → Linear(1280, 1280)`，隐藏维度等于模型维度（不像 Toto 用 4× 扩展的 SwiGLU）。

**4. 输出投影**：
   - **Point Head**：ResidualBlock(1280 → 1280 → 1280)，产生点预测。
   - **Quantile Head**：ResidualBlock(1280 → 1280 → 10240)，其中 10240 = 1280 × 8，对应 output_patch_size=128 的多步分位数预测。最终 reshape 为 `(128, 9+1)` 得到 9 个分位数和 1 个均值。

**5. 单变量为主**：TimesFM 逐条序列预测，没有 Toto 的 SpaceWise 或 Chronos-2 的 GroupSelfAttention 那种跨变量/跨序列机制。协变量通过独立通道处理。

### 1.4 三模型架构对比总表

| 组件 | Toto-1.0 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|
| **总体规模** | ~150M, 12 层 | ~40M, 6 层 | ~200M, 20 层 |
| **归一化** | CausalPatch StdMean（Welford） | InstanceNorm | 可选输入归一化 |
| **Patch 机制** | Linear(64→768), 无重叠 | ResidualBlock, 可重叠 | ResidualBlock(32→1280) |
| **时间注意力** | TimeWise (因果, RoPE+xpos) | TimeSelfAttention (因果, RoPE) | MultiHead (因果, RoPE, PerDimScale, QK-Norm) |
| **跨变量注意力** | SpaceWise (双向, 无RoPE, id_mask) | GroupSelfAttention (沿batch, 无RoPE) | **无** |
| **FFN** | SwiGLU (768→3072) | ReLU MLP (512→2048) | Swish MLP (1280→1280) |
| **层间 Norm** | RMSNorm (pre-norm) | T5-style RMSNorm (pre-norm) | RMSNorm (pre-norm, + QK-Norm) |
| **位置编码** | RoPE + xpos | RoPE | RoPE |
| **输出头** | Mixture-of-StudentTs (24分量) | 直接分位数 (9个) | Point + 分位数 (ResidualBlock) |
| **多变量** | ✓ 原生 (SpaceWise) | ✓ 原生 (Group) | ✗ 逐序列 |
| **协变量** | 实验性 (Fusion模块) | ✓ 原生 | ✓ (通过covariate接口) |

---

## 2. 结构级消融的实验设计

### 2.1 核心思路

参数级消融的"旋钮"是输入配置；结构级消融的"旋钮"是模型内部组件。具体操作是：

```
结构级消融 = 取一个完整的预训练模型
            → 外科手术式地移除或替换某个内部组件
            → 在完全相同的评估协议下（同 v1.0 的滚动回测）跑预测
            → 对比"完整版"与"缺损版"的指标差异
            → 差异越大 → 该组件越关键
```

**注意**：第一批结构级消融（S-G1、S-A1/A2、S-B1）是**零样本**的——直接在预训练权重上关闭某个模块，不重新训练。**之所以不训练，是因为目的是诊断而非改造**：如果关掉某组件后重新训练，模型的其他层会自适应补偿被移除组件的功能，你测出来的就不是"这个组件原本有多重要"，而是"模型能不能在没有它的情况下重新学会"——这两个问题完全不同。零样本消融避免了这种补偿效应，直接暴露每个组件在当前预训练权重中的真实贡献。

### 2.2 实验基准配置

沿用 v1.0 的基准，确保与参数级消融的结论可比：

```
市场: ERCOT | 节点: volatility(3个) | 频率: 1h | 上下文: 168 | 步长: 24
评估: ≥30 起报点滚动回测 | 指标: MAE / RMSE / MASE / Spike-F1 / Pinball
```

所有结构级消融都固定这套基准，只改变模型内部结构。

### 2.3 消融实验矩阵

以下按"可消融的组件类型"组织实验。每个实验对三个模型分别执行（如果该模型有对应组件的话）。

#### S-A：注意力机制消融

**目标**：量化不同注意力机制对电价预测的贡献。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-A1 | 移除 SpaceWise 注意力层 | Toto | 将 SpaceWise 层替换为恒等映射（输入直接跳过） | 跨变量注意力对电价预测有多重要？ |
| S-A2 | 移除 GroupSelfAttention | Chronos-2 | 在每个 EncoderBlock 中跳过 `self.layer[1]`（GroupSelfAttention） | 跨序列信息共享对 Chronos-2 贡献多大？ |
| S-A3 | 减少注意力头数（halve） | 三个模型 | 只保留前 n/2 个头，其余头置零 | 模型是否过参数化？头数冗余度？ |
| S-A4 | 替换因果注意力为双向注意力 | 三个模型 | 移除因果掩码（时间维度变为双向） | 因果约束对预测质量的影响？ |

**S-A1 实现细节**（Toto SpaceWise 移除）：

```python
# 在 TotoBackbone 中，找到 SpaceWise 层并替换
for i, layer in enumerate(model.model.transformer.layers):
    if layer.attention_axis == AttentionAxis.SPACE:
        # 方案一：恒等映射（保留 FFN）
        layer.attention = IdentityAttention()
        # 方案二：完全跳过该层（更激进）
        # model.model.transformer.layers[i] = nn.Identity()
```

SpaceWise 在 Toto-1.0 中 `spacewise_every_n_layers=12` 且 `spacewise_first=False`，意味着只有第 12 层（最后一层）是 SpaceWise。移除它等于去掉唯一的跨变量交互。如果性能不降，说明对电价单变量场景空间注意力是冗余的；如果显著下降，说明即使在单变量模式下 SpaceWise 也通过 `id_mask` 编码了有用信息。

#### S-B：位置编码消融

**目标**：评估不同位置编码方案对时序建模的影响。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-B1 | 移除 RoPE | 三个模型 | 将 rotary embedding 置为 no-op（不旋转 Q/K） | 位置编码对时序预测到底有多重要？ |
| S-B2 | 移除 xpos 缩放 | Toto | 保留 RoPE 但禁用 xpos（`use_xpos=False`） | xpos 的长距离衰减对电价序列有帮助吗？ |
| S-B3 | 替换 RoPE 为绝对位置编码 | 三个模型 | 用可学习的绝对位置嵌入替换 RoPE | RoPE vs 绝对编码哪个更适合电价？ |
| S-B4 | 移除 PerDimScale | TimesFM | 将 per_dim_scale 替换为标准 1/√d_k 缩放 | TimesFM 的 PerDimScale 设计有多大贡献？ |
| S-B5 | 移除 QK-Norm | TimesFM | 跳过 Q/K 的 RMSNorm | QK-Norm 对稳定性的作用？ |

**S-B1 实现细节**（移除 RoPE）：

```python
# Toto: 在 TimeAwareRotaryEmbedding.forward() 中直接返回未旋转的输入
# 或更简洁地：把 rotary_emb 替换为 identity
class NoOpRoPE(nn.Module):
    def forward(self, q, k, **kwargs):
        return q, k

model.model.transformer.rotary_emb = NoOpRoPE()
```

#### S-C：Patch 嵌入消融

**目标**：理解 Patch 策略对时间粒度感知的影响。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-C1 | 改变 Patch 大小 | 三个模型 | 重新初始化 PatchEmbedding 的权重（随机初始化），patch_size 从 64/32 变为 16/8 | 更细粒度的 Patch 对电价波动捕捉更好吗？ |
| S-C2 | 替换 Patch 类型 | Toto vs Chronos-2 | Toto 的 Linear 投影 vs Chronos-2 的 ResidualBlock | 哪种 Patch 嵌入更好？ |
| S-C3 | 添加 Patch 重叠 | Toto | 将 stride 从 64 改为 32（50% 重叠） | 重叠 Patch 能否捕捉到更多跨 Patch 边界的模式？ |

**重要提醒**：改变 Patch 大小会改变序列长度，进而影响注意力的计算量和位置编码的有效范围。S-C1 实验必须同时调整相关组件（如 Unembed 层的维度），且因为权重需要重新初始化，**该实验需要训练**而非纯零样本消融。建议先用小规模数据快速训练验证方向，再决定是否投入完整训练。

#### S-D：输出头消融

**目标**：对比不同输出策略对电价预测（尤其是尖峰捕捉）的影响。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-D1 | Student-t → 分位数 | Toto | 替换 MixtureOfStudentTs 为直接分位数回归 | 参数化分布 vs 非参数分位数，谁更好捕捉尖峰？ |
| S-D2 | 分位数 → Student-t | Chronos-2, TimesFM | 替换分位数输出为 Student-t 分布头 | 反向验证 |
| S-D3 | 减少混合分量数 | Toto | k_components 从 24 减至 4/8/12 | 24 分量是否过多？边际收益在哪？ |
| S-D4 | Student-t → Gaussian | Toto | 用 Normal 分布替换 Student-t | 厚尾分布对电价尖峰预测的贡献？ |

**S-D1/D2 需要训练**：替换输出头后权重随机初始化，必须在电价数据上训练新头部（可冻结 Transformer backbone 只训练输出头，即 Head-only Fine-tuning）。

#### S-E：归一化策略消融

**目标**：归一化是时序模型的关键预处理，不同策略可能对电价的极端分布有不同表现。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-E1 | 因果归一化 → InstanceNorm | Toto | 替换 CausalPatchStdMeanScaler 为简单的均值/标准差归一化 | 因果约束对归一化真的必要吗？ |
| S-E2 | InstanceNorm → 因果归一化 | Chronos-2 | 在 Chronos-2 前端加上因果归一化 | 因果归一化能否提升 Chronos-2？ |
| S-E3 | 移除层间 RMSNorm | 三个模型 | 将 Transformer 层间的 RMSNorm 替换为恒等映射 | LayerNorm/RMSNorm 对推理稳定性的贡献？ |

**S-E1/E2 为零样本**：只修改前端归一化，不涉及权重变化。S-E3 通常会导致数值爆炸，预期性能急剧下降——这是一个"验证直觉"的消融，目的是确认 Norm 层的不可或缺性。

#### S-F：FFN 与激活函数消融

**目标**：三个模型的 FFN 设计差异很大（SwiGLU vs ReLU vs Swish），这些差异是否影响电价预测表现？

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-F1 | SwiGLU → ReLU MLP | Toto | 替换 SwiGLU FFN 为标准 ReLU MLP（需训练适配） | SwiGLU 相比 ReLU 有多大优势？ |
| S-F2 | ReLU → SwiGLU | Chronos-2 | 替换 FFN 模块（需训练适配） | Chronos-2 能否受益于更强的 FFN？ |
| S-F3 | 缩减 FFN 隐藏维度 | 三个模型 | 将 FFN 隐藏层维度减半 | FFN 容量的冗余度？ |

**S-F1/F2 需要训练**：替换激活函数后权重不兼容，需要重新训练 FFN 层（冻结其余组件）。

#### S-G：层深度消融

**目标**：定位"关键层"——哪些 Transformer 层贡献最大，哪些可以去掉。

| 编号 | 操作 | 适用模型 | 实现方式 | 想回答的问题 |
|---|---|---|---|---|
| S-G1 | 逐层移除 | 三个模型 | 每次跳过第 i 层（输入直接传给第 i+1 层），遍历 i=1..N | 哪一层的移除导致最大性能下降？ |
| S-G2 | 只保留前 k 层 | 三个模型 | 截断模型，只用前 k 层的输出（k = N/4, N/2, 3N/4） | 模型的"有效深度"是多少？ |
| S-G3 | 只保留后 k 层 | 三个模型 | 跳过前面的层，只用最后 k 层 | 浅层特征 vs 深层特征，谁对电价更重要？ |

**S-G1 是最核心的结构消融**，直接产出"每层的重要性分数"：

```python
# 伪代码：逐层消融
baseline_metrics = evaluate(full_model)
layer_importance = {}

for i in range(model.num_layers):
    # 跳过第 i 层
    modified_model = skip_layer(model, layer_idx=i)
    ablated_metrics = evaluate(modified_model)
    
    # 重要性 = 完整模型性能 - 缺损模型性能
    layer_importance[i] = {
        "mae_drop": ablated_metrics.mae - baseline_metrics.mae,
        "spike_f1_drop": baseline_metrics.spike_f1 - ablated_metrics.spike_f1,
    }

# 排序 → 得到"关键层排名"
```

**S-G1 的预期产出**：一张 12×3（或 6×3、20×3）的热力图，横轴是层编号，纵轴是指标（MAE/RMSE/Spike-F1），颜色深浅代表移除该层后的性能下降幅度。这张图直接告诉我们每个模型的哪些层是"核心层"。

### 2.4 实验优先级排序

结构消融的实验数量较多，建议分两批执行：

**第一批（必做，纯零样本，不涉及训练）**：

1. **S-G1 逐层移除** → 最高价值，直接产出层重要性排名，是后续融合的核心依据
2. **S-A1/A2 跨变量注意力移除** → 定量回答"多变量注意力有没有用"
3. **S-B1 移除 RoPE** → 定量回答"位置编码有没有用"
4. **S-E3 移除层间 Norm** → 验证性实验（预期崩溃）

**第二批（选做，需要训练或更多工程）**：

5. **S-D1/D2 输出头替换** → 需 Head-only Fine-tuning
6. **S-C1 Patch 大小改变** → 需训练
7. **S-F1/F2 FFN 替换** → 需训练

### 2.5 消融的工程实现

为保持与 v1.0 框架一致，结构消融也应**配置驱动**。建议在 YAML 中增加 `structural_ablation` 字段：

```yaml
# configs/experiments/structural/ablation_SG1_layer_skip.yaml
name: "structural_ablation_SG1_layer_skip"
base_config: ablation_baseline     # 继承基准的市场/节点/频率/回测设置

# 结构消融专用字段
structural_ablation:
  type: layer_skip                 # 消融类型
  target_model: toto               # 对哪个模型做手术
  parameters:
    skip_layers: [0]               # 跳过的层编号（遍历 0..11 共 12 个实验）
```

在 `run_experiment()` 中新增一个 `apply_structural_ablation(model, config)` 步骤：

```python
def apply_structural_ablation(model, ablation_config):
    """在推理前对模型做结构修改（不改权重文件，只修改内存中的模型对象）"""
    abl_type = ablation_config["type"]
    
    if abl_type == "layer_skip":
        skip_layers = ablation_config["parameters"]["skip_layers"]
        for idx in skip_layers:
            model.transformer.layers[idx] = IdentityLayer()
    
    elif abl_type == "remove_rope":
        model.transformer.rotary_emb = NoOpRoPE()
    
    elif abl_type == "remove_spacewise":
        for layer in model.transformer.layers:
            if layer.attention_axis == AttentionAxis.SPACE:
                layer.attention = IdentityAttention()
    
    # ... 更多类型
    return model
```

---

## 3. 从消融结果到融合模型

### 3.1 结构消融产出的"组件价值图"

结构消融完成后，整理成一张**组件价值图**——比 v1.0 的"能力地图"更深一层：

| 组件类型 | Toto-1.0 | Chronos-2 | TimesFM-2.5 | 电价预测最优选 |
|---|---|---|---|---|
| 关键层深度 | 待定（S-G1 结果） | 待定 | 待定 | 选"重要层最多"的 |
| 时间注意力 | RoPE+xpos | RoPE | RoPE+PerDimScale+QK-Norm | 待 S-B 确认 |
| 跨变量注意力 | SpaceWise (S-A1) | GroupSelfAttn (S-A2) | 无 | 看消融降幅 |
| FFN | SwiGLU 768→3072 | ReLU 512→2048 | Swish 1280→1280 | 待 S-F 确认 |
| 输出头 | MixtureStudentT(24) | 直接分位数(9) | Point+分位数 | 看 Spike-F1 表现 |
| 归一化 | CausalPatch | InstanceNorm | 可选 | 看 S-E 结果 |

### 3.2 融合模型的设计思路

融合的目标：**取各家之长，组装一个在电价预测上超越任何单模型的专用模型**。

融合分三个层次，按复杂度递进：

#### 层次一：组件级替换（Component Swap）

最直接的方式：以一个模型为"底座"，将其中表现不佳的组件替换为另一个模型中表现更好的同类组件。

```
例如（假设性结论，以实际消融结果为准）：
  底座 = Toto（综合最强的 Transformer backbone）
  替换输出头：Toto 的 MixtureStudentT → 借鉴 Chronos-2 的直接分位数（如果 S-D 显示分位数更好）
  添加 QK-Norm：借鉴 TimesFM 的设计（如果 S-B5 显示有帮助）
  保留 SpaceWise：如果 S-A1 显示有价值
```

**实现方式**：

```python
class FusionModel(nn.Module):
    """从各模型取最强组件拼装"""
    def __init__(self):
        # 底座：Toto 的 Patch + Transformer backbone
        self.scaler = TotoCausalPatchScaler()           # 来自 Toto
        self.patch_embed = TotoPatchEmbedding(64, 768)  # 来自 Toto
        
        # Transformer 层：混合设计
        self.layers = nn.ModuleList([
            FusionTransformerLayer(
                time_attention=...,      # 基于消融选择最佳方案
                space_attention=...,     # 基于 S-A1/A2 决定用哪种
                ffn=...,                 # 基于 S-F 决定 SwiGLU/ReLU/Swish
                norm=...,               # 基于 S-E 决定 Norm 类型
            )
            for _ in range(num_layers)   # 基于 S-G 决定层数
        ])
        
        # 输出头：基于 S-D 决定
        self.output_head = ...           # MixtureStudentT / 分位数 / 混合
```

#### 层次二：层级嫁接（Layer Grafting）

更精细的方式：基于 S-G1 的逐层消融结果，从不同模型中挑选"最强的层"拼接。

```
例如（假设性）：
  TimesFM 的第 3-8 层在 MAE 消融中表现最稳
  Toto 的 SpaceWise 层在多变量场景中独特
  Chronos-2 的浅层特征提取对协变量最友好
  
  融合策略：
  Layer 1-2: Chronos-2 浅层（协变量感知）
  Layer 3-8: TimesFM 中间层（时序模式提取）
  Layer 9:   Toto SpaceWise 层（跨变量交互）
```

**技术挑战**：不同模型的隐藏维度不同（768 / 512 / 1280），嫁接时需要线性投影层做维度适配：

```python
class DimensionAdapter(nn.Module):
    """维度适配器：连接不同维度的层"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.RMSNorm(out_dim)
    
    def forward(self, x):
        return self.norm(self.proj(x))
```

#### 层次三：知识蒸馏融合（Distillation Fusion）

最成熟的方式：不直接拼装组件，而是训练一个新的紧凑模型（Student），用三个预训练模型（Teacher）的输出做蒸馏：

```
Teacher 1 (Toto):     pred_toto     = Toto(input)
Teacher 2 (Chronos):  pred_chronos  = Chronos2(input)
Teacher 3 (TimesFM):  pred_timesfm  = TimesFM(input)

Student (新模型):      pred_student  = FusionStudent(input)

Loss = α × TaskLoss(pred_student, ground_truth)           # 硬标签
     + β × DistillLoss(pred_student, ensemble_teachers)   # 软标签
     + γ × FeatureLoss(student_hidden, teacher_hidden)    # 中间特征对齐（可选）
```

Student 模型的架构可以自由设计——借鉴消融中发现的最佳组件选择，但参数量可以远小于三个 Teacher 的总和。

### 3.3 融合模型的推荐路径

基于工程复杂度和预期收益，推荐按以下路径推进：

```
Step 1 → 先跑完 S-G1（逐层消融）+ S-A1/A2（跨变量消融）+ S-B1（RoPE 消融）
         产出：每个模型的组件价值图

Step 2 → 选择一个模型作为底座（通常选 v1.0 基准中综合最强的那个）
         做组件级替换（层次一），快速验证融合是否有增益

Step 3 → 如果层次一有效，进一步尝试层级嫁接（层次二）
         训练维度适配器 + 微调整体模型

Step 4 → 如果有精力且前两步效果有限，考虑知识蒸馏（层次三）
         设计 Student 架构 → 蒸馏训练 → 评估
```

---

## 4. 训练与微调方案

### 4.1 什么时候需要训练

| 场景 | 是否需要训练 | 训练范围 |
|---|---|---|
| 零样本结构消融（S-A1, S-B1, S-E1, S-G1 等） | 否 | — |
| 输出头替换（S-D1/D2） | 是 | 只训练新输出头，冻结 backbone |
| Patch 大小改变（S-C1） | 是 | 训练 Patch 层 + 微调 backbone |
| FFN 替换（S-F1/F2） | 是 | 训练新 FFN 层，冻结其余 |
| 融合模型（层次一） | 是 | 端到端微调或分阶段训练 |
| 融合模型（层次二 嫁接） | 是 | 训练适配层 + 端到端微调 |
| 融合模型（层次三 蒸馏） | 是 | 从零训练 Student |

### 4.2 微调策略

对于需要训练的实验，采用以下分层策略：

**Head-only Fine-tuning**（输出头替换实验）：

```python
# 冻结 backbone 所有参数
for param in model.backbone.parameters():
    param.requires_grad = False

# 只训练新输出头
new_head = QuantileRegressionHead(embed_dim=768, n_quantiles=9)
optimizer = Adam(new_head.parameters(), lr=1e-3)
```

训练数据：使用 v1.0 §2 中定义的 ERCOT 数据，按 60/20/20 分割训练/验证/测试集。注意测试集**必须与 v1.0 基准使用完全相同的起报点集合**，确保可比。

**Backbone 微调**（融合模型）：

```
阶段一（热身）：冻结 backbone，只训练新增模块（适配层/新输出头），lr=1e-3，约 500 步
阶段二（联合）：解冻 backbone，全参数微调，lr=1e-5（backbone）+ 1e-4（新模块），余弦退火
                 早停：验证集 loss 连续 5 个 epoch 不降则停止
```

**知识蒸馏训练**：

```
Teacher 推理：三个模型分别对训练集做完整推理，缓存所有输出（点预测 + 分位数）
Student 训练：
  Loss = 0.5 × PinballLoss(student_quantiles, ground_truth)
       + 0.3 × MSE(student_mean, teacher_ensemble_mean)
       + 0.2 × KL(student_distribution, teacher_distribution)
  lr = 3e-4, warmup = 1000 步, 余弦退火 → 0
  batch_size = 32, 最大 10000 步
```

### 4.3 数据切分与防泄露

**关键原则**：微调/训练用的数据**必须严格早于**评估用的起报点集合。推荐切分：

```
ERCOT 数据时间线：
|--- 2025-06-01 ~ 2025-07-31 ---|--- 2025-08-01 ~ 2025-08-14 ---|--- 2025-08-15 ~ 2025-09-30 ---|
         训练集                         验证集                          测试集
    （融合模型微调/蒸馏用）         （早停/超参选择用）            （最终评估，与 v1.0 相同）
```

Spike 阈值 P95 只能用训练集计算。验证集做超参搜索和早停。测试集只在最终报告时使用一次。

---

## 5. 评估与对比

### 5.1 评估协议

沿用 v1.0 的滚动回测协议（§4），确保所有对比公平：

- 同一组起报点（≥30 个）
- 同一指标体系（MAE / RMSE / MASE / Pinball / Spike-F1）
- 同一基准配置（ERCOT / volatility / 1h / 168 上下文 / 24 步长）

### 5.2 对比基准线

结构消融和融合模型的性能必须与以下基准线对比：

```
1. v1.0 参数级消融中的最佳单模型配置（来自消融 A-D 的最优组合）
2. 三个模型的完整版零样本性能（结构消融的直接参照）
3. v1.0 §7.2 中的简单融合策略（加权平均、Stacking）
4. 统计基线（Naive、ETS、Theta）作为下界
```

### 5.3 结果呈现

#### 结构消融结果

每个 S-* 消融产出一张表和一张图：

**表（示例：S-G1 逐层消融 for Toto）**：

| 跳过层 | MAE | ΔMAE | RMSE | ΔRMSE | Spike-F1 | ΔSpike-F1 | 重要性排名 |
|---|---|---|---|---|---|---|---|
| None (完整) | — | 0 | — | 0 | — | 0 | — |
| Layer 0 | — | +? | — | +? | — | -? | ? |
| Layer 1 | — | +? | — | +? | — | -? | ? |
| ... | | | | | | | |
| Layer 11 | — | +? | — | +? | — | -? | ? |

**图**：横轴为层编号，纵轴为 ΔMAE（或 ΔSpike-F1），bar chart，颜色按重要性从红到蓝。三个模型并排或叠加对比。

#### 融合模型结果

融合模型的最终对比表应包含：

| 模型 | 类型 | MAE ± std | RMSE ± std | Spike-F1 ± std | Pinball ± std |
|---|---|---|---|---|---|
| Toto-1.0 (完整) | 零样本 | — | — | — | — |
| Chronos-2 (完整) | 零样本 | — | — | — | — |
| TimesFM-2.5 (完整) | 零样本 | — | — | — | — |
| 加权平均融合 | v1.0 §7.2 | — | — | — | — |
| Stacking 融合 | v1.0 §7.2 | — | — | — | — |
| **组件融合模型** | v2.0 层次一 | — | — | — | — |
| **层嫁接融合模型** | v2.0 层次二 | — | — | — | — |
| **蒸馏融合模型** | v2.0 层次三 | — | — | — | — |

---

## 6. 执行路线与产出清单

| 阶段 | 内容 | 预估时间 | 关键产出 | 前置依赖 |
|---|---|---|---|---|
| 0 | v1.0 消融 A-D 基准结果 | （已在 v1.0 范围内） | 三模型完整版基准性能 | v1.0 框架搭建完成 |
| 1 | S-G1 逐层消融 | 2-3 天 | 三模型层重要性热力图 | 阶段 0 完成 |
| 2 | S-A1/A2 + S-B1 核心消融 | 1-2 天 | 跨变量注意力/RoPE 价值量化 | 阶段 0 完成（可与阶段 1 并行） |
| 3 | 分析消融结果，产出组件价值图 | 0.5 天 | 组件价值总表 + 融合方案设计 | 阶段 1-2 完成 |
| 4 | 组件级融合（层次一） | 3-5 天 | 融合模型 v1 + 微调 + 评估结果 | 阶段 3 完成 |
| 5 | 层嫁接融合（层次二，可选） | 3-5 天 | 融合模型 v2 + 评估对比 | 阶段 4 结果不满意时 |
| 6 | 知识蒸馏（层次三，可选） | 5-7 天 | Student 模型 + 蒸馏训练 + 评估 | 阶段 4-5 结果不满意时 |
| 7 | 论文写作与结果整理 | 按需 | 实验报告 / 论文初稿 | 阶段 4-6 择优完成 |

---

## 7. 工程实现指南

### 7.1 目录结构扩展

在 v1.0 的目录结构基础上，新增结构消融与融合相关目录：

```
school/
├── configs/experiments/
│   ├── structural/                    # 结构消融配置
│   │   ├── ablation_SG1_toto.yaml
│   │   ├── ablation_SG1_chronos2.yaml
│   │   ├── ablation_SG1_timesfm.yaml
│   │   ├── ablation_SA1_spacewise.yaml
│   │   ├── ablation_SB1_rope.yaml
│   │   └── ...
│   └── fusion/                        # 融合模型配置
│       ├── fusion_component_swap.yaml
│       ├── fusion_layer_graft.yaml
│       └── fusion_distill.yaml
├── src/
│   ├── models/
│   │   ├── ablation_ops.py            # apply_structural_ablation() 实现
│   │   ├── fusion_model.py            # FusionModel 定义
│   │   ├── fusion_trainer.py          # 微调/蒸馏训练循环
│   │   └── dimension_adapter.py       # 跨模型维度适配
│   └── analysis/
│       ├── layer_importance.py        # S-G1 结果分析与可视化
│       └── component_value_map.py     # 组件价值图生成
└── data/results/
    ├── structural_ablation/           # 结构消融结果
    │   ├── SG1_toto/
    │   ├── SG1_chronos2/
    │   └── ...
    └── fusion/                        # 融合模型结果
        ├── component_swap/
        ├── layer_graft/
        └── distill/
```

### 7.2 核心工具函数

**IdentityLayer**：用于跳过某层的占位模块

```python
class IdentityLayer(nn.Module):
    """替换被消融的 Transformer 层，直接传递输入"""
    def forward(self, layer_idx, inputs, attention_mask=None, kv_cache=None):
        return inputs
```

**IdentityAttention**：用于只移除注意力但保留 FFN 的消融

```python
class IdentityAttention(nn.Module):
    """替换被消融的注意力模块，返回零向量（不贡献残差）"""
    def forward(self, layer_idx, inputs, attention_mask=None, kv_cache=None):
        return torch.zeros_like(inputs)
```

**模型手术函数**：

```python
def skip_layer(model, model_name: str, layer_idx: int):
    """跳过指定层（用于 S-G1 逐层消融）"""
    if model_name == "toto":
        model.model.transformer.layers[layer_idx] = IdentityLayer()
    elif model_name == "chronos2":
        model.encoder.block[layer_idx] = IdentityEncoderBlock()
    elif model_name == "timesfm":
        model.stacked_transformer.layers[layer_idx] = IdentityLayer()
    return model

def remove_rope(model, model_name: str):
    """移除 RoPE 位置编码（用于 S-B1）"""
    if model_name == "toto":
        model.model.transformer.rotary_emb = NoOpRoPE()
    elif model_name == "chronos2":
        for block in model.encoder.block:
            block.layer[0].time_self_attn.rope = NoOpRoPE()
    elif model_name == "timesfm":
        for layer in model.stacked_transformer.layers:
            layer.attention.rope = NoOpRoPE()
    return model
```

### 7.3 Worker 适配

当前 v1.0 的 worker 模式（子进程调用各模型 venv）需要扩展以支持结构消融。建议在 worker 脚本中接收 `ablation_config` 参数：

```python
# workers/worker_toto.py 扩展
def run_tasks(tasks, ablation_config=None):
    model = load_toto_model()
    
    if ablation_config:
        model = apply_structural_ablation(model, ablation_config)
    
    results = []
    for task in tasks:
        pred = model.predict(task.context, task.horizon)
        results.append(pred)
    return results
```

---

## 8. 常见陷阱与注意事项

1. **零样本消融 vs 需训练消融**：改变模型的输入/输出形状（如 Patch 大小、输出头维度）需要训练；只"关闭"某组件（如跳过一层、关闭 RoPE）可以零样本测试。混淆两者会导致无效结论。

2. **数值稳定性**：移除 Norm 层或注意力缩放后，模型可能输出 NaN 或 Inf。实验代码需加入数值检查（`torch.isnan`, `torch.isinf`），并在结果中标记"该消融导致数值崩溃"——这本身就是一个结论。

3. **维度适配**：跨模型嫁接时，768/512/1280 维度不一致是最大工程挑战。确保适配层的梯度能正确回传，且初始化不破坏预训练特征。

4. **对比的公平性**：结构消融的性能必须与**同一模型的完整版**对比，而非与其他模型对比。"Toto 去掉 RoPE 后仍然比 Chronos-2 强"不是结论；"Toto 去掉 RoPE 后 MAE 上升 15%"才是。

5. **因果关系 vs 相关性**：结构消融只能说明"该组件被移除后性能下降"，不能直接说"该组件是性能好的原因"。可能存在冗余——A 和 B 单独移除都不降，但同时移除就降。如有精力可做二阶消融（组合移除）验证。

6. **训练数据不足**：三个预训练模型都是在海量数据上训练的（Toto 用了 1000 万条序列、TimesFM 用了 Google Trends + 公开数据集、Chronos-2 用了合成+真实混合数据）。仅用 ERCOT 几个月的电价数据微调，容易过拟合。使用早停、Dropout 增大、学习率衰减等正则化手段。

7. **计算预算**：S-G1 逐层消融对 TimesFM 需要跑 20 次（每次跳一层），每次都是完整的 30+ 起报点回测。提前估算 GPU 时间，考虑在 CPU/MPS 上是否可行（三个模型参数量在 40M-200M 级别，单次推理在 MacBook 上应可接受）。

---

## 附录 A：结构消融配置文件完整示例

```yaml
# configs/experiments/structural/ablation_SG1_toto_layer0.yaml
name: "SG1_toto_skip_layer_0"
description: "Toto 逐层消融：跳过第 0 层"

# 继承 v1.0 基准配置
market: ERCOT
nodes_group: volatility
freq: 1h
context_len: 168
horizon: 24
backtest:
  test_start: "2025-08-15"
  test_end: "2025-09-30"
  stride_hours: 24
models: [Toto]                     # 只对 Toto 做手术
spike:
  quantile: 0.95
  threshold_mode: global
  signal: [mean, q90]

# 结构消融配置
structural_ablation:
  type: layer_skip
  target_model: toto
  parameters:
    skip_layers: [0]               # 跳过第 0 层
  
  # 数值安全检查
  safety:
    check_nan: true                # 输出含 NaN 时标记并跳过该起报点
    check_inf: true
    max_abs_value: 1e6             # 超过此值视为数值爆炸
```

## 附录 B：融合模型配置文件示例

```yaml
# configs/experiments/fusion/fusion_component_swap_v1.yaml
name: "fusion_component_swap_v1"
description: "组件级融合：Toto backbone + 分位数输出头"

market: ERCOT
nodes_group: volatility
freq: 1h
context_len: 168
horizon: 24
backtest:
  test_start: "2025-08-15"
  test_end: "2025-09-30"
  stride_hours: 24

fusion:
  backbone: toto                   # 底座模型
  components:
    scaler: toto                   # CausalPatchStdMeanScaler
    patch_embed: toto              # Linear(64, 768)
    transformer_layers: toto       # 12 层 TransformerLayer
    output_head: chronos2          # 直接分位数回归（替换 MixtureStudentT）
  
  training:
    strategy: head_only            # 只训练新输出头
    train_data:
      start: "2025-06-01"
      end: "2025-07-31"
    val_data:
      start: "2025-08-01"
      end: "2025-08-14"
    optimizer: adam
    lr: 1e-3
    max_steps: 2000
    early_stopping:
      patience: 5
      monitor: val_pinball_loss
```

## 附录 C：参考文献与延伸阅读

以下论文和资源对理解本手册中的概念有帮助：

- **Toto 论文**：Datadog, "Toto: Time Series Optimized Transformer for Observability" (2025)
- **Chronos-2 论文**：Amazon, "Chronos: Learning the Language of Time Series" (2024) 及 v2 更新
- **TimesFM 论文**：Google, "A decoder-only foundation model for time-series forecasting" (2024)
- **RoPE**：Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **SwiGLU**：Shazeer, "GLU Variants Improve Transformer" (2020)
- **知识蒸馏**：Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **层级消融**：Ganesh et al., "Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning" (2021)