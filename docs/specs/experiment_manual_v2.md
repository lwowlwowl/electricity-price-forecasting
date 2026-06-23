# 电价预测：结构级消融与模型融合实验手册

> 版本：v2.1 ｜ 适用项目：`school/`（电价预测）
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

> **v2.1 变更**：Toto 2.0 替代 Toto 1.0 作为基础模型。Toto 2.0 修复了 1.0 的 Student-T 输出不稳定问题，架构更先进（GQA、xPos RoPE、MuP 缩放、SwiGLU），是更合适的实验对象。

### 1.1 Toto-2.0（Datadog）

**源码位置**：`external/toto/toto2/toto2/model.py`

Toto 2.0 的核心思想是用交替的"时间注意力+变量注意力"同时建模时序依赖和变量间相关性，并通过 MuP（μ-Parametrization）保证模型在不同宽度下超参数可迁移。其数据流：

```
原始序列 (batch × n_var × time)
  → PatchedCausalStdScaler（因果归一化，Welford 在线算法，Patch 级统计）
    → InputResidualMLP（残差 MLP 投影，2×patch_size → d_model）
      → VariateTimeTransformerDecoder（N 层，交替 Time/Variate 注意力）
        → QuantileKnotsOutputHead（分位数输出，通过 OutputResidualMLP）
```

**关键超参数**（来自 `configuration.py` 的 `Toto2ModelConfig`）：

| 参数 | 说明 |
|---|---|
| patch_size | Patch 大小（将时间步分组） |
| d_model | 隐藏维度 |
| num_heads | 注意力头数 |
| num_layers | Transformer 总层数 |
| layer_group_size | 每组层数（控制 Time/Variate 交替模式） |
| num_variate_layers_per_group | 每组中 Variate 层的数量 |
| variate_layer_first | Variate 层在组内是否排在前面 |
| num_groups（GQA） | K/V 的 head 组数（< num_heads 时启用 GQA） |
| qk_norm | 是否对 Q/K 做 RMSNorm |
| per_dim_scale | 是否使用 PerDimScale |
| use_xpos | 是否使用 xPos RoPE |
| residual_attn_ratio | τ-rule 残差缩放系数 |

**各组件详细说明**：

**1. PatchedCausalStdScaler**（`model.py`）：对输入序列做因果归一化——在每个 Patch 边界上用 Welford 在线算法计算到当前时刻为止的 running mean 和 std，再做 z-score。"因果"意味着不偷看未来数据。归一化后还会对数据做 `asinh()` 变换以压缩极端值。

**2. InputResidualMLP**（`model.py`）：将每个 Patch（patch_size 个时间步 + patch_size 个 mask 标志，共 2×patch_size 维）通过一个带残差跳连的 MLP 投影到 d_model 维嵌入空间。结构为 `Linear(2×patch_size → 4×d_model) → SiLU → Linear(4×d_model → d_model) + SkipProj(2×patch_size → d_model)`。

**3. VariateTimeTransformerDecoder**（`model.py`）：N 层 `SelfAttentionTransformerLayer`，按 `layer_group_size` 和 `num_variate_layers_per_group` 配置交替排列为"时间层"和"变量层"。时间层做因果注意力（沿时间轴，只能看过去），变量层做双向注意力（同一时刻的不同变量互相看）。

**4. SelfAttentionTransformerLayer**（`model.py`）：每层结构为：
```
输入 x
  → τ-split → RMSNorm → SelfAttention(GQA) → τ-merge 残差
  → τ-split → RMSNorm → SwiGLU FFN → τ-merge 残差
```
其中 τ-rule 根据层的深度自动调节残差贡献比例（越深的层贡献越小），避免深层网络训练不稳定。

**5. SelfAttention（GQA）**（`model.py`）：使用 Grouped Query Attention——Q 有 `num_heads` 个头，但 K/V 只有 `num_groups` 个头，每 `heads_per_group` 个 Q head 共享一组 K/V。投影方式为单个 fused `in_proj`（一次矩阵乘法同时产出 Q/K/V），输出通过 `out_proj`。注意力分数缩放为 `1/d_k`（MuP 规则，非标准的 `1/√d_k`）。

**6. QueryKeyProjection + xPos RoPE**（`model.py`）：仅在时间层中使用。对 Q 和 K 的前 50% 维度施加旋转位置编码。当 `use_xpos=True` 时，使用 `ExtrapolatableRotaryProjection`（标准 RoPE + 随距离衰减的缩放因子），帮助模型处理超长序列。**变量层不使用位置编码**（因为变量间没有顺序关系）。

**7. PerDimScale**（可选）：对 Q 的每个维度学一个缩放系数（初始化为 1），替代固定的全局缩放。

**8. QK-Norm**（可选）：对 Q 和 K 分别做 RMSNorm 后再计算注意力分数，防止数值爆炸。

**9. SwiGLU FFN**（`model.py`）：`Linear(d_model → 2×d_ff)` 输出一半做 SiLU 激活、一半做门控信号，两者逐元素相乘后过 `Linear(d_ff → d_model)`。比标准 ReLU MLP 表达能力更强。

**10. QuantileKnotsOutputHead**（`model.py`）：输出 9 个固定分位数（0.1, 0.2, ..., 0.9）的预测值。通过 `FusedPatchedParamProjection` 内部的 `OutputResidualMLP` 将 Transformer 输出映射到 `patch_size × 9` 维。

**代码中的精确属性路径**（消融操作需要用到）：

```python
# 从 worker 加载模型后的访问路径：
model = Toto2Model.from_pretrained(...)

model.scaler                              # PatchedCausalStdScaler
model.patch_proj                          # InputResidualMLP
model.transformer                         # VariateTimeTransformerDecoder
model.transformer.layers                  # DepthModuleList[SelfAttentionTransformerLayer]
model.transformer.layers[i]               # 第 i 层
model.transformer.layers[i].attn          # SelfAttention
model.transformer.layers[i].attn.in_proj  # fused QKV 投影 (Linear)
model.transformer.layers[i].attn.out_proj # 输出投影 (Linear)
model.transformer.layers[i].attn.qk_proj  # QueryKeyProjection (仅 time layer 有，variate layer 为 None)
model.transformer.layers[i].attn.q_norm   # RMSNorm (可选)
model.transformer.layers[i].attn.k_norm   # RMSNorm (可选)
model.transformer.layers[i].attn._pds     # PerDimScale (可选)
model.transformer.layers[i].ffn           # GatedLinearUnitFeedForwardNetwork
model.transformer.layers[i].norm1         # RMSNorm (attention 前)
model.transformer.layers[i].norm2         # RMSNorm (FFN 前)
model.transformer.out_norm                # 最终 RMSNorm
model.output_head                         # QuantileKnotsOutputHead
```

**判断某层是时间层还是变量层**：

```python
def is_variate_layer(model, layer_idx):
    """根据配置判断第 layer_idx 层是变量层还是时间层"""
    cfg = model.config
    if cfg.variate_layer_first:
        return layer_idx % cfg.layer_group_size < cfg.num_variate_layers_per_group
    return layer_idx % cfg.layer_group_size >= cfg.layer_group_size - cfg.num_variate_layers_per_group
```

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
   - **TimeSelfAttention**（`layers.py`）：沿时间轴的因果自注意力，使用 RoPE 位置编码。Q/K/V 为**分离投影**（三个独立的 Linear 层）。
   - **GroupSelfAttention**（`layers.py`）：沿 batch 维度做双向注意力——当多条序列标记为同一 `group_id` 时，它们的 patch 嵌入可以互相关注。这不同于 Toto 的 Variate 层（沿变量维度），而是沿样本维度做跨序列注意力。不使用 RoPE。
   - **FeedForward**（`layers.py`）：Pre-RMSNorm → MLP（d_model → d_ff → d_model），激活函数为 GELU。

**4. Encoder**（`model.py`）：6 个 EncoderBlock 堆叠 + 最终 RMSNorm + Dropout。

**5. OutputPatchEmbedding**：将编码器输出映射到 `output_patch_size × n_quantiles` 维，reshape 后直接输出各时刻的分位数预测。

**6. 分位数输出头**：Chronos-2 **不使用参数化概率分布**，而是直接回归预设的 9 个分位数（如 q0.1, q0.2, ..., q0.9）。训练时用 Pinball Loss。

**7. 协变量处理**：Chronos-2 原生支持未来协变量（`future_covariates`），通过额外的 patch 通道与目标序列拼接后一起输入 Encoder。

**代码中的精确属性路径**：

```python
# 从 worker 加载模型后的访问路径：
pipeline = Chronos2Pipeline.from_pretrained(...)
model = pipeline.model  # Chronos2Model

model.instance_norm                              # InstanceNorm
model.patch                                      # PatchEmbedding
model.input_patch_embedding                      # ResidualBlock
model.encoder                                    # Chronos2Encoder
model.encoder.block                              # ModuleList[Chronos2EncoderBlock]
model.encoder.block[i]                           # 第 i 个 block
model.encoder.block[i].time_self_attention       # TimeSelfAttention
model.encoder.block[i].time_self_attention.mha   # MHA (含 q_proj, k_proj, v_proj, out_proj, rope)
model.encoder.block[i].time_self_attention.mha.rope  # RoPE（每个 block 独立实例）
model.encoder.block[i].group_self_attention      # GroupSelfAttention
model.encoder.block[i].group_self_attention.mha  # MHA（含独立 rope）
model.encoder.block[i].group_self_attention.mha.rope  # RoPE（每个 block 第二个独立实例）
model.encoder.block[i].ffn                       # FeedForward
model.encoder.block[i].norm_time_attn            # Chronos2LayerNorm
model.encoder.block[i].norm_group_attn           # Chronos2LayerNorm
model.encoder.block[i].norm_ffn                  # Chronos2LayerNorm
model.encoder.final_layer_norm                   # 最终 Chronos2LayerNorm
model.output_patch_embedding                     # ResidualBlock
```

**⚠️ 注意：RoPE 不是全局共享的**。Chronos-2 每个 block 有**两个独立的 RoPE 实例**（TimeSelfAttention 内一个、GroupSelfAttention 内一个）。消融 RoPE 时必须逐 block 逐 MHA 替换。

### 1.3 TimesFM-2.5（Google，200M 参数）

**源码位置**：`external/timesfm/src/timesfm/`

TimesFM 是一个"纯 decoder"堆叠 Transformer，20 层深、1280 维宽，是三个模型中最大的：

```
原始序列
  → 输入归一化（可选）
    → Tokenizer（ResidualBlock：Linear → Swish → Linear + 残差）
      → 20 层 Stacked Transformer（每层：LayerNorm → MultiHeadAttention(RoPE) → LayerNorm → 残差 → LayerNorm → FFN → LayerNorm → 残差）
        → Output Projection Point（ResidualBlock → 点预测）
        → Output Projection Quantiles（ResidualBlock → 分位数 × output_patch）
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
| ff_activation | swish (SiLU) | FFN 激活函数 |
| qk_norm | rms | QK 归一化方式 |
| context_limit | 16384 | 最大上下文 token 数 |

**各组件详细说明**：

**1. Tokenizer**（ResidualBlock）：与 Chronos-2 的 InputPatchEmbedding 类似，是一个 `Linear(input_dims, hidden_dims) → Swish → Linear(hidden_dims, output_dims) + 残差跳连` 的结构，将每个 input patch（32 时间步）映射到 1280 维。

**2. Stacked Transformer**（`transformer.py`）：20 层完全相同的 Transformer 层，每层结构：

   - Pre-LayerNorm → MultiHeadAttention → Post-LayerNorm → 残差
   - Pre-LayerNorm → FFN → Post-LayerNorm → 残差

   TimesFM 的注意力有几个独特设计：
   - **PerDimScale**：在 Q 上对每个维度施加可学习的缩放因子（初始化为 1），替代标准的 `1/√d_k` 缩放。
   - **QK-Norm**：对 Q 和 K 分别做 RMSNorm 再点积，稳定深层训练。
   - **RoPE**（无 xpos）：标准旋转位置编码，不带 Toto 那样的 xpos 衰减。**每层有独立的 RoPE 实例。**
   - **Q/K/V 分离投影**：Q/K/V 各自有独立的 Linear 层（非 fused）。

**3. FFN**：`Linear(1280, 1280) → Swish → Linear(1280, 1280)`，隐藏维度等于模型维度（不像 Toto 用 4× 扩展的 SwiGLU）。

**4. 输出投影**：
   - **Point Head**：ResidualBlock(1280 → output_patch_len)，产生点预测。
   - **Quantile Head**：ResidualBlock(1280 → output_patch_len × num_quantiles)，得到多步分位数预测。

**5. 单变量为主**：TimesFM 逐条序列预测，没有 Toto 的 Variate 层或 Chronos-2 的 GroupSelfAttention 那种跨变量/跨序列机制。协变量通过独立通道处理。

**代码中的精确属性路径**：

```python
# 从 worker 加载模型后的访问路径：
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
module = tfm.model  # TimesFM_2p5_200M_torch_module

module.tokenizer                           # ResidualBlock (输入 Patch 嵌入)
module.stacked_xf                          # ModuleList[Transformer × 20]
module.stacked_xf[i]                       # 第 i 层 Transformer
module.stacked_xf[i].attn                  # MultiHeadAttention
module.stacked_xf[i].attn.q_proj           # Linear (Q 投影)
module.stacked_xf[i].attn.k_proj           # Linear (K 投影)
module.stacked_xf[i].attn.v_proj           # Linear (V 投影)
module.stacked_xf[i].attn.o_proj           # Linear (输出投影)
module.stacked_xf[i].attn.per_dim_scale    # PerDimScale
module.stacked_xf[i].attn.qk_norm          # RMSNorm (QK-Norm)
module.stacked_xf[i].attn.rotary_pos_emb   # RotaryPositionalEmbedding（每层独立）
module.stacked_xf[i].pre_attn_ln           # LayerNorm
module.stacked_xf[i].post_attn_ln          # LayerNorm
module.stacked_xf[i].pre_ff_ln             # LayerNorm
module.stacked_xf[i].ff0                   # Linear (FFN 第一层)
module.stacked_xf[i].activation            # SiLU
module.stacked_xf[i].ff1                   # Linear (FFN 第二层)
module.stacked_xf[i].post_ff_ln            # LayerNorm
module.output_projection_point             # Linear (点预测)
module.output_projection_quantiles         # Linear (分位数预测)
```

### 1.4 三模型架构对比总表

| 组件 | Toto-2.0 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|
| **总体规模** | N 层 | ~40M, 6 层 | ~200M, 20 层 |
| **归一化** | PatchedCausalStdScaler（Welford）+ asinh | InstanceNorm | 可选输入归一化 |
| **Patch 嵌入** | InputResidualMLP(2×patch → d_model) | ResidualBlock, 可重叠 | ResidualBlock(32→1280) |
| **时间注意力** | GQA (因果, xPos RoPE, MuP 1/d_k 缩放) | MHA (因果, 标准RoPE) | MHA (因果, RoPE, PerDimScale, QK-Norm) |
| **跨变量注意力** | Variate 层 (双向, 无RoPE, group_ids) | GroupSelfAttention (沿batch, 无RoPE) | **无** |
| **FFN** | SwiGLU (d_model → 2×d_ff, 门控) | GELU MLP (512→2048) | Swish MLP (1280→1280) |
| **层间 Norm** | RMSNorm (pre-norm) | Chronos2LayerNorm (pre-norm) | LayerNorm (pre+post) + QK-Norm |
| **位置编码** | xPos RoPE（仅 time layer，50% 维度） | 标准 RoPE（每 block 独立） | 标准 RoPE（每层独立） |
| **QKV 投影** | Fused in_proj（GQA） | 分离 q/k/v_proj | 分离 q/k/v_proj |
| **残差策略** | τ-rule 深度自适应缩放 | 标准加法残差 | 标准加法残差 |
| **输出头** | 分位数 (9个, 通过 OutputResidualMLP) | 直接分位数 (9个) | Point + 分位数 (ResidualBlock) |
| **多变量** | ✓ 原生 (Variate 层) | ✓ 原生 (Group) | ✗ 逐序列 |
| **协变量** | 通过额外序列拼接 | ✓ 原生 | ✓ (通过 covariate 接口) |

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

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-A1 | 冻结注意力权重（所有层 attn.requires_grad=False，forward 正常执行） | 三个模型 | 否（零样本） |
| S-A2 | 跳过注意力输出（将 attn 替换为 IdentityLayer，直接传递残差） | 三个模型 | 否（零样本） |
| S-A3 | 将注意力头数减半（仅保留前 50% head，输出投影对应截断） | 三个模型 | 否（零样本） |

**S-A1 实现细节**（以 Toto-2.0 为例）：

```python
# 冻结所有层的注意力参数
for layer in model.transformer.layers:
    for param in layer.attn.parameters():
        param.requires_grad = False
# 注意：forward 照常执行，只是反向传播时不更新这些参数
# 在零样本推理中 requires_grad=False 不影响前向传播结果
# 这个实验在零样本场景下等价于基准——它的意义体现在后续微调阶段
```

**S-A2 实现细节**——IdentityLayer 需要匹配各模型注意力层的调用签名：

```python
# === Toto-2.0 的 IdentityLayer ===
# SelfAttention.forward(x, seq_ids=None, **kwargs) → Tensor
class IdentityLayerToto2(nn.Module):
    def forward(self, x, seq_ids=None, **kwargs):
        return x

# 替换时间层和变量层的 attn
for layer in model.transformer.layers:
    layer.attn = IdentityLayerToto2()

# === Chronos-2 的 IdentityLayer ===
# TimeSelfAttention / GroupSelfAttention 被 EncoderBlock 内部调用
# 其 MHA.forward(x, mask, ...) → (output, position_bias)
# EncoderBlock 通过 .mha 子模块调用，但外层 TimeSelfAttention
# 的 forward 签名是 (hidden_states, mask, ...)
# 最简方案：替换整个 time_self_attention / group_self_attention
class IdentityLayerChronos2(nn.Module):
    """替代 TimeSelfAttention 或 GroupSelfAttention"""
    def forward(self, hidden_states, mask=None, position_bias=None, **kwargs):
        # 返回格式需要匹配 Chronos2EncoderBlockOutput 中的使用方式
        return hidden_states

for block in model.encoder.block:
    block.time_self_attention = IdentityLayerChronos2()
    # 如果要同时消融 Group 注意力：
    # block.group_self_attention = IdentityLayerChronos2()

# === TimesFM-2.5 的 IdentityLayer ===
# MultiHeadAttention.forward(input_embeddings, atten_mask) → tensor
class IdentityLayerTimesFM(nn.Module):
    def forward(self, input_embeddings, atten_mask=None, **kwargs):
        return input_embeddings

for layer in module.stacked_xf:
    layer.attn = IdentityLayerTimesFM()
```

**S-A3 实现细节**——注意力头减半：

```python
# === Toto-2.0 (Fused GQA in_proj) ===
# in_proj 的输出维度 = head_dim * (num_heads + 2 * num_groups)
# 减半 Q heads: 保留前 num_heads//2 个 Q head
# 同时需要修改 out_proj 的输入维度
import torch.nn as nn

def halve_heads_toto2(model):
    cfg = model.config
    half_heads = cfg.num_heads // 2
    head_dim = cfg.d_model // cfg.num_heads
    # GQA: num_groups 个 KV head，保持不变（它们已经是共享的）
    # 只截断 Q 部分
    for layer in model.transformer.layers:
        attn = layer.attn
        old_in_proj = attn.in_proj.weight  # shape: [out_features, d_model]
        # in_proj 输出布局: [Q heads | K groups | V groups]
        q_size = cfg.num_heads * head_dim
        kv_size = cfg.num_groups * head_dim  # K 和 V 各占这么多
        new_q_size = half_heads * head_dim
        # 截取前 half_heads 的 Q 权重 + 保持 KV 不变
        new_weight = torch.cat([
            old_in_proj[:new_q_size],           # Q: 取前半
            old_in_proj[q_size:q_size+kv_size], # K: 全保留
            old_in_proj[q_size+kv_size:]        # V: 全保留
        ], dim=0)
        attn.in_proj = nn.Linear(cfg.d_model, new_q_size + 2*kv_size, bias=False)
        attn.in_proj.weight = nn.Parameter(new_weight)
        # out_proj: 输入维度 = half_heads * head_dim
        old_out = attn.out_proj.weight  # [d_model, num_heads * head_dim]
        attn.out_proj = nn.Linear(new_q_size, cfg.d_model, bias=False)
        attn.out_proj.weight = nn.Parameter(old_out[:, :new_q_size])
        # 更新内部配置
        attn._num_heads = half_heads

# === Chronos-2 / TimesFM-2.5 (分离 Q/K/V 投影) ===
def halve_heads_separated(attn_module, num_heads, head_dim):
    """适用于有独立 q_proj, k_proj, v_proj 的模型"""
    half_heads = num_heads // 2
    half_dim = half_heads * head_dim
    # Q 投影：截取前 half_heads 个 head 的权重
    old_q = attn_module.q_proj.weight  # [num_heads*head_dim, d_model]
    attn_module.q_proj = nn.Linear(old_q.shape[1], half_dim, bias=False)
    attn_module.q_proj.weight = nn.Parameter(old_q[:half_dim])
    # K 投影：同步截取
    old_k = attn_module.k_proj.weight
    attn_module.k_proj = nn.Linear(old_k.shape[1], half_dim, bias=False)
    attn_module.k_proj.weight = nn.Parameter(old_k[:half_dim])
    # V 投影：同步截取
    old_v = attn_module.v_proj.weight
    attn_module.v_proj = nn.Linear(old_v.shape[1], half_dim, bias=False)
    attn_module.v_proj.weight = nn.Parameter(old_v[:half_dim])
    # 输出投影：输入维度变为 half_dim
    old_o = attn_module.o_proj.weight  # 或 out_proj
    attn_module.o_proj = nn.Linear(half_dim, old_o.shape[0], bias=False)
    attn_module.o_proj.weight = nn.Parameter(old_o[:, :half_dim])
```

#### S-B：跨变量/跨序列注意力消融

**目标**：电价预测中，多节点/多变量之间的信息共享有多重要？

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-B1 | 消融 Variate 层（Toto-2.0：跳过 variate layer；Chronos-2：跳过 GroupSelfAttention） | Toto-2.0, Chronos-2 | 否（零样本） |
| S-B2 | 消融 Time 层，只保留 Variate/Group（互补实验） | Toto-2.0, Chronos-2 | 否（零样本） |
| S-B3 | 替换 Group → Time 注意力（在 Chronos-2 中，把 GroupSelfAttention 替换成额外的 TimeSelfAttention） | Chronos-2 | **需要训练** |

**S-B1 实现细节**：

```python
# === Toto-2.0：跳过 variate layer ===
# 方法1：替换 variate layer 的 attn 为 Identity
for idx, layer in enumerate(model.transformer.layers):
    if is_variate_layer(model, idx):
        layer.attn = IdentityLayerToto2()

# === Chronos-2：跳过 GroupSelfAttention ===
for block in model.encoder.block:
    block.group_self_attention = IdentityLayerChronos2()
```

**S-B2 实现细节**：

```python
# === Toto-2.0：跳过 time layer（只保留 variate layer） ===
for idx, layer in enumerate(model.transformer.layers):
    if not is_variate_layer(model, idx):
        layer.attn = IdentityLayerToto2()

# === Chronos-2：跳过 TimeSelfAttention ===
for block in model.encoder.block:
    block.time_self_attention = IdentityLayerChronos2()
```

**⚠️ S-B3 需要训练**：将 GroupSelfAttention 替换成一个新的 TimeSelfAttention 后，新注入的权重是随机初始化的（与其它层的预训练权重不匹配），直接推理会产生垃圾输出。因此 S-B3 **不是零样本实验**，需要在电价数据上微调一段时间才能得出有意义的结论。建议先完成所有零样本消融后再执行 S-B3。

**⚠️ 单变量场景下的局限**：如果实验配置为"单节点单变量"，S-B1/B2 的消融将**无法体现效果**——Toto 的 Variate 层在单变量时不做任何跨变量计算，Chronos-2 的 GroupSelfAttention 在单序列时退化为自注意力（只有自己一条序列在组内）。因此 S-B 系列实验应在**多变量配置**（多节点联合预测，或加入协变量）下执行才有意义。

#### S-G1：位置编码消融（RoPE）

**目标**：电价的时间模式（日周期、周周期）对位置编码的依赖程度如何？

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G1a | 移除 RoPE（将 RoPE 替换为恒等操作） | 三个模型 | 否（零样本） |
| S-G1b | 仅 Toto-2.0：关闭 xPos（保留标准 RoPE，只禁用 xPos 衰减因子） | Toto-2.0 | 否（零样本） |

**S-G1a 实现细节**：

```python
# === Toto-2.0 ===
# RoPE 在 QueryKeyProjection 中，仅 time layer 有 qk_proj
# 将 qk_proj 设为 None 可跳过 RoPE（代码中有 if qk_proj is not None 的判断）
for idx, layer in enumerate(model.transformer.layers):
    if not is_variate_layer(model, idx):  # 只有 time layer 有 RoPE
        layer.attn.qk_proj = None

# === Chronos-2 ===
# 每个 block 的 TimeSelfAttention.mha.rope 和 GroupSelfAttention.mha.rope
# 需要替换为不做任何变换的 Identity
class IdentityRoPE(nn.Module):
    def forward(self, q, k, **kwargs):
        return q, k

for block in model.encoder.block:
    block.time_self_attention.mha.rope = IdentityRoPE()
    block.group_self_attention.mha.rope = IdentityRoPE()

# === TimesFM-2.5 ===
# 每层独立的 rotary_pos_emb
class IdentityRoPE(nn.Module):
    def forward(self, x, position=None, **kwargs):
        return x

for layer in module.stacked_xf:
    layer.attn.rotary_pos_emb = IdentityRoPE()
```

**⚠️ RoPE 不是全局共享的**：Chronos-2 和 TimesFM 中每层/每 block 都有独立的 RoPE 实例，消融时需要逐层替换，不能只替换"全局一份"。

**S-G1b 实现细节**（仅 Toto-2.0）：

```python
# xPos 通过 ExtrapolatableRotaryProjection 实现
# 将其替换为标准 RotaryProjection（无衰减因子）
# 需要查看源码确认具体类名和初始化参数
# 简化方案：将 xPos 的 scale 因子设为 1（取消衰减）
for idx, layer in enumerate(model.transformer.layers):
    if not is_variate_layer(model, idx) and layer.attn.qk_proj is not None:
        # 如果 xPos 是通过 scale 属性控制的：
        if hasattr(layer.attn.qk_proj, 'scale'):
            layer.attn.qk_proj.scale = None  # 禁用衰减，保留旋转
```

#### S-G2：Patch 嵌入消融

**目标**：Patch 嵌入（将原始时间步映射到模型维度的第一个投影层）对预测质量的贡献。

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G2 | 替换 Patch 嵌入为简单线性投影（去除 ResidualBlock / MLP 结构，仅保留一层 Linear） | 三个模型 | 否（零样本） |

**实现细节**：

```python
import torch.nn as nn

# === Toto-2.0 ===
# InputResidualMLP: 2*patch_size → d_model (带残差 MLP)
# 替换为单层线性
patch_size = model.config.patch_size
d_model = model.config.d_model
model.patch_proj = nn.Linear(2 * patch_size, d_model)
# 用原 MLP 的近似初始化（或随机初始化看性能掉多少）

# === Chronos-2 ===
# input_patch_embedding: ResidualBlock
d_in = model.input_patch_embedding.in_features  # 需确认
d_out = model.input_patch_embedding.out_features
model.input_patch_embedding = nn.Linear(d_in, d_out)

# === TimesFM-2.5 ===
# tokenizer: ResidualBlock(input_dims → model_dims)
in_dim = 32  # patch_size
out_dim = 1280  # model_dims
module.tokenizer = nn.Linear(in_dim, out_dim)
```

**注意**：由于替换后是随机初始化的 Linear 权重（与原 ResidualBlock 的预训练权重不匹配），性能必然大幅下降。这个实验的意义在于量化"精心设计的 Patch 嵌入比简单线性投影好多少"。如果下降不大，说明这个组件不是关键。

#### S-G3：输出头消融

**目标**：输出头的复杂度对预测质量（尤其是不确定性估计）的贡献。

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G3a | 将分位数输出头替换为直接线性映射（去除 ResidualBlock/MLP，只保留一层 Linear） | 三个模型 | 否（零样本） |
| S-G3b | TimesFM 特有：只保留点预测头，去掉分位数头（观察点预测本身是否足够） | TimesFM | 否（零样本） |

**实现细节**（S-G3a）：

```python
# === Toto-2.0 ===
# output_head: QuantileKnotsOutputHead 内含 OutputResidualMLP
# 替换为直接的 Linear(d_model → patch_size * n_quantiles)
n_quantiles = 9
patch_size = model.config.patch_size
d_model = model.config.d_model
model.output_head = nn.Linear(d_model, patch_size * n_quantiles)

# === Chronos-2 ===
# output_patch_embedding: ResidualBlock
# 类似处理
d_model = 512
output_dim = model.output_patch_embedding.out_features
model.output_patch_embedding = nn.Linear(d_model, output_dim)

# === TimesFM-2.5 (S-G3b) ===
# 去掉 quantile head，只用 point head
# 推理时只调用 output_projection_point，忽略 output_projection_quantiles
# 这不需要修改模型结构，只需修改推理代码中的输出解析逻辑
```

#### S-G4：FFN 消融

**目标**：FFN（前馈网络）对注意力输出的后处理有多重要？

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G4 | 跳过所有层的 FFN（将 FFN 替换为 Identity） | 三个模型 | 否（零样本） |

**实现细节**：

```python
# === Toto-2.0 ===
class IdentityFFN(nn.Module):
    def forward(self, x):
        return x

for layer in model.transformer.layers:
    layer.ffn = IdentityFFN()

# === Chronos-2 ===
for block in model.encoder.block:
    block.ffn = IdentityFFN()

# === TimesFM-2.5 ===
# FFN 由 ff0 + activation + ff1 组成，将 ff0/ff1 设为 Identity
for layer in module.stacked_xf:
    layer.ff0 = nn.Identity()
    layer.ff1 = nn.Identity()
    # 注意：这样 activation 作用于 Identity 输出，等价于跳过整个 FFN
```

#### S-G5：层归一化消融

**目标**：LayerNorm/RMSNorm 在深层模型中的训练稳定性作用，在推理时是否可以移除？

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G5 | 将所有层归一化替换为 Identity | 三个模型 | 否（零样本） |

**注意**：此实验可能导致数值溢出（尤其是 20 层的 TimesFM），如果推理直接 NaN 了，记录"移除归一化后模型崩溃"本身就是结论——说明归一化对该模型是生死攸关的。

#### S-G6：深度消融

**目标**：模型需要多少层才能做好电价预测？（尤其是 TimesFM 的 20 层可能冗余）

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-G6a | 只保留前 50% 的层（截断后半部分） | 三个模型 | 否（零样本） |
| S-G6b | 只保留前 25% 的层 | TimesFM | 否（零样本） |
| S-G6c | 只保留后 50% 的层（截断前半部分） | 三个模型 | 否（零样本） |

**实现细节**：

```python
# === Toto-2.0 ===
total = len(model.transformer.layers)
half = total // 2
model.transformer.layers = model.transformer.layers[:half]  # 保留前半

# === Chronos-2 ===
model.encoder.block = model.encoder.block[:3]  # 6 层取前 3 层

# === TimesFM-2.5 ===
module.stacked_xf = module.stacked_xf[:10]  # 20 层取前 10 层
# S-G6b: module.stacked_xf = module.stacked_xf[:5]  # 取前 5 层
```

**注意**：截断层后输出的维度不变（最后一层的输出维度 = d_model），所以输出头仍然兼容。

#### S-L：逐层消融（Per-Layer Ablation）

**目标**：精确定位每一层在推理中的贡献——逐一跳过单独某层，观察性能变化，绘制"逐层贡献曲线"。

**为什么需要逐层消融**：S-G6 系列只能告诉我们"前半 vs 后半"的粗粒度信息，但无法回答"第 3 层和第 4 层谁更重要？""是否存在某些'冗余层'可以安全移除？"逐层消融通过每次只跳过一层来精确定位每层的边际贡献。

| 编号 | 操作 | 适用模型 | 是否需要训练 |
|---|---|---|---|
| S-L0 ~ S-L5 | 逐层跳过（Toto-2.0, 6 层） | Toto-2.0 | 否（零样本） |
| S-L0 ~ S-L5 | 逐层跳过（Chronos-2, 6 层） | Chronos-2 | 否（零样本） |
| S-L0 ~ S-L19 | 逐层跳过（TimesFM-2.5, 20 层） | TimesFM-2.5 | 否（零样本） |

**各模型层数**：

- **Toto-2.0**：6 层（layer_group_size=6，其中 5 层为 time layer，第 5 层为 variate layer）
- **Chronos-2**：6 层（每层含 TimeSelfAttention + GroupSelfAttention + FFN）
- **TimesFM-2.5**：20 层（每层含 MultiHeadAttention + FFN）

**实现原理**：

```python
# 逐层消融 = 将整个第 N 层替换为 IdentityBlock（跳过该层所有子组件）
# 每次只跳过一层，其余层正常执行

# === Toto-2.0 ===
class IdentityBlockToto2(nn.Module):
    """forward(x, seq_ids=None, **kwargs) → x"""
    def forward(self, x, seq_ids=None, **kwargs):
        return x

layers = list(model.transformer.layers)
layers[N] = IdentityBlockToto2()  # 跳过第 N 层
model.transformer.layers = nn.ModuleList(layers)

# === Chronos-2 ===
class IdentityBlockChronos2(nn.Module):
    """forward(hidden_states, *, position_ids, attention_mask, ...) → output[0]=hidden_states"""
    def forward(self, hidden_states, *, position_ids=None, attention_mask=None,
                group_time_mask=None, output_attentions=False, **kwargs):
        return IdentityOutput(hidden_states)  # 支持 [0] 索引

blocks = list(model.encoder.block)
blocks[N] = IdentityBlockChronos2()
model.encoder.block = nn.ModuleList(blocks)

# === TimesFM-2.5 ===
class IdentityBlockTimesFM(nn.Module):
    """forward(x, atten_mask, decode_cache) → (x, None)"""
    def forward(self, x, atten_mask=None, decode_cache=None, **kwargs):
        return x, None  # TimesFM 层返回 (output, cache) 元组

layers = list(module.stacked_xf)
layers[N] = IdentityBlockTimesFM()
module.stacked_xf = nn.ModuleList(layers)
```

**配置方式**：使用 `skip_layer_N` 格式的 ablation_type 字符串，N 为层索引（从 0 开始）：

```yaml
structural_ablation:
  models: [toto2]
  ablations:
    - skip_layer_0
    - skip_layer_1
    - skip_layer_2
    - skip_layer_3
    - skip_layer_4
    - skip_layer_5
```

**预期产出**：

1. **逐层贡献曲线**：x 轴为层索引，y 轴为 SMAPE 退化百分比。曲线形状揭示：
   - 如果是递增的（越深越重要）→ 后层是关键
   - 如果是 V 形或 U 形 → 首尾关键，中间冗余
   - 如果某层跳过后 SMAPE 接近 0% 退化 → 该层冗余，融合时可省略
2. **Toto-2.0 time vs variate 层对比**：第 5 层（variate）与其他 time 层的对比
3. **TimesFM 深层冗余性分析**：20 层中哪些可以安全移除

**总实验数**：6（Toto）+ 6（Chronos）+ 20（TimesFM）= 32 次零样本推理

### 2.4 消融实验总结表

| 实验 | 操作 | Toto-2.0 | Chronos-2 | TimesFM-2.5 | 训练要求 |
|---|---|---|---|---|---|
| S-A1 | 冻结 attn 权重 | ✓ | ✓ | ✓ | 零样本 |
| S-A2 | 跳过 attn | ✓ | ✓ | ✓ | 零样本 |
| S-A3 | Head 减半 | ✓(fused) | ✓(分离) | ✓(分离) | 零样本 |
| S-B1 | 跳过 Variate/Group | ✓ | ✓ | N/A | 零样本 |
| S-B2 | 跳过 Time | ✓ | ✓ | N/A | 零样本 |
| S-B3 | Group→Time 替换 | N/A | ✓ | N/A | **需训练** |
| S-G1a | 移除 RoPE | ✓ | ✓ | ✓ | 零样本 |
| S-G1b | 关闭 xPos | ✓ | N/A | N/A | 零样本 |
| S-G2 | 简化 Patch 嵌入 | ✓ | ✓ | ✓ | 零样本 |
| S-G3a | 简化输出头 | ✓ | ✓ | ✓ | 零样本 |
| S-G3b | 去掉分位数头 | N/A | N/A | ✓ | 零样本 |
| S-G4 | 跳过 FFN | ✓ | ✓ | ✓ | 零样本 |
| S-G5 | 跳过 LayerNorm | ✓ | ✓ | ✓ | 零样本 |
| S-G6a | 保留前 50% 层 | ✓ | ✓ | ✓ | 零样本 |
| S-G6b | 保留前 25% 层 | N/A | N/A | ✓ | 零样本 |
| S-G6c | 保留后 50% 层 | ✓ | ✓ | ✓ | 零样本 |
| S-L | 逐层跳过第 N 层 | ✓(×6) | ✓(×6) | ✓(×20) | 零样本 |

### 2.5 消融的执行顺序建议

```
第一批（快速验证，只改推理逻辑，不改权重）：
  S-A2, S-B1, S-G1a, S-G4, S-G5, S-G6a
  → 最直观地看"去掉 X 后崩溃多少"

第二批（需要小心处理权重的截断/替换）：
  S-A3, S-G2, S-G3a, S-G6b, S-G6c

第三批（逐层消融，精细定位每层贡献）：
  S-L: skip_layer_0 ~ skip_layer_N（三个模型各跑一遍）
  → 产出逐层贡献曲线，指导融合时的层选择

第四批（需要微调）：
  S-B3
  → 放在所有零样本消融做完后，积累了足够认知再做

辅助实验（与微调阶段结合）：
  S-A1（冻结 attn 的意义要在训练中才体现）
```

---

## 3. 实现框架

### 3.1 消融器基类设计

```python
from abc import ABC, abstractmethod
import torch.nn as nn

class StructuralAblation(ABC):
    """结构级消融的基类"""

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """对模型应用消融操作，返回修改后的模型（in-place 修改）"""
        ...

    @abstractmethod
    def describe(self) -> str:
        """返回此消融操作的人类可读描述"""
        ...

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """此消融是否需要重新训练才能得出有意义的结论"""
        ...
```

### 3.2 配置驱动

结构级消融的实验应集成到现有的 YAML 配置驱动框架中：

```yaml
# configs/structural_ablation/toto2_skip_attn.yaml
model:
  name: toto2
  checkpoint: external/toto/toto2/checkpoints/...

ablation:
  type: skip_attention       # 对应 S-A2
  target_layers: all         # 可选：all / time_only / variate_only / [0,2,4]
  identity_class: IdentityLayerToto2

evaluation:
  market: ERCOT
  nodes: [node1, node2, node3]  # 高波动节点
  context_length: 168
  prediction_length: 24
  rolling_starts: 30
  metrics: [mae, rmse, mase, spike_f1, pinball]
```

### 3.3 与现有代码的集成点

结构级消融应作为模型加载后、推理前的一个"钩子"注入：

```python
# 伪代码 - 在 worker 中的集成点
def run_structural_ablation_experiment(config):
    # 1. 加载基础模型（完整预训练权重）
    model = load_model(config.model)

    # 2. 应用消融操作
    ablation = create_ablation(config.ablation)
    model = ablation.apply(model)

    # 3. 运行标准评估流程（与 v1.0 共享）
    results = evaluate(model, config.evaluation)

    # 4. 与基准对比
    baseline = load_baseline_results(config.model.name)
    delta = compute_delta(results, baseline)

    return results, delta
```

---

## 4. 模型融合（Phase 2）

> 先完成第 2 章的结构级消融，得到"组件贡献热力图"后再执行本章。

### 4.1 融合的目标

结构级消融告诉我们每个模型内部哪些组件"最重要"。融合的目标是：

```
从三个模型中各取最强组件 → 拼装成一个新架构 → 在电价数据上训练
```

例如（假设性的消融结论）：

- 如果 Toto 的 Variate 层在多节点场景下贡献最大，而 TimesFM 的深层 FFN 对长程依赖最重要——那就用 Toto 的 Variate 结构 + TimesFM 的 FFN 设计。
- 如果 Chronos-2 的 GroupSelfAttention 比 Toto 的 Variate 层更适合电价的跨节点建模——就选择 Chronos 的跨变量机制。

### 4.2 融合路径（待消融结论确定后细化）

```
消融结论 → 确定各组件"冠军" → 设计融合架构 → 实现 → 训练 → 评估
```

融合架构的具体方案需要等消融结果出来后再确定。但可以预见的选择包括：

1. **位置编码**：RoPE vs xPos vs 无位置编码 → 消融 S-G1 的结果决定
2. **跨变量机制**：Variate 层 vs GroupSelfAttention vs 无 → S-B 系列决定
3. **FFN 类型**：SwiGLU vs 标准 MLP → S-G4 的差异对比
4. **深度配置**：几层够用？→ S-G6 决定
5. **输出头**：分位数直接回归 vs 更复杂的参数化分布 → S-G3 决定

### 4.3 训练计划

融合模型需要在电价数据上从头训练或微调：

```
数据: ERCOT 全历史（2020-2024）+ 外部协变量
训练: 先在大量普通节点上预训练 → 在高波动节点上微调
评估: 与三个基础模型的 v1.0 基准对比
目标: 在所有指标上超过单一基础模型的最优结果
```

---

## 5. 注意事项与常见陷阱

### 5.1 零样本 vs 需要训练

**绝大多数消融是零样本的**（直接在预训练权重上操作，不训练）。这是刻意的设计——目的是诊断"当前权重中各组件的贡献"，而非"模型能否适应缺少某组件"。

唯一需要训练的实验是 **S-B3**（替换 Group→Time），因为新注入的随机权重无法直接使用。

### 5.2 注意力消融的 fused vs 分离问题

Toto-2.0 使用 **fused in_proj**（一次矩阵乘法同时产出 Q/K/V），而 Chronos-2 和 TimesFM 使用**分离的 q_proj / k_proj / v_proj**。

这意味着：

- 对 Toto-2.0 做 S-A3（Head 减半）时，需要理解 in_proj 的输出布局（Q 在前、KV 在后），精确截断。
- 对 Chronos-2 / TimesFM 做 S-A3 时，直接对各投影层独立截断即可，更简单。

### 5.3 单变量场景下 S-B 系列无效

**再次强调**：如果实验配置为单节点单变量：

- Toto 的 Variate 层只有一个变量 token，attention 矩阵是 1×1（恒等），消融它等于没消融。
- Chronos-2 的 GroupSelfAttention 在单序列时只有自己一个成员，同理。

因此 S-B 系列必须在多变量/多节点配置下执行。基准配置已设为 3 个高波动节点，满足此要求。

### 5.4 数值稳定性

某些消融（特别是 S-G5 去归一化）可能导致数值爆炸。如果推理产出 NaN 或 Inf：

1. 记录"此消融导致模型崩溃"——这本身是有价值的结论
2. 不需要调试使其"正常工作"——崩溃说明该组件对模型稳定性至关重要

### 5.5 消融粒度控制

某些消融可以做更精细的控制（不是"全部移除"而是"部分移除"）：

- S-A2 可以只对偶数层/奇数层跳过 attn
- S-B1 可以只消融一部分 variate 层
- S-G6 可以测试不同的截断比例（30%/50%/70%）

如果时间允许，精细控制能提供更 gradual 的洞察（"从什么层数开始，性能出现悬崖式下降？"）。

---

## 6. 预期产出

完成本手册的实验后，应产出：

1. **组件贡献热力图**：三个模型 × 各组件 的性能贡献矩阵
2. **架构洞察报告**：回答"电价预测最需要什么样的 Transformer 组件？"
3. **融合模型方案**：基于消融结论的最优组件组合设计
4. **训练好的融合模型**：在电价数据上达到 SOTA 的专用模型

---

## 附录 A：IdentityLayer 签名速查表

| 模型 | 被替换组件 | IdentityLayer 签名 |
|---|---|---|
| Toto-2.0 | SelfAttention | `forward(self, x, seq_ids=None, **kwargs) → Tensor` |
| Toto-2.0 | GatedLinearUnitFeedForwardNetwork | `forward(self, x) → Tensor` |
| Chronos-2 | TimeSelfAttention / GroupSelfAttention | `forward(self, hidden_states, mask=None, position_bias=None, **kwargs) → Tensor` |
| Chronos-2 | FeedForward | `forward(self, hidden_states) → Tensor` |
| TimesFM | MultiHeadAttention | `forward(self, input_embeddings, atten_mask=None, **kwargs) → Tensor` |
| TimesFM | FFN (ff0+ff1) | 用 `nn.Identity()` 替换 `ff0` 和 `ff1` |

---

## 附录 B：模型属性路径速查表

```
=== Toto-2.0 ===
model.scaler                              # 因果归一化
model.patch_proj                          # InputResidualMLP
model.transformer.layers[i].attn          # SelfAttention (GQA)
model.transformer.layers[i].attn.in_proj  # fused QKV
model.transformer.layers[i].attn.out_proj # 输出投影
model.transformer.layers[i].attn.qk_proj  # RoPE (time layer only, None for variate)
model.transformer.layers[i].ffn           # SwiGLU FFN
model.transformer.layers[i].norm1/norm2   # RMSNorm
model.transformer.out_norm                # 最终 norm
model.output_head                         # 分位数输出头

=== Chronos-2 ===
model.instance_norm                       # 实例归一化
model.input_patch_embedding               # ResidualBlock
model.encoder.block[i].time_self_attention.mha         # TimeMHA
model.encoder.block[i].time_self_attention.mha.rope    # RoPE (独立)
model.encoder.block[i].group_self_attention.mha        # GroupMHA
model.encoder.block[i].group_self_attention.mha.rope   # RoPE (独立)
model.encoder.block[i].ffn                # FeedForward
model.output_patch_embedding              # 输出映射

=== TimesFM-2.5 ===
module.tokenizer                          # ResidualBlock
module.stacked_xf[i].attn                 # MultiHeadAttention
module.stacked_xf[i].attn.q_proj/k_proj/v_proj  # 分离投影
module.stacked_xf[i].attn.rotary_pos_emb  # RoPE (每层独立)
module.stacked_xf[i].attn.per_dim_scale   # PerDimScale
module.stacked_xf[i].ff0/ff1             # FFN
module.output_projection_point            # 点预测头
module.output_projection_quantiles        # 分位数头
```