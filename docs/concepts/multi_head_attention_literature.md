# 多头注意力（Multi-Head Attention）相关文献综述

> 整理时间：2026年8月
> 用途：组会汇报——"别人如何用多头注意力架构，解决了什么问题"

---

## 一、奠基之作：原始 Transformer 与多头注意力

### 1. Attention Is All You Need (Vaswani et al., 2017)
- **论文链接**：[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **会议**：NeurIPS 2017
- **核心贡献**：提出 Transformer 架构，完全摒弃 RNN 和 CNN，仅依赖注意力机制。引入多头注意力（MHA），通过将 Q、K、V 投影到多个子空间并行计算注意力，使模型能从不同表示子空间、不同位置联合关注信息。
- **解决的问题**：传统 RNN 难以并行化、长距离依赖建模困难。多头机制让模型在不同子空间中捕获多样化的注意力模式。
- **汇报要点**：这是所有多头注意力工作的起点，必须讲清楚 MHA 的基本公式和"为什么需要多个头"。

---

## 二、多头注意力的分析与可解释性

这一方向研究"多个头到底在做什么、是否冗余、能否裁剪"。

### 2. Are Sixteen Heads Really One Head in MHA? (Michel et al., 2019)
- **论文链接**：[arxiv.org/abs/1905.10650](https://arxiv.org/abs/1905.10650)（NeurIPS 2019）
- **核心发现**：大量注意力头可以在测试时被剪掉而几乎不影响性能；许多头学习到的功能是冗余的。仅保留少量关键头即可维持模型质量。
- **解决的问题**：多头注意力是否真的需要这么多头？揭示了头的冗余性，为模型压缩提供依据。

### 3. Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting (Voita et al., 2019)
- **论文链接**：[arxiv.org/abs/1905.09418](https://arxiv.org/abs/1905.09418)（ACL 2019）
- **核心发现**：分析了 Transformer 编码器中各个注意力头的角色，发现最重要的头通常扮演一致且语言上可解释的功能（如关注相邻词、关注分隔符、关注句法依赖等）。大部分头可以剪枝，但少数关键头不可替代。
- **解决的问题**：理解多头注意力中每个头的语言学功能，为"哪些头重要"提供实证。

### 4. What Does BERT Look At? The Analysis of BERT's Attention (Clark et al., 2019)
- **论文链接**：[arxiv.org/abs/1906.04341](https://arxiv.org/abs/1906.04341)（ACL 2019 Workshop）
- **核心发现**：分析 BERT 中 144 个注意力头的行为，发现某些头直接编码了句法依赖关系（如 dobj、nsubj 等），提出了 attention probing 方法。
- **解决的问题**：注意力头是否真的学到了语言结构？答案是部分头确实学到了。

### 5. The Heads Hypothesis: A Unifying Statistical Approach Towards Understanding Multi-Headed Attention in BERT (Pande et al., 2022)
- **论文链接**：[arxiv.org/abs/2101.09115](https://arxiv.org/abs/2101.09115)（AAAI 2022）
- **核心贡献**：提出统一的统计框架来分类和理解 BERT 中多头注意力的行为模式。
- **解决的问题**：不同分析方法结论不一致，需要一个统一框架来系统性地分析注意力头。

---

## 三、多头注意力的多样性增强

这一方向关注"如何让多个头学到真正不同的东西，而非互相冗余"。

### 6. On the Diversity of Multi-Head Attention (Li et al., 2021)
- **论文链接**：[sciencedirect.com/science/article/pii/S0925231221005725](https://www.sciencedirect.com/science/article/pii/S0925231221005725)（Neurocomputing 2021）
- **核心贡献**：提出两种方法增强多头注意力的多样性——disagreement regularization（显式鼓励头间差异）和 head disentanglement（解耦不同头的表示）。
- **解决的问题**：标准 MHA 中多个头可能学到相似的注意力模式（冗余），降低了多头的有效性。

### 7. Diversifying Multi-Head Attention in the Transformer Model (MDPI, 2024)
- **论文链接**：[mdpi.com/2504-4990/6/4/126](https://www.mdpi.com/2504-4990/6/4/126)
- **核心贡献**：基于 Hebbian 学习的约束优化算法，强制 Transformer 不同头之间的多样化。
- **解决的问题**：头间多样性不足导致的表示冗余和信息浪费。

---

## 四、效率导向的多头注意力变体（KV Cache 优化）

这一方向是当前大模型最活跃的研究领域，核心目标是减少推理时 KV Cache 的内存开销。

### 8. Fast Transformer Decoding: One Write-Head is All You Need — Multi-Query Attention (Shazeer, 2019)
- **论文链接**：[arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
- **核心贡献**：提出 Multi-Query Attention（MQA），所有注意力头共享同一组 Key 和 Value，仅保留不同的 Query。极大减少 KV Cache 的大小。
- **解决的问题**：自回归解码时，每个头都需要加载独立的 K/V 张量，内存带宽成为瓶颈。MQA 共享 K/V 后大幅加速推理。
- **代价**：质量略有下降（因为所有头共享同一份 K/V 表示）。

### 9. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)
- **论文链接**：[arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)（EMNLP 2023）
- **核心贡献**：提出 Grouped-Query Attention（GQA），是 MHA 和 MQA 的通用化。将多个 Query 头分成 G 组，每组共享一份 K/V。组数 G=1 时退化为 MQA，组数 G=头数时退化为 MHA。还提出了从已有 MHA checkpoint 转换为 GQA 的 uptraining 方法。
- **解决的问题**：MQA 质量下降明显，MHA 推理慢。GQA 在质量和效率之间取得平衡，是 LLaMA-2 70B 等模型的标准配置。

### 10. DeepSeek-V2: Multi-Head Latent Attention (MLA) (DeepSeek, 2024)
- **论文链接**：[arxiv.org/abs/2405.02456](https://arxiv.org/abs/2405.02456)（DeepSeek-V2 技术报告）
- **核心贡献**：提出 Multi-Head Latent Attention（MLA），通过将 K/V 投影到低维 latent 向量来压缩 KV Cache（压缩约 57 倍），同时保持接近 MHA 的模型质量。推理时只需缓存 latent 向量，再通过上投影恢复各头的 K/V。
- **解决的问题**：GQA/MQA 通过"硬编码丢弃"来减少 K/V，损失了头间独立性；MLA 通过"压缩-恢复"机制在大幅减少缓存的同时保留了表达能力。是 DeepSeek-V2/V3/R1 的核心架构创新。
- **延伸阅读**：[Towards Economical Inference: Enabling DeepSeek's MLA (arxiv.org/abs/2502.14837)](https://arxiv.org/abs/2502.14837) — 关于 MLA 的高效推理实现。

> **MHA → MQA → GQA → MLA 演进路线**是开会讲解的重点，这条线清晰展示了"如何在保持质量的前提下逐步压缩 KV Cache"。

---

## 五、稀疏注意力变体（长序列效率）

这一方向解决标准 MHA 计算复杂度 O(N²) 的问题。

### 11. Longformer: The Long-Document Transformer (Beltagy et al., 2020)
- **论文链接**：[arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
- **核心贡献**：提出局部窗口注意力 + 少量全局注意力，将复杂度从 O(N²) 降至 O(N)。可处理数千 token 的长文档。
- **解决的问题**：标准 MHA 无法处理长文档（如全文级别的 NLP 任务）。

### 12. BigBird: Transformers for Longer Sequences (Zaheer et al., 2020)
- **论文链接**：[arxiv.org/abs/2007.14062](https://arxiv.org/abs/2007.14062)（NeurIPS 2020）
- **核心贡献**：结合局部窗口注意力、全局 token 注意力和随机注意力，将复杂度降至线性。证明了 BigBird 是通用近似器且图灵完备。
- **解决的问题**：标准 MHA 的二次复杂度限制了长序列建模。

### 13. Efficient Transformers: A Survey (Tay et al., 2020)
- **论文链接**：[arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
- **核心贡献**：系统综述了各种高效 Transformer 变体（Reformer, Linformer, Performer, Longformer, BigBird 等），按稀疏注意力、线性注意力、低秩注意力等分类。
- **解决的问题**：高效注意力变体太多，需要一个统一框架来理解和对比。适合作为综述性参考。

---

## 六、组合式/动态多头注意力

这一方向探索"如何让多头之间的组合方式更灵活"。

### 14. Improving Transformers with Dynamically Composable Multi-Head Attention (DCMHA, 2024)
- **论文链接**：[arxiv.org/abs/2405.08553](https://arxiv.org/abs/2405.08553)
- **核心贡献**：提出 Dynamically Composable MHA（DCMHA），让注意力头之间可以动态组合，而非固定独立计算。通过动态路由机制增加模型表达能力，同时保持参数和计算效率。
- **解决的问题**：标准 MHA 中各头独立计算，缺乏头间交互；并非所有头对所有输入都同等重要。

### 15. MoH: Multi-Head Attention as Mixture-of-Head Attention (Jin et al., 2024)
- **论文链接**：[arxiv.org/abs/2410.11842](https://arxiv.org/abs/2410.11842)
- **核心贡献**：将多头注意力重新表述为求和形式，借鉴 MoE（Mixture-of-Experts）思想，把每个注意力头当作一个"专家"，通过动态路由只激活部分头。用 50%~90% 的头即可超越全量 MHA。
- **解决的问题**：标准 MHA 中所有头都被激活，存在计算浪费；不同输入可能只需要不同子集的头。
- **亮点**：可以从预训练的 LLaMA3-8B 继续微调为 MoH 模型，在 14 个 benchmark 上平均准确率从 63.1% 提升到 64.0%。

---

## 七、多头注意力在视觉领域的应用

### 16. An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT, Dosovitskiy et al., 2021)
- **论文链接**：[arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)（ICLR 2021）
- **核心贡献**：将 Transformer 的多头自注意力直接应用于图像 patch 序列，证明在足够数据下可以超越 CNN。
- **解决的问题**：多头注意力从 NLP 迁移到视觉，验证了注意力机制的通用性。

### 17. Improving Vision Transformers by Overlapping Heads in Multi-Head Self-Attention (2024)
- **论文链接**：[arxiv.org/abs/2410.14874](https://arxiv.org/abs/2410.14874)
- **核心贡献**：在 ViT 的 MHSA 中让不同头的 token 划分有重叠，增强了头间的信息共享。
- **解决的问题**：标准 MHSA 中不同头的 token 划分互不重叠，可能丢失跨头的信息。

### 18. BViT: Broad Attention Based Vision Transformer (2022)
- **论文链接**：[arxiv.org/abs/2202.06268](https://arxiv.org/abs/2202.06268)
- **核心贡献**：提出"宽带注意力"，通过跨层连接引入不同层之间的注意力关系。
- **解决的问题**：标准 MHA 仅在同一层内计算注意力，BViT 让注意力跨越不同层。

---

## 八、注意力计算的高效实现

### 19. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- **论文链接**：[arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)（NeurIPS 2022）
- **核心贡献**：提出 IO 感知的精确注意力算法，通过 tiling（分块）和 recomputation（反向重计算）避免将 N×N 注意力矩阵写入 GPU HBM，大幅减少内存读写。精确（非近似），已成为 PyTorch 2、vLLM 等的默认实现。
- **解决的问题**：标准 MHA 的内存访问开销大，成为训练和推理的瓶颈。不改变数学语义，只改变计算方式。

---

## 九、理论分析

### 20. On the Optimization and Generalization of Multi-head Attention (2023)
- **论文链接**：[arxiv.org/abs/2310.12680](https://arxiv.org/abs/2310.12680)
- **核心贡献**：从理论上研究多头注意力通过梯度下降训练时的优化和泛化性质，利用多头的结构给出有限时间收敛保证。
- **解决的问题**：MHA 的训练动力学缺乏理论理解。

### 21. Provably Learning a Multi-Head Attention Layer (2024)
- **论文链接**：[arxiv.org/abs/2402.04084](https://arxiv.org/abs/2402.04084)
- **核心贡献**：首次从可证明学习（provably learning）的角度研究多头注意力层，给出非平凡的上界和下界。
- **解决的问题**：多头注意力层的样本复杂度和学习难度。

---

## 汇报建议：按"演进逻辑"组织讲解

建议按以下逻辑线来组织你的汇报，而不是逐篇讲：

**第一条线：效率与推理优化（MHA → MQA → GQA → MLA）**
- 从原始 MHA 的 KV Cache 瓶颈出发 → Shazeer 提出 MQA（共享 K/V）→ GQA 折中（分组共享）→ DeepSeek MLA（压缩-恢复）。这条线是当前大模型最核心的架构演进。

**第二条线：分析与理解（多头到底在干什么？）**
- Michel 2019（头很冗余）→ Voita 2019（部分头有语言功能）→ Clark 2019（BERT 头编码句法）。展示了从"发现冗余"到"理解功能"的深入。

**第三条线：改进多头机制本身**
- 多样性增强（Li 2021, MDPI 2024）→ 动态组合（DCMHA）→ MoE 化头（MoH）。展示了如何让多头更高效、更灵活。

**第四条线：长序列效率**
- 稀疏注意力（Longformer, BigBird）→ FlashAttention 实现。展示了从"改架构"到"改实现"的思路。

---

## 快速参考表

| 编号 | 论文 | 年份 | 类别 | 核心问题 |
|------|------|------|------|----------|
| 1 | Attention Is All You Need | 2017 | 奠基 | 提出多头注意力 |
| 2 | Are Sixteen Heads Really One Head | 2019 | 分析 | 头的冗余性 |
| 3 | Analyzing MHA (Voita) | 2019 | 分析 | 头的语言学功能 |
| 4 | What Does BERT Look At | 2019 | 分析 | 头编码句法结构 |
| 5 | The Heads Hypothesis | 2022 | 分析 | 统一分析框架 |
| 6 | On the Diversity of MHA | 2021 | 多样性 | 头间冗余改进 |
| 7 | Diversifying MHA | 2024 | 多样性 | Hebbian 学习强制多样化 |
| 8 | Multi-Query Attention (MQA) | 2019 | 效率 | KV Cache 压缩（共享 K/V） |
| 9 | Grouped-Query Attention (GQA) | 2023 | 效率 | MHA-MQA 折中 |
| 10 | Multi-Head Latent Attention (MLA) | 2024 | 效率 | 低维压缩 KV Cache |
| 11 | Longformer | 2020 | 长序列 | 局部+全局注意力 |
| 12 | BigBird | 2020 | 长序列 | 局部+全局+随机注意力 |
| 13 | Efficient Transformers Survey | 2020 | 综述 | 高效注意力变体汇总 |
| 14 | DCMHA | 2024 | 动态组合 | 头间动态交互 |
| 15 | MoH (Mixture-of-Head) | 2024 | 动态组合 | MoE 化注意力头 |
| 16 | ViT | 2021 | 视觉 | 多头注意力用于图像 |
| 17 | Overlapping Heads in ViT | 2024 | 视觉 | 头间 token 重叠 |
| 18 | BViT | 2022 | 视觉 | 跨层注意力 |
| 19 | FlashAttention | 2022 | 实现 | IO 感知精确注意力 |
| 20 | Optimization of MHA | 2023 | 理论 | 训练动力学理论 |
| 21 | Provably Learning MHA | 2024 | 理论 | 可证明学习边界 |
