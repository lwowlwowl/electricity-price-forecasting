# 三天吃透本项目所需的 Transformer

> 目标不是在三天内学完整个深度学习领域，而是能够独立阅读、解释、调试和修改本项目中的 Transformer，并能判断修改会如何影响张量形状、梯度、信息边界与实验结果。

## 使用方法

- 默认每天投入 8～9 小时，每 75～90 分钟休息 10～15 分钟。
- 学习比例建议：25% 阅读，50% 手算与代码，25% 闭卷复述。
- 每个概念必须完成三件事：手算一次、在项目代码中找到一次、闭卷讲一次。
- 每完成一个任务，将 `- [ ]` 改为 `- [x]`。
- 如果某道验收题讲不清楚，不要继续堆新知识，先返回对应代码和手算例子。

## 三天后的验收目标

- 能解释 Q、K、V 和多头注意力的完整计算过程。
- 能区分 batch、序列长度、`d_model`、头数和每头维度。
- 能解释残差连接、LayerNorm、FFN、Dropout 和 RoPE。
- 能区分 self-attention、cross-attention、causal mask 和数据泄漏。
- 能从 batch 开始追踪项目全部关键张量形状。
- 能讲清 `StreamEncoder → CrossModalFusion → QueryDecoder → 输出头`。
- 能解释五条输入流、modality embedding 和 learnable query。
- 能理解预测损失、反向传播、`detach`、零阶梯度和 `L_proxy` 的基本关系。
- 能设计 DA-only 模型的最小改造方案。
- 能检查一次实验是否存在未来信息泄漏或评价口径不一致。

---

# 第一天：Transformer 本体

当天目标：从张量形状出发，手算并解释一个完整的 Transformer Block。

## 09:00—09:30：建立知识地图

阅读 `transformer_from_scratch.md`：

- 第一、二章快速浏览，只理解 RNN 为什么有局限。
- 第三章 Q、K、V 仔细阅读。
- 第四章自注意力仔细阅读。
- 第五章整体架构仔细阅读。
- 第六章多头注意力仔细阅读。
- 第七章暂时跳过。

先记住主线：

```text
输入序列
→ 生成 Q/K/V
→ 计算注意力
→ 多头拼接
→ 残差和 LayerNorm
→ FFN
→ 得到新的序列表示
```

任务：

- [ ] 写下目前最困惑的 5 个问题。
- [ ] 不看资料画一次 Encoder Block，允许画错。

## 09:30—11:00：张量和矩阵基础

掌握：

- `B`：batch size，一批样本数。
- `T`：序列长度，例如 168 个历史小时。
- `D`：每个时刻的表示维度，例如 256。
- 输入常见形状为 `[B,T,D]`。
- 矩阵乘法、reshape、transpose、permute、broadcasting 和 softmax。

项目示例：

```text
d_model = 256
n_heads = 4
每头维度 dh = 256 / 4 = 64
```

任务：

- [ ] 能解释 `[64,168,256]` 的每一个维度。
- [ ] 能解释 `[B,4,168,64]` 的每一个维度。
- [ ] 完成 5 道 reshape/permute 形状题。

## 11:15—13:00：Q、K、V 与缩放点积注意力

直觉：

- Query：当前位置在寻找什么。
- Key：每个位置提供的索引标签。
- Value：真正被取回的信息。

计算过程：

```text
Q = XWq
K = XWk
V = XWv
scores = QKᵀ / √dh
weights = softmax(scores)
output = weights × V
```

必须理解：

- 除以 `√dh` 是为了控制点积数值尺度，避免 softmax 过早饱和。
- 对每一个 Query，在所有 Key 上做 softmax。
- 注意力矩阵的一行表示当前 Query 对所有位置的关注权重。

任务：

- [ ] 手算一个 3 个 token、每个 token 2 维的注意力例子。
- [ ] 用“问题—索引—内容”的比喻讲清 Q/K/V。

## 14:00—15:30：多头注意力与项目代码

阅读 `src/decision_aware/model.py` 中的 `RotaryMHA.forward`。

追踪形状：

```text
x                 [B,T,D]
qkv projection    [B,T,3D]
reshape           [B,T,3,H,dh]
permute           [3,B,H,T,dh]
q, k, v           [B,H,T,dh]
attention output  [B,H,T,dh]
concat heads      [B,T,D]
```

任务：

- [ ] 逐行标注 `RotaryMHA.forward` 的形状。
- [ ] 解释多头不是复制同一个注意力，而是把表示空间拆成多个子空间。
- [ ] 解释为什么 `d_model` 必须能被头数整除。

## 15:45—17:15：TransformerBlock

阅读 `TransformerBlock`。

项目使用 Pre-LN：

```python
x = x + attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

必须理解：

- 残差连接保留旧信息并改善梯度传播。
- LayerNorm 稳定每个 token 的特征尺度。
- Attention 负责不同时间点之间的信息交流。
- FFN 负责每个时间点内部的非线性加工。
- Dropout 用于减少过拟合。
- Final LayerNorm 稳定整个 Pre-LN 堆栈出口的尺度。

任务：

- [ ] 能解释两个残差分支各自包围什么模块。
- [ ] 能解释 Attention 与 FFN 的职责差异。
- [ ] 能解释 Pre-LN 和 Final LayerNorm 为什么不重复。

## 17:30—18:30：位置编码与 RoPE

阅读 `RotaryEmbedding`。

掌握：

- 注意力本身不天然知道先后顺序。
- RoPE 将位置信息写入 Q 和 K，改变位置之间的注意力关系。
- Value 主要承载内容，一般不旋转。
- 当前阶段不要求推导全部三角函数。

任务：

- [ ] 解释完全没有位置编码会发生什么。
- [ ] 解释为什么历史流使用 RoPE。
- [ ] 暂时记录“融合层为什么关闭 RoPE”，第二天回答。

## 19:30—21:00：最小实现与复述

写一个最小注意力前向：

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(d)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

任务：

- [ ] 加入 shape assert。
- [ ] 检查输出没有 NaN/Inf。
- [ ] 闭卷讲 10 分钟：Q/K/V、缩放、多头、残差、LayerNorm、FFN、RoPE。

## 第一天验收题

1. 为什么 attention score 要除以 `√dh`？
2. `d_model=256`、4 个头时每头多少维？
3. softmax 的一行代表什么？
4. 多头如何拆开并拼回？
5. 残差连接解决什么问题？
6. Attention 与 FFN 有什么分工？
7. 没有位置编码会怎样？
8. Pre-LN 的计算顺序是什么？

---

# 第二天：吃透本项目的模型结构

当天目标：闭卷追踪 v8 的完整前向传播，并解释每个架构选择。

主要材料：

- `src/decision_aware/model.py`
- `configs/decision_aware/formal_ercot_v8.yaml`
- `src/decision_aware/dataset_v3.py`

## 09:00—10:30：五条输入流与 StreamEncoder

当前 v3 模型输入：

1. 历史日前价格。
2. 历史实时价格。
3. 历史负荷。
4. 风电、光伏等系统变量。
5. 日历特征。

各流经过：

```text
原始输入
→ Linear 投影到 d_model
→ Transformer 或 MLP
→ final LayerNorm
```

典型形状：

```text
[B,168,输入维度] → [B,168,256]
```

任务：

- [ ] 找到五个 Encoder 的定义。
- [ ] 写出每条流原始输入维度。
- [ ] 解释为什么先分别编码，再做融合。

## 10:45—12:15：modality embedding 与 CrossModalFusion

每条流先加 modality embedding，再拼接：

```text
5 × [B,168,256]
→ [B,840,256]
```

必须区分：

- 位置编码表示“这是第几个小时”。
- modality embedding 表示“这是哪一种数据”。

融合层关闭 RoPE 的原因：拼接后相邻 token 可能属于不同模态，并不代表物理时间相邻。如果直接按拼接位置使用 RoPE，可能形成错误的位置先验。

任务：

- [ ] 解释 `5×168=840`。
- [ ] 找到 `_fuse` 中添加 modality embedding 的代码。
- [ ] 解释融合层为什么不用 RoPE。
- [ ] 理解融合注意力对长度约为 `O(L²)`，840 长度会形成约 `840×840` 的注意力矩阵。

## 13:15—14:45：QueryDecoder

QueryDecoder 包含：

- Learnable content query。
- Query 位置编码。
- 从 memory 均值产生的输入条件化 `ctx`。
- Query 之间的 self-attention。
- Query 对 memory 的 cross-attention。
- FFN 和 final LayerNorm。

直觉：

- 48 个 Query 对应未来第 1～48 小时。
- Query self-attention 让未来小时彼此协调。
- Cross-attention 让每个未来小时从历史 memory 中取信息。
- Cross-attention 中 Q 来自未来小时槽，K/V 来自历史 memory。

任务：

- [ ] 逐行阅读 `QueryDecoder.forward`。
- [ ] 解释 query self-attention 与 cross-attention 的区别。
- [ ] 解释 `query_pos` 与 `ctx` 分别解决什么问题。

## 15:00—16:30：完整 forward 形状账本

五流编码：

```text
h_da    [B,168,256]
h_rt    [B,168,256]
h_load  [B,168,256]
h_sys   [B,168,256]
h_cal   [B,168,256]
```

融合：

```text
memory [B,840,256]
```

日前联合 Decoder：

```text
rep               [B,48,256]
head_da(rep)       [B,48,2]
p_da               [B,48]
p_rt_da            [B,48]
```

实时 Decoder：

```text
rt_rep             [B,24,256]
p_rt               [B,24]
p_rt_windows       [B,24,4]
```

最后还会进行反归一化和价格范围 clamp。

任务：

- [ ] 闭卷写出上述全部形状。
- [ ] 从 `batch` 字段一路追到四种输出。
- [ ] 解释 `[B,24,4]` 中三个维度分别是什么。

## 16:45—18:00：mask 与未来信息泄漏

当前模型没有 GPT 式 causal mask，不一定是错误：

- Encoder 看到的是起报时已经发生的 168 小时历史。
- 历史窗口中的位置可以彼此注意。
- Decoder 一次性预测未来曲线，并不是逐 token 生成文本。
- Future query 之间交流不等于读取未来真实价格。

必须区分：

- Causal mask：限制模型内部可见位置。
- 数据泄漏：输入中出现预测时点尚未公开的信息。

任务：

- [ ] 判断历史实际天气能否用于 DA 起报。
- [ ] 判断天气预报应使用哪个发布版本。
- [ ] 解释“没有 causal mask”为什么不等于“没有时间约束”。

## 19:00—20:00：超参数与计算成本

研究：

- `d_model`
- `n_heads_enc`
- `n_heads_fusion`
- `n_layers_enc`
- `n_layers_fusion`
- `dim_ff`
- `dropout`
- `context_len`
- `horizon_da`
- `horizon_rt`

任务：

- [ ] 解释增大 `d_model` 对参数量和表达能力的影响。
- [ ] 解释增加头数为什么不一定增加总表示维度。
- [ ] 解释增大上下文为什么会显著增加注意力计算量。
- [ ] 解释 Encoder 层数与 Fusion 层数的区别。

## 20:00—21:15：合成 batch 调试

构造符合字段与形状的随机 batch，运行一次 forward。目标不是训练，而是检查：

- 输入字段是否正确。
- 中间形状是否正确。
- 输出是否包含 NaN/Inf。
- 出现维度错误时能否定位。

任务：

- [ ] 打印五流 Encoder 输出形状。
- [ ] 打印 memory、rep 和四类预测的形状。
- [ ] 故意制造一个输入维度错误并定位原因。

## 第二天验收题

1. 五条长度 168 的流拼接后 memory 长度是多少？
2. modality embedding 和位置编码有什么区别？
3. 融合层错误开启 RoPE 会引入什么先验？
4. QueryDecoder 中 self-attention 的 Q/K/V 来自哪里？
5. Cross-attention 的 Q/K/V 来自哪里？
6. 为什么需要 `query_pos`？
7. 为什么需要输入条件化 `ctx`？
8. 为什么没有 causal mask 不一定泄漏？

---

# 第三天：训练、业务闭环与独立改模

当天目标：理解梯度如何穿过项目，并能设计 DA-only 改造及公平实验。

主要材料：

- `src/decision_aware/loss.py`
- `src/decision_aware/zero_order.py`
- `src/decision_aware/policy.py`
- `scripts/decision_aware/train_formal.py`

## 09:00—10:15：PyTorch 反向传播基础

掌握最小知识集：

- 计算图。
- `requires_grad`。
- `loss.backward()`。
- 参数的 `.grad`。
- `detach()`。
- `torch.no_grad()`。
- 一个输出没有进入 loss 时，对应参数不会获得梯度。

练习：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 4
```

任务：

- [ ] 用一个 Linear 层检查权重梯度。
- [ ] 在中间加入 `detach()`，观察梯度如何消失。
- [ ] 画出 `loss → heads → decoder → fusion → encoders` 的梯度路径。

## 10:30—11:45：预测损失与数值稳定性

理解：

- MSE。
- Huber loss。
- `pred_scale`。
- 标准化与反标准化。
- 输出 clamp。
- 梯度裁剪。
- NaN/Inf。

电价尖峰会让 MSE 平方项非常大。Huber 在小误差区间类似 MSE，在大误差区间更接近 MAE，因此对尖峰更加稳健。

任务：

- [ ] 比较误差 10 与误差 100 时 MAE、MSE 的变化。
- [ ] 找到项目选择 Huber 的配置。
- [ ] 解释不同损失项为什么必须做尺度平衡。

## 12:00—13:00：预测到储能收益的闭环

复习：

```text
预测价格
→ HardTopK 选择充放电时段
→ SOC、效率与循环约束裁剪
→ 双结算收益
→ 业务损失
```

任务：

- [ ] 用 2～3 小时的小例子手算一次 SOC 与收益。
- [ ] 解释预测 MAE 更低为什么不保证收益更高。
- [ ] 解释为什么 Oracle 必须使用与模型相同的约束和成本。

## 14:00—15:30：零阶梯度与 L_proxy

理解流程：

1. HardTopK 包含排序和离散选择。
2. 普通反向传播无法直接穿过硬排序。
3. 将预测沿随机方向增加一点，计算业务损失。
4. 沿相反方向减少一点，再计算业务损失。
5. 比较两边函数值，估计修改预测的方向。
6. 通过 `L_proxy` 把估计方向注入神经网络。

重点参数：

- `ε`：扰动幅度。
- `ρ`：扰动相对价格标准差的比例。
- `K`：随机方向数量。
- `stopgrad/detach`：把估计业务梯度视为常数。

任务：

- [ ] 解释 ε 太小为什么可能产生零梯度。
- [ ] 解释 ε 太大为什么会偏离当前预测。
- [ ] 解释 K 增大为什么更稳定但更慢。
- [ ] 解释 `L_proxy` 的数值为什么不等于真实业务损失。

## 15:45—17:00：训练循环

追踪一个 batch：

```text
DataLoader
→ model(batch)
→ Lpred
→ 储能仿真
→ 零阶业务梯度
→ Lproxy
→ L = αLpred + βLproxy
→ backward
→ gradient clipping
→ optimizer.step
```

理解预训练与退火：

- 前期 `β≈0`，先学习基本预测。
- 后期逐渐增加 `β`，再引入业务收益目标。
- 防止模型尚未学会基本价格规律时被高噪声业务信号带偏。

任务：

- [ ] 找到 α/β 的退火函数。
- [ ] 找到 optimizer、AMP、grad clip 和 checkpoint 逻辑。
- [ ] 解释为什么 pretrain checkpoint 不应直接当成最终 decision-aware 模型。

## 17:15—18:15：梯度检查

检查参数梯度：

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm())
```

梯度为 `None` 的常见原因：

- 对应输出没有进入 loss。
- 中间被 `detach()`。
- 在 `torch.no_grad()` 中计算。
- 条件分支没有执行。

梯度为 0 的常见原因：

- `β=0`。
- 扰动没有改变 TopK。
- 损失项被乘 0。
- 输出被裁剪到饱和区。
- 业务梯度尺度太小。

梯度爆炸的常见原因：

- MSE 遇到极端价格。
- 学习率过大。
- 多个损失尺度不匹配。
- 缺少梯度裁剪。
- fp16 数值溢出。

任务：

- [ ] 检查每个预测 head 是否有非零梯度。
- [ ] 人为 `detach()` 一次并定位梯度中断点。
- [ ] 记录正常梯度范数的大致数量级。

## 19:00—20:00：实验正确性与数据泄漏

掌握：

- 按时间顺序划分训练、验证、测试集。
- 测试集不能参与超参数选择。
- 日前模型只能使用起报时已经公开的信息。
- 实际天气不能冒充天气预报。
- 事后修订负荷不能作为当时可用输入。
- 不同模型使用相同测试窗口、随机种子和储能参数。
- Oracle 使用相同 SOC、效率、循环与成本约束。

指标分为：

- 预测指标：MAE、RMSE。
- 业务指标：收益、Oracle 收益、Regret、PCR。

任务：

- [ ] 写出一份数据泄漏检查清单。
- [ ] 解释验证集和测试集职责的区别。
- [ ] 解释为什么 MAE 与收益可能给出不同模型排序。

## 20:00—21:30：DA-only 毕业设计

先设计，不急着修改代码。

DA-only 应当：

- 只使用日前起报时可获得的信息。
- 只输出未来 24 小时 `p_da [B,24]`。
- 不使用 `p_rt_da`、`p_rt`、`p_rt_windows`。
- 不让 RT 信号参与运行时主动动作。
- 保留 StreamEncoder、modality embedding、CrossModalFusion、QueryDecoder 和 final LayerNorm。
- 保留日前预测损失和日前决策目标。
- 重新确认计划 SOC 的可行性约束。
- 单独定义 DA-only 的收益与 LP Oracle。

任务：

- [ ] 列出需要删除的输出头和损失分支。
- [ ] 列出保留的 Transformer 底座。
- [ ] 写出输入时间截止规则。
- [ ] 写出与联合模型公平比较的控制变量。
- [ ] 用 10 分钟闭卷讲完整项目。

---

# 最终闭卷考试

至少答对 12/15；其中张量形状题必须全对。

1. Q、K、V 分别代表什么？
2. 为什么注意力分数要除以 `√dh`？
3. softmax 在哪个维度计算？
4. `d_model=256`、4 个头时每个头多少维？
5. 多头注意力如何从 `[B,T,D]` 拆开并拼回？
6. 残差连接解决什么问题？
7. Pre-LN 与 Final LayerNorm 分别在哪里？
8. FFN 与 Attention 的职责有什么不同？
9. RoPE 与 modality embedding 有什么区别？
10. 五条长度 168 的输入流融合后序列长度是多少？
11. QueryDecoder 中 self-attention 和 cross-attention 各自做什么？
12. 为什么当前模型没有 causal mask 也不一定泄漏？
13. `p_da`、`p_rt_da`、`p_rt` 和 `p_rt_windows` 分别是什么？
14. `detach` 放错位置为什么会让某个预测头学不到业务信号？
15. DA-only 改造应保留和删除哪些模块？

## 毕业标准

- 形状能力：任给 B、T、D、H，能写出 Q/K/V、融合 memory 和四类输出的形状。
- 解释能力：10 分钟白纸讲完模型，能回答“为什么这样设计”。
- 调试能力：能定位 shape mismatch、无梯度、零梯度和 NaN 的常见原因。
- 改模能力：能设计 DA-only，并说明每处改动对输出、损失、梯度和实验公平性的影响。

---

# 三天内暂时跳过

- Transformer 的严格理论证明。
- BERT/GPT 的完整预训练流程。
- KV Cache、MQA、GQA、MLA。
- FlashAttention 的 CUDA 内核实现。
- 分布式训练。
- RLHF 与大模型对齐。
- 全部注意力变体论文。

`multi_head_attention_literature.md` 只需要阅读原始 Transformer、注意力头冗余和 FlashAttention 的基本动机。

## 最终原则

读完不等于学会。只有当你能闭卷解释、正确推导形状，并在代码中定位对应实现时，才算真正掌握。
