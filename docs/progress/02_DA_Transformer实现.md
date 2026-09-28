# 02｜独立DA Transformer实现

## 最终采用的结构

代码：`src/decision_aware/model_da.py`。

五条历史流先各自编码，输出都是`[B,168,D]`。随后没有把它们首尾排成
`[B,840,D]`，而是按同一个物理小时堆成`[B,168,5,D]`：

1. 在每个小时内部，让5个来源做Self-Attention；
2. 用可学习权重把5个来源汇聚成1个时间token；
3. 得到`[B,168,D]`后，再用Temporal Transformer学习跨小时关系；
4. 输出的来源权重为`[B,168,5]`，以后可用于检查模型在何时依赖哪个来源。

这个设计直接回应“不要机械地把五个来源线性放在一起”。它是需要消融验证的工程方案，不宣称老师指定了唯一的5×5注意力结构。

## 24小时解码

DA Decoder有24个learnable query。每个query由三部分组成：内容query、query_pos、目标日对应小时的日历投影。之后依次经过Query Self-Attention、对168个memory token的Cross-Attention、FFN和final LayerNorm。

独立DA路径关闭了旧`mean(memory)`上下文偏置，因为它尚未验证有效，而且可能稀释尖峰。旧联合模型仍保留原行为以兼容历史checkpoint。最终输出只有：

- `p_da [B,24]`：真实价格尺度的DA预测；
- `source_weights [B,168,5]`：逐小时来源权重；
- 中间`memory`和`rep`供调试。

它没有`p_rt`或`p_rt_da`输出头，因此不是把旧大模型换个名字。

## 已通过的结构检查

- 输出、memory、representation和来源权重形状正确；
- 每个小时的五来源权重和为1；
- 梯度能回到五个Stream Encoder；
- 改变目标日日历会改变预测；
- 两个模型实例不共享参数；
- DA模型没有RT输出头。

测试：`tests/test_decision_aware_model_da.py`。
