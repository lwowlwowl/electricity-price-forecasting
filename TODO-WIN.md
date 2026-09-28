# TODO-WIN

## QueryDecoder 的全局 `ctx` 是否应该使用 840 个 memory token 的简单平均

**状态：待验证，不直接判定为必须修改。**

### 当前实现

当前 `QueryDecoder` 使用：

```python
ctx = self.ctx_proj(self.ctx_norm(memory.mean(dim=1, keepdim=True)))
q = (self.queries + self.query_pos).unsqueeze(0).expand(B, -1, -1) + ctx
```

- 每个样本包含 5 条输入流，每条流有 168 个历史小时，因此融合后的 `memory` 共有 `5 × 168 = 840` 个 token。
- `memory` 的形状为 `[B, 840, d_model]`；`mean(dim=1)` 是每个样本分别对自己的 840 个 token 求平均，不会在不同样本之间求平均。
- 五种原始数据已先投影到统一的隐藏空间并经过跨模态融合，因此这里不是直接把电价、负荷和日历原始数值相加。
- 得到的 `ctx` 是当前样本共享的全局摘要，同一个样本的所有预测 Query 使用同一个 `ctx`，不同样本通常得到不同的 `ctx`。
- 该设计是项目后续加入的工程方案，不是 PDF 建模说明或标准 Transformer 强制要求的结构。

### 潜在问题

1. 简单平均默认 840 个 token 同等重要，每个 token 的直接权重都是 `1/840`。
2. 五个来源及所有历史时刻被统一汇总，可能削弱来源差异和时间结构。
3. 少数与盈利高度相关的极端电价 token 可能被大量普通 token 稀释。
4. 平均后的 LayerNorm 可能进一步弱化由向量整体尺度承载的市场异常程度。
5. 没有 `ctx` 并不代表 Decoder 最终无法依赖当前样本：后续 Cross-Attention 仍会使用完整 `memory`。因此 `ctx` 只能视为辅助条件化设计，不能视为必需修复。

### 不会发生的事情

- 求平均不会删除原始 Shared Memory；Cross-Attention 仍然可以查询全部 840 个 token。
- 因此风险主要存在于 `ctx` 这条全局摘要支路，而不是全部历史信息都被平均掉。

### 建议消融方案

在数据划分、随机种子、训练预算和其他超参数一致的条件下，对比：

1. `no_ctx`：不添加 `ctx`，仅使用 learnable query、`query_pos` 和 Cross-Attention。
2. `mean_ctx`：保留当前的 840-token 全局平均，作为现有基线。
3. `per_stream_ctx`：五条流分别产生摘要，再通过可学习模块组合，保留来源身份。
4. `attention_pool_ctx`：让模型学习 840 个 token 的不同权重，避免强制等权平均。
5. `global_token_ctx`：在融合层加入可学习的全局 token，用其融合输出作为摘要。

### 评价口径

本项目以业务结果为主，不应仅根据 MAE 决定方案。至少比较：

- 累计净收益；
- LP Oracle Regret；
- PCR（适用时）；
- 极端价格时段的收益；
- 充放电时段选择是否正确；
- MAE、RMSE 仅作为辅助指标。

### 当前建议

在消融结果出来之前保留现有实现，但不要把 `mean_ctx` 写成已经验证有效或理论上必需的结构。优先对比 `no_ctx`、`mean_ctx` 和 `attention_pool_ctx`，再决定是否引入更复杂的分来源摘要或全局 token。
