# src/archive — 旧范式代码归档（冻结零样本 ElecFM）

> 状态：**已废弃，不再维护**。保留仅供查阅可复用片段。
> 新方向：`src/decision_aware/`（Decision-aware 多模态 TSFM，从零训练）。
> 范式依据：`docs/team_notes/w8/研究方案_Decision-aware_多模态TSFM.md`、`docs/team_notes/w9/模型相关.md`、`docs/todo3.md`。

## 为什么归档

旧代码围绕「冻结 TimesFM 权重 + 挂 spike head 的 zero-shot 融合」范式构建，与新方向「从零训练多流 Encoder + 业务损失闭环」在架构层面不可调和（`fusion_model/model.py` 写死了 `timesfm.from_pretrained`、剪枝映射、`quant_head`、`revin`）。强行改造比重写更脏。

**注意**：`foundation.py + workers/`（外部 TSFM 子进程适配器）原曾归档于此，**已移回 `src/models/`** —— 它是范式中立的基线工程管线，不是旧范式专属逻辑，新方向 W9 §7 公平对比仍要用（见 `src/models/README.md`）。

## 归档内容

| 目录 | 原职责 | 是否有可复用片段 |
|------|--------|------------------|
| `fusion_model/` | ElecFM 融合模型（冻结骨干 + spike head + CrossNodeAttention） | `dataset.py` 滑窗逻辑、`evaluate.py` τ\* 阈值搜索、三窗口回测口径可参考 |
| `parameter_ablation/` | v1.0 输入配置消融执行器 | 实验代码废弃；**结论**（协变量加全 +6.7% spike-F1、720h context 最优、15min 微增）已沉淀到 `docs/archive/参数消融*.md`，指导新模型输入设计 |
| `structural_ablation/` | v2.0 结构消融（14 种手术操作 + 逐层消融） | 方法论（Wilcoxon + Bonferroni、逐层/组件消融流程）可迁移到对新自有模型的消融 |

## 保留在主线（未归档）的 src 模块

- `src/models/` — 预测器层：`base.py`（抽象基类）、`forecasters.py`（7 个统计/树基线）、`foundation.py`+`workers/`（外部 TSFM 基线层，双重用途：零样本参考 + 从零训练对照 plumbing 模板）
- `src/data_processing/loader.py` — 多模态数据对齐 `load_slice()`
- `src/evaluation/` — 指标 / 统计检验 / 回测（新增 business 指标后复用）
