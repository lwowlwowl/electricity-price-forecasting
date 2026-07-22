# configs/archive — 旧范式配置归档

> 状态：**已废弃**。新方向配置将放在 `configs/decision_aware/`（待建）。

## 归档内容

| 目录 | 内容 |
|------|------|
| `fusion/` | ElecFM v1→V7 各版本训练配置 |
| `structural_ablation/` | v2.0 结构消融配置（TimesFM/Chronos2/Toto2 全量 + 逐层 + 重跑） |
| `parameter_ablation/` | v1.0 参数消融配置（协变量/上下文/多变量/步长/频率/微调，含 smoke 与子窗口） |

## 保留在主线

- `configs/nodes.yaml` — ERCOT 节点分组（波动率/尖峰/稳定），新方向仍可用。
