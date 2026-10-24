# 无ID通用模型与结果口径

## 为什么必须移除实体ID

早期 Global 模型把 `node_id` 和 `market_id` 作为可学习 embedding。它适合训练期已见
节点，但新市场没有对应向量，因此不能证明零样本迁移。带 ID 结果仍可作为已知实体
上界和工程验证，不再称为通用模型。

最终无 ID 合同固定为：

```yaml
global_use_entity_embeddings: false
global_num_nodes: 0
global_num_markets: 0
global_normalization_mode: context_robust
```

节点和市场标识只用于采样、指标汇总、SOC 隔离和文件审计，不进入模型前向、损失或
门控。`context_robust` 只根据起报前历史窗口做中位数/MAD归一化，不依赖目标节点的
训练期 scaler。

## 留一市场协议

分别留出 CAISO、ERCOT、NYISO、PJM；目标市场完全不提供训练标签。三个任务独立训练，
每折只使用另外三个市场 2020—2022，并在目标市场 2023 评价。所有市场复用同一结构、
信息边界和训练规则，不为目标市场搜索阈值。

## 2023预测结果

| 留出市场 | DA MAE | RT-at-DA MAE | rolling RT MAE |
|---|---:|---:|---:|
| CAISO | 18.20 | 21.03 | 9.75 |
| ERCOT | 37.27 | 34.80 | 28.86 |
| NYISO | 8.55 | 10.73 | 7.13 |
| PJM | 6.20 | 8.96 | 6.57 |

12/12 项均优于同市场 lag-168 Seasonal，说明模型学到可迁移规律；这只证明预测迁移，
不等于跨市场真实盈利。CAISO、NYISO、PJM 的独立市场结算合同未冻结，因此不报告
真实市场利润。

## ERCOT完整结算

ERCOT 留出折只在 CAISO、NYISO、PJM 训练。三条预测接入统一连续 LP/MPC，在15节点、
362个共同交付日上评价：

| 系统 | 收益（USD/节点·日） | 正收益日 | 最大回撤 |
|---|---:|---:|---:|
| 无 ID Transformer seed 0 | **377.84** | 72.9% | **195.01** |
| 逐节点稳健 XGBoost | 292.29 | 79.3% | 206.07 |
| lag-168 Seasonal | 213.78 | 46.7% | 9733.25 |

相对 XGBoost 点估计为 +85.55，但7日块区间 `[-12.70, 204.87]` 跨0；删除最高5日后
仍高于 XGBoost，但尚需多 seed 和外层年份确认。后续同数据 pooled LOMO XGBoost
把公平门槛提高到 418.0649，详见 `BASELINES.md`。

## 2024冻结确认

冻结 seed 0、结构和超参数，不重训、不读取2025，只在2024生成预测并复用同一结算：

| 系统 | 收益 | 相对差 | 配对7日块95%区间 |
|---|---:|---:|---:|
| 无 ID Transformer | 73.38 | -17.32 | [-30.86, -5.06] |
| 逐节点 XGBoost | 90.69 | — | — |

自有模型只有1/15节点超过XGBoost；去掉各自最高5日后仍落后。2023的点估计领先没有
跨年延续，因此不得用2024反向调参或改选 seed。

## 适用边界

- “无 ID”表示网络前向不使用节点/市场查找表，不表示适用于所有世界市场；
- 新市场仍需满足字段、频率、发布时间和目标定义；
- 带 ID 的历史结果没有作废，但只能解释已见实体；
- 2023是开发结果，2024是冻结确认，二者不能混池选模；
- 2025最终结论见 `5_最终四市场训练与2025锁定结果.md`。

## 证据

- `data/results/universal_lomo_ercot_profit_continuous_mpc_v1.json`；
- `data/results/universal_lomo_ercot_outer_frozen_summary_v1.json`；
- `data/results/universal_outer_frozen_completed_20261013.tar.gz`。
