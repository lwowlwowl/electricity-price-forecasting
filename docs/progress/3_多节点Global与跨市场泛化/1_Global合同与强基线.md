# Global合同、连续MPC与强基线

> 本页汇总 ERCOT 15 节点的 2023 内部开发阶段。模型使用已知节点身份，不能作为
> 新市场零样本证据；无 ID 公平协议从阶段 4 开始。

## 任务与样本

Global 表示同一任务在 15 个 ERCOT 节点间共享参数，仍保留三个互不共享参数的模型：

1. DA：D-1 起报，预测交付日 24 小时 DA；
2. RT-at-DA：同一 D-1 信息集下预测交付日 24 小时 RT；
3. rolling RT：每小时预测未来 4 小时，只执行第一步。

三项预测在 15 节点、362 个共同交付日上严格对齐。不同节点的 SOC 独立，真实时间
缺口才重置；预测档案中的同小时真实值不一致时评价入口直接失败。

## 冻结结算合同

- 电池：1 MW / 4 MWh，效率 0.95，SOC 0.4—3.6 MWh，初始 SOC 2 MWh；
- 退化成本：`5.7 USD/MWh × 实际吞吐量`；
- 偏差容忍：`0.03 × |锁定DA计划|`；
- 超额偏差罚金：`2 × |真实RT价| × 超额偏差MWh`；
- DA 计划锁定，RT 每小时滚动重优化，期末 SOC 只报告、不计入现金收益；
- 真实 DA/RT 价格只在动作确定后结算，不参与决策。

日前阶段先根据预测 `DA - RT-at-DA` 的符号确定充放方向，再由连续 LP 决定
0—1 MW 功率；实时阶段在锁定计划、真实 SOC、剩余日放电额度和 4 小时 RT 预测下
求解 MPC，只执行首小时。该方法应称为“价差方向约束的连续 LP/MPC”，不是 Oracle。

## 低预算结果与问题定位

3 epoch、seed 0 的最佳 Global 三元组为 `DA F1 + RT-at-DA F0 + RT F0`：

| 系统 | 收益（USD/节点·日） | 7日块95%区间 | 尾部10% | 最大回撤 |
|---|---:|---:|---:|---:|
| 低预算 Global | 134.58 | [44.00, 248.47] | -98.83 | 2685.83 |
| Seasonal | 213.78 | [54.02, 411.64] | -460.93 | 9733.25 |

Global 的三任务 MAE 均低于 Seasonal，但预测价差方向正确率只有 48.72%，预测/真实
价差相关系数为 0.026，预测价差绝对值均值 17.09，真实值 34.89。问题不是单项
MAE，而是独立模型压缩了真正驱动日前决策的跨任务价差方向和极端幅度。

## 稳定训练结果

降低学习率、延长到最多 20 epoch，并先按训练内部 MAE 每任务保留两个候选，随后只
评价 `2×2×2` 个完整三元组。稳定冠军为 DA F1 epoch 10、RT-at-DA F0 epoch 10、
rolling RT F1 epoch 8：

- 收益：`192.83 USD/节点·日`，旧低预算结果为 134.58；
- 7日块区间：`[74.37, 335.79]`；
- 15/15 节点全年平均为正，SOC 违规为 0；
- DA、RT-at-DA、rolling RT 首步 MAE：41.59、43.66、31.40；
- 价差方向正确率提高到 56.58%，但价差 99% 分位仍只有 189.82，低于真实 721.26。

完整训练修复了欠训练和部分价差压缩，但没有超过 Seasonal，配对差区间仍跨 0。

## 稳健XGBoost强基线

损失函数只用 2020—2022 内部切分选择：DA 和 rolling RT 使用 absolute error，
RT-at-DA 使用 pseudo-Huber；随后用完整 2020—2022 重训并在 2023 一次评价。

| 系统 | DA MAE | RT-at-DA MAE | rolling RT MAE | 收益 |
|---|---:|---:|---:|---:|
| 稳定 Global | 41.59 | 43.66 | 31.40 | 192.83 |
| 稳健 XGBoost | 35.51 | 35.23 | 27.14 | **292.29** |

XGBoost 相对 Seasonal 提高 78.51，配对 7 日块区间 `[18.58, 151.62]`；15/15
节点为正，SOC 违规为 0。逐任务替换显示收益缺口主要来自 DA（-133.39），其次是
RT-at-DA（-39.99），rolling RT 影响较小（-4.26）。

## 已否决的修正

- 继续用 MAE 微调 DA 虽将 MAE 从 41.51 降到 39.59，收益却从 158.90 降到
  153.47；
- `1.20 × XGBoost - 0.20 × Global` 的 seed 0 内部收益达到 298.28，但 seed 1/2
  分别为 279.78/278.78，三 seed 集成为 291.28，未超过 XGBoost 292.29；
- 因此 seed 0 反向融合没有通过多 seed 稳定性门槛，不进入外层年份。

## 决策训练链

连续 LP/MPC 的成对零阶收益梯度已经打通，DA、RT-at-DA、rolling RT 三套参数梯度
均有限非零且参数集合无交集。这只证明训练链成立，不证明模型优于强基线。后续结构
试验及 2024 外层验证见本阶段其余文档。

## 主要证据

- `data/results/global_continuous_mpc_backtest_stable_v2.json`；
- `data/results/global_f0_f1_stable_v2_summary.json`；
- `data/results/global_xgboost_robust_continuous_mpc_backtest_v1.json`；
- `data/results/global_task_replacement_diagnosis_v1.json`；
- `data/results/global_da_robust_continuous_mpc_v1.json`；
- `data/results/global_joint_continuous_zo_smoke.json`。
