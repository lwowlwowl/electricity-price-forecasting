# 15｜RT执行策略对照与MPC候选

## 这一步比较什么

滚动RT Transformer每小时预测未来4小时价格。这里比较的不是模型能否看到4小时，
而是怎样把同一组4小时预测变成当前动作。

- `rt_only`：HardTopK比较4小时价格高低，生成候选动作，只执行第1步。
- `follow_da`：实际动作完全照搬已经锁定的DA计划。
- 旧`track_adjust`：先由HardTopK生成RT候选，再用当前小时的局部预计收益决定
  是否偏离DA。它没有连续推演后续SOC，现已废弃。
- 4小时约束MPC：比较`充/停/放`的`3^4=81`条路线，每一步都计算SOC、效率、
  日放电额度和吞吐成本，选择4小时总价值最高的路线，但只执行第1步。

因此MPC不是第一次“看未来4小时”。真正新增的是完整的多步动作和SOC推演。

## 为什么不强迫RT跟随DA

当前不启用偏差罚金。到了实时阶段，`u_DA`已经锁定。在
`p_RT × (u_RT_actual - u_DA)`中，比较不同RT动作时，`-p_RT × u_DA`是共同常数，
不会改变哪个`u_RT_actual`最好。

所以`u_DA`仍进入最终双结算账本，但无罚金时不应为了“看起来有关联”而强制RT
跟随DA。如果以后加入真实偏差罚金，MPC再把`u_DA`和罚金显式纳入目标。

## 相同旧RT模型下的策略对照

固定RT-at-DA seed 2、滚动RT seed 2，并对Transformer DA与XGBoost DA两个参考
系统的完整双结算验证收益取平均：

| 执行方式 | 验证平均收益/天 |
|---|---:|
| `rt_only` | **148.51** |
| 4小时约束MPC | 145.91 |
| `follow_da` | 62.13 |
| 旧`track_adjust` | 34.13 |

MPC把旧`track_adjust`的大部分损失修复了，但仅更换策略仍略低于`rt_only`。

## 加入MPC收益微调以后

再分别微调三个RT seed，并用验证集完整双结算收益选择，最佳候选是：

`RT seed 2 + 4小时约束MPC = 151.18/天`

它在验证集比正式`rt_only`的148.51/天高2.67/天。但是最终测试结果为：

| DA参考 | 正式`rt_only` | MPC候选 | 变化 |
|---|---:|---:|---:|
| Transformer DA | 82.98 | 80.73 | -2.25 |
| XGBoost DA | 95.89 | 93.64 | -2.25 |
| 两者平均 | **89.44** | 87.18 | -2.25 |

MPC平均每天少约5.07的吞吐成本，但RT偏差腿少约7.32，净收益最终少约2.25。
因此它是“逻辑更完整但泛化未通过”的候选，不替换正式方案。

## 当前结论与证据

1. `track_adjust`不再作为后续候选或当前结果使用。
2. 正式执行仍是`RT seed 2 + rt_only`。
3. MPC代码和checkpoint保留用于以后在新验证切分上研究更长窗口和终点机会价值。
4. 已经看过的测试期不再用于MPC调参。

证据：

- 策略实现：`src/decision_aware/policy.py`；
- 滚动回测：`src/decision_aware/backtest_rt.py`；
- 完整双结算：`src/decision_aware/backtest_dual.py`；
- 策略对照：`data/results/mpc_strategy_existing_rt_comparison.json`；
- 最终选择：`data/results/mpc_decision_aware_validation_selection.json`。
