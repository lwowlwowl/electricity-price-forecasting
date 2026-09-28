# 15｜DA-aware RT固定策略对照

## 这一步比较什么

DA计划像是“昨天已经写好的24小时计划”，RT动作像是“今天每小时临时决定怎么做”。
本次不改RT Transformer，只比较三种固定执行规则：

1. `rt_only`：按RT价格预测产生的动作执行，不跟随DA计划。
2. `follow_da`：完全按已锁定的DA计划执行。
3. `track_adjust`：默认跟随DA；只有当RT预测表明“这一小时偏离DA更赚钱”时，才改用RT候选动作。

`track_adjust`的判断只用当时可见的RT预测价格、DA计划和退化成本，
没有看真实RT价格。真实价格只在动作确定后用于结算。

## 怎样避免“挑中测试集”

候选组合是3个RT checkpoint × 3种规则，共9组。每组都在验证集上分别搭配：

- 固定的Transformer DA seed 2；
- 固定的XGBoost DA seed 0。

选择分数是两个完整DA+RT系统的验证集平均日收益再取平均。
这样做是为了防止RT规则只适合某一种DA模型。测试集不参与选择。

## 验证集结果

| RT seed | 执行方式 | Transformer DA | XGBoost DA | 选择分数 |
|---:|---|---:|---:|---:|
| 2 | `rt_only` | 157.37 | 137.39 | **147.38** |
| 0 | `rt_only` | 151.37 | 131.38 | 141.37 |
| 1 | `rt_only` | 129.61 | 109.62 | 119.62 |
| 0/1/2 | `follow_da` | 69.27 | 72.16 | 70.72 |
| 2 | `track_adjust` | 38.09 | 27.38 | 32.73 |
| 0 | `track_adjust` | 37.91 | 22.49 | 30.20 |
| 1 | `track_adjust` | 25.23 | 14.18 | 19.70 |

单位都是“每天平均收益”。`follow_da`根本不使用RT预测，所以3个seed结果相同。

## 为什么简单DA-aware规则没赢

以RT seed 2 + Transformer DA为例：

| 方式 | DA腿 | RT偏差腿 | 退化成本 | 日净收益 |
|---|---:|---:|---:|---:|
| `rt_only` | 106.45 | 88.85 | 37.93 | **157.37** |
| `follow_da` | 106.45 | 0.00 | 37.18 | 69.27 |
| `track_adjust` | 106.45 | -63.63 | 4.73 | 38.09 |

`track_adjust`像是开车时只看眼前10米：它会算这一小时的预计收益，
但不会算这次充放电对后面SOC和后续偏差结算的影响。它虽然减少了退化成本，
却造成了更大的负RT偏差收益。

因此结论是：当前两个简单DA-aware固定规则都没有改善收益。
这不等于“DA信息对RT永远没用”；它只说明协调需要考虑多小时SOC的训练或优化方法。

## 可复现证据

- 代码：`scripts/decision_aware/select_rt_coordination.py`
- 完整JSON：`data/results/rt_coordination_validation_selection.json`
- 固定规则实现：`src/decision_aware/backtest_dual.py`
- 自动测试：`tests/test_decision_aware_backtest_dual.py`
