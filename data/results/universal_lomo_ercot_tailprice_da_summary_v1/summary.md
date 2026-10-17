# ERCOT无ID DA尾部价格加权：单因素试验

> 只使用2023 inner开发口径；2024不参与选择，2025未读取。

| 指标 | 当前DA | 尾部加权DA |
|---|---:|---:|
| DA全时段MAE | 37.2707 | 37.0903 |
| DA普通时段MAE | 15.5962 | 15.5837 |
| DA最高1% MAE | 1329.1531 | 1320.2121 |
| DA尖峰召回率 | 0.0000 | 0.0000 |
| DA负价召回率 | 0.2222 | 0.2749 |

## 完整收益

当前系统：377.84 USD/节点·日；尾部加权DA系统：386.35。
配对差：+8.51，7日块95%区间[-19.05, 35.31]。

## 预注册门槛

- [x] highest_1pct_da_mae_improves
- [x] ordinary_da_mae_within_2pct
- [x] negative_price_recall_not_worse
- [x] mean_daily_profit_improves

只有四项全部通过，才扩展到RT-at-DA或其他LOMO折；否则作为负结果停止。
