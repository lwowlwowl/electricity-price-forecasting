# 四市场逐节点本地cutoff-safe LEAR增量汇总

> LEAR使用目标节点2020—2022历史并每天按截止时点滚动重估，只用于DA和RT-at-DA；它是目标市场监督强基线，不是同数据零样本架构对照。未读取2025。

| 市场 | 任务 | 本地LEAR MAE | 无ID Transformer MAE | pooled LOMO XGBoost MAE | 当前可用最低MAE | LEAR墙钟 |
|---|---|---:|---:|---:|---|---:|
| CAISO | da | **10.88** | 18.20 | 19.95 | 本地LEAR | 25.6分 |
| CAISO | rt_at_da | **14.84** | 21.03 | 23.72 | 本地LEAR | 17.3分 |
| ERCOT | da | **34.48** | 37.27 | 35.90 | 本地LEAR | 62.0分 |
| ERCOT | rt_at_da | **34.39** | 34.80 | 35.99 | 本地LEAR | 55.4分 |
| NYISO | da | **6.98** | 8.55 | 10.83 | 本地LEAR | 81.5分 |
| NYISO | rt_at_da | **10.64** | 10.73 | 14.74 | 本地LEAR | 67.2分 |
| PJM | da | -（缓存5/35节点） | **6.20** | 7.25 | Transformer | - |
| PJM | rt_at_da | - | **8.96** | 10.21 | Transformer | - |

当前完成 **6/8** 个市场×任务；全部完成前不写总排名。
滚动RT不在LEAR定义内，保留lag-168、pooled XGBoost、TSFM和当前Transformer对照。
