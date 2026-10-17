# TimesFM-2.5源市场监督LoRA四折结果

> 完成时间：2026-10-16 19:33（北京时间）  
> 协议：三个源市场2020—2022监督训练，源2023内部验证，留出市场2023零样本测试；不读取节点/市场ID或留出市场标签。  
> 模型：`google/timesfm-2.5-200m-transformers`，rank-4 all-linear LoRA；仅使用168小时目标价格历史。

## 预测MAE

| 留出市场 | 任务 | TimesFM-2.5 LoRA | 当前Transformer | 胜者 |
|---|---|---:|---:|---|
| CAISO | DA | 21.35 | **18.20** | Transformer |
| CAISO | RT-at-DA | 21.14 | **21.03** | Transformer |
| CAISO | 滚动RT | 9.81 | **9.75** | Transformer |
| ERCOT | DA | **36.14** | 37.27 | TimesFM-2.5 LoRA |
| ERCOT | RT-at-DA | 35.42 | **34.80** | Transformer |
| ERCOT | 滚动RT | 29.26 | **28.86** | Transformer |
| NYISO | DA | 9.77 | **8.55** | Transformer |
| NYISO | RT-at-DA | **10.53** | 10.73 | TimesFM-2.5 LoRA |
| NYISO | 滚动RT | **7.01** | 7.13 | TimesFM-2.5 LoRA |
| PJM | DA | 6.57 | **6.20** | Transformer |
| PJM | RT-at-DA | 9.12 | **8.96** | Transformer |
| PJM | 滚动RT | 6.96 | **6.57** | Transformer |

当前Transformer胜9/12项，TimesFM-2.5 LoRA胜3/12项。这是“当前多模态Transformer”与
“只读取目标价格历史的TimesFM-2.5源市场LoRA”的系统对比，不能归因为单一结构差异。

## ERCOT同合同盈利

| 系统 | 日均净收益（USD/节点·日） | 7日块95%区间 | 排名 |
|---|---:|---:|---:|
| 当前Transformer | **377.84** | [151.13, 657.71] | **1** |
| 逐节点本地XGBoost | **292.29** | [146.00, 469.58] | **2** |
| TimesFM-2.5 LoRA | **262.79** | [111.06, 455.60] | **3** |
| TimesFM-3 frozen | **-91.14** | [-174.13, -31.21] | **4** |

TimesFM-2.5 LoRA相对当前Transformer少`115.05 USD/节点·日`，配对7日块95%区间
`[-225.13, -28.04]`，明确落后；相对逐节点本地XGBoost少`29.50`，区间
`[-93.18, 29.98]`跨0，不能证明两者有稳定收益差异。

TimesFM-2.5 LoRA的收益分解为：DA腿`307.99`、RT修正腿`-3.66`、退化成本`20.64`、
偏差罚金`20.89 USD/节点·日`。它显著好于frozen TimesFM-3，但仍未超过当前Transformer。

## 完整性

- 云端正式任务：12/12完成；ERCOT LP/MPC双结算完成。
- 归档：`universal_lomo_timesfm25_lora_completed_20261016.tar.gz`，SHA-256
  `e72e6bd365c7db24a289e9de5f9d7fcaea70e0486a6090d383a563cbe9641248`。
- manifest：98/98文件在本地逐项校验通过。
- 2025未读取，未用2024回头调参。
