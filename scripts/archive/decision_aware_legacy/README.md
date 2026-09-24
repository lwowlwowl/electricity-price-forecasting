# Decision-aware 历史实验入口

本目录保存 v1/v2 先行版和 v3 过渡实验脚本，仅用于复现实验，不再作为当前主线。

- `train_pilot.py`：v1/v2 先行版训练。
- `train_pilot_v3.py`：v3 DA+RT 过渡版训练。
- `compare_baselines.py`：与 v1/v2 数据及策略配套的旧基线对比。
- `eval_v3_da_oracle.py`：修复旧 Oracle 口径后重评 v3 checkpoint。

当前训练入口仍为 `scripts/decision_aware/train_formal.py`。当前基线比较脚本为
`scripts/decision_aware/compare_baselines_formal.py`，但它仍偏向 v4 口径，使用 v7/v8
结果前需要补齐 `dual_split` 与偏差罚金评估逻辑。
