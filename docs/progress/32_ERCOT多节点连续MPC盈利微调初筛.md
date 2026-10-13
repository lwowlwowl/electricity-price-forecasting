# ERCOT多节点连续MPC盈利微调初筛

> **通用性边界**：本文件使用已知实体身份条件，结论不外推到训练时未见市场；详见第46号文档。

> 本页是2024外层失败后的内部研发记录。训练只使用2020—2022，评价只使用2023；
> 2024不再用于调参，2025完全未读取。

## 1. 为什么做这一轮

冻结DA校正候选在严格2024外层验证中比稳健XGBoost低`4.20 USD/节点·日`，且95%区间
`[-8.01, -1.23]`全部低于0。因此不能继续靠调融合权重修补，也不能进入2025。

仓库原来已经有连续MPC零阶收益梯度的单节点/单批次smoke，但没有把它真正用于15节点训练。
本轮补齐了最小正式训练器：三个模型仍完全独立，只通过完整收益目标共同产生梯度。

公开方法依据与本项目方向一致：

- [Perturbed Decision-Focused Learning for Modeling Strategic Energy Storage](https://arxiv.org/abs/2406.17085)
  使用扰动使储能优化层可用于决策导向训练，并强调预测损失与决策损失的混合目标。
- [Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-focused Approach](https://arxiv.org/abs/2305.00362)
  直接以储能套利决策误差/遗憾构造训练信号，而不是只最小化价格误差。

## 2. 实际实现

新增`train_global_joint_continuous_zo.py`，流程如下：

1. 从稳定Global DA、RT-at-DA、RT三个独立checkpoint开始。
2. 每个节点抽取一个连续2日episode，共15个batch；不跨节点、不跨真实时间缺口。
3. 在完整连续DA-LP、锁定DA计划、滚动RT-MPC、偏差罚金和退化成本下结算真实净收益。
4. 对DA/RT-at-DA价差和滚动RT预测做成对零阶扰动，得到收益方向。
5. 收益代理与价格Huber锚点组成混合损失，只微调三个模型各自的价格头。
6. 每个batch做梯度有限性检查与`1.0`裁剪；三个模型仍没有共享参数。

本地1节点smoke成功后，AutoDL RTX 4090D完成15节点正式小筛选。

## 3. 两个冻结小候选

除收益权重外完全相同：

- `globalzo_b001`：收益权重`0.001`。
- `globalzo_b003`：收益权重`0.003`。

两轮所有15个batch的收益梯度均为有限值，训练、保存、三任务预测导出和完整收益回测均完成。

## 4. 2023内部结果

- 原稳定Global完整系统：约`192.83 USD/节点·日`。
- `globalzo_b001`完整三Global系统：`194.84`，提高约`2.01`。
- `globalzo_b003`完整三Global系统：`194.82`，提高约`1.99`。
- 稳健XGBoost三任务基线：`292.29`。
- `b001`只替换DA、其余使用XGBoost：`141.75`。
- `b003`只替换DA、其余使用XGBoost：`141.75`。

与XGBoost相比：

- `b001`完整Global差`-97.45`，7日块95%区间`[-151.10, -54.64]`。
- `b003`完整Global差`-97.47`，7日块95%区间`[-151.13, -54.68]`。

## 5. 结论

零阶收益梯度不是无效的：一次很小的价格头更新让完整Global收益提高约`2`。但收益权重从
`0.001`加到`0.003`没有继续改善，且与XGBoost仍有约`97`的巨大差距。因此：

- 两个候选都不晋级，不补seed，也不进入2024/2025。
- 不再继续放大同一路线的收益权重，避免盲目烧GPU。
- 当前主要瓶颈仍是Global基础预测/价差信号弱，而不是“少做一次收益微调”。
- 下一条更合理的主线是以稳健XGBoost作为主预测，训练决策相关的残差或门控模型；
  必须先用嵌套的2020—2022/2023内部证据证明稳定，再讨论任何新的外层确认。

## 6. 保存与清理

保留：

- `data/results/global_joint_continuous_zo_b001_train_v1.json`
- `data/results/global_joint_continuous_zo_b001_evaluation_v1.json`
- `data/results/global_joint_continuous_zo_b003_train_v1.json`
- `data/results/global_joint_continuous_zo_b003_evaluation_v1.json`
- `data/results/cloud_global_joint_zo_v1/`

未晋级权重和可复现预测缓存已从云端删除，避免与正式checkpoint混淆。
