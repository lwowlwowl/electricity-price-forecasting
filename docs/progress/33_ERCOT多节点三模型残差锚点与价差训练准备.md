# ERCOT多节点三模型残差锚点与价差训练准备

> **通用性边界**：本文件使用已知实体身份条件，只作 ERCOT 已知节点历史记录；详见第46号文档。

> 历史状态：准备记录，正式结果已由`34_ERCOT多节点三模型残差锚点与价差训练结论.md`覆盖；不要按本文件重复启动训练。

> 日期：2026-10-10  
> 阶段：2020—2022训练准备、2023内部诊断；2024不参与本轮调参，2025完全未读取。  
> 本页记录的是新候选的实现和训练前筛选，不是正式性能结论。

## 1. 为什么不继续优化XGBoost

最终模型必须保留三个独立预测任务：DA、DA截止时点的RT（RT-at-DA）和滚动RT。
稳健XGBoost只作为必须击败的基线，不进入新模型结构，也不给新模型提供预测输入。

现有稳定Global三模型在2023内部验证的完整收益约为`192.83 USD/节点·日`，低于稳健
XGBoost三任务的`292.29`。已有任务替换诊断说明DA是第一瓶颈、RT-at-DA第二、滚动RT第三。
同时，稳定Global的DA减RT-at-DA价差仍有明显压缩：重要价差的方向质量不足，极端幅度偏小。

## 2. 本轮结构改动

### 2.1 合法的168小时残差锚点

三个独立模型新增可选配置`global_residual_anchor: lag168`。启用后：

1. 数据集只取目标交付小时之前168小时的同一UTC小时价格；
2. 每个锚点必须位于当前样本的历史上下文中；
3. 目标时间与锚点时间必须精确相差168小时；
4. 模型价格头预测“相对历史锚点的标准化残差”，再加回锚点；
5. 价格头从接近零残差初始化，避免训练开始时随机破坏季节起点。

DA使用历史DA价作为锚点；RT-at-DA和滚动RT使用历史RT价。该结构不读取未来价格、未来
负荷或未来风光，也不依赖XGBoost。默认配置仍为`none`，所以旧checkpoint和旧实验不受影响。

### 2.2 显式价差幅度和方向损失

联合微调入口新增两项只在训练batch上计算的辅助损失：

- 价差幅度：对`(预测DA−预测RT-at-DA)−(真实DA−真实RT)`使用逐节点标准化伪Huber损失；
- 价差方向：只在绝对真实价差不小于`6 USD/MWh`时监督正负方向，并限制尖峰权重上限。

它们与原价格锚定损失、完整连续MPC零阶盈利信号共同使用。三个模型仍是三套独立参数；
联合只表示同一盈利目标能同时给三套模型产生训练信号。

## 3. 2023全年无训练诊断

先只评价168小时锚点本身，避免为明显无效的起点烧GPU。15节点2023内部验证结果：

- DA锚点MAE：`47.99`；
- RT-at-DA锚点MAE：`50.60`；
- 滚动RT锚点MAE：`50.72`；
- 绝对真实价差至少`6 USD/MWh`时，DA−RT-at-DA方向准确率：`60.49%`。

稳定Global三任务MAE约为`41.59 / 43.66 / 31.40`，其已记录价差方向准确率约`56.58%`。
因此锚点本身不是更强预测器，但它在关键价差方向上提供了互补起点。冻结决定为：

- DA与RT-at-DA进入正式残差学习候选；
- 滚动RT锚点明显弱于现有Global RT，本轮默认停止该支线，不浪费GPU；
- 最初完整三元组继续使用现有稳定Global RT，随后再用联合盈利训练更新它。

机器可读诊断：`data/results/global_lag168_anchor_diagnosis_v1.json`。

## 4. 已完成的运行验证

- DA、RT-at-DA、滚动RT分别完成两节点、单轮CUDA训练冒烟；损失、梯度、checkpoint和
  candidate manifest均正常。
- 新的“历史锚点 + 价差辅助损失 + 连续MPC零阶盈利”链路完成两节点端到端冒烟；两个
  batch的价差与RT零阶梯度均为有限非零值。
- 全仓回归为`193 passed`。所有带`smoke`的数值只证明流程能运行，不能排名。

## 5. 下一步与晋级规则

云端第一轮只训练DA和RT-at-DA两个seed-0候选，预计RTX 4090D约`30—45分钟`。滚动RT锚点
默认跳过。训练完成后：

1. 导出保留的DA和RT-at-DA候选，与冻结稳定Global RT组成三元组；
2. 在2023完整连续MPC、偏差罚金和退化成本下选候选，而不是只按MAE；
3. 只有相对稳定Global有实质改善才进入联合盈利微调；
4. 只有2023内部证据稳定后才补seed 1/2；
5. 本轮不重新使用2024调参，2025继续锁定。

云端入口：`scripts/decision_aware/run_cloud_global_lag168_v1.sh`。脚本带源数据SHA256和CUDA
预检，支持中断续训；除非显式设置`TRAIN_RT_ANCHOR=1`，否则不会训练已被诊断淘汰的RT锚点。

## 6. 主要文件

- `src/decision_aware/joint_prediction.py`
- `scripts/decision_aware/diagnose_global_lag168_anchor.py`
- `scripts/decision_aware/train_global_joint_continuous_zo.py`
- `configs/decision_aware/global_ercot_lag168_da_mae_v1.yaml`
- `configs/decision_aware/global_ercot_lag168_rt_at_da_pseudohuber_v1.yaml`
- `configs/decision_aware/global_ercot_lag168_rt_mae_v1.yaml`
- `scripts/decision_aware/run_cloud_global_lag168_v1.sh`
- `tests/test_decision_aware_joint_prediction.py`
