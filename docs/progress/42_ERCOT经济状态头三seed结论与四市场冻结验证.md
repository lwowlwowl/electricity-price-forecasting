# ERCOT经济状态头三seed结论与四市场冻结验证

> **结果适用范围修订**：本文件第1—5节的 ERCOT 三seed 经济状态结果使用已知实体身份条件化基础模型，
> 只证明 ERCOT 已知节点上的可复现领先。第6节的四市场身份条件化基础训练也不等于新市场零样本泛化。
> 无 ID 通用模型的正式口径和后续留一市场实验见第46号文档。

> 日期：2026-10-11  
> 当前实验数据边界：2020—2022训练，2023内部验证；本轮未读取2024或2025。  
> 目标：在不引入XGBoost、不搜索超参数或损失权重的前提下，提高自有DA、RT-at-DA、滚动RT系统的完整结算收益，并冻结同一结构进入四市场验证。

## 1. 本轮唯一晋级的结构改变

显式价差回归把价差幅度压缩到真实值的约五分之一；物理稀缺残差虽然改善MAE，完整收益最高仅
`264.49 USD/节点·日`。因此不再调回归损失权重，而把RT-at-DA的决策相关部分改写为三类经济状态：

```text
0：|DA-RT|不超过电池盈亏平衡价差，保持冻结基础模型方向
1：DA-RT > κ/η
2：RT-DA > κ/η
```

- `κ/η = 5.7/0.95 = 6.0 USD/MWh`，直接来自同一电池合同，不是ERCOT验证集阈值；
- 冻结pre-encoder RT-at-DA基础模型和上游自有DA模型；
- 只训练一个输入为冻结表征与标准化基础价差的三分类线性头，约390个参数；
- 经济状态只允许改变价差方向，绝对价差幅度仍由冻结的自有基础模型提供；
- 三分类使用普通交叉熵，没有类别权重、盈利权重或阈值搜索；
- XGBoost不参与训练、标签、门控或集成，只在最后作强基线。

该结构对应“先判断是否跨过可交易经济边界，再保留基础价格幅度”。它直接修复此前显式价差模型的
幅度收缩问题，也比对所有小时统一优化MAE更贴近DA锁仓方向。

## 2. 三个随机种子的预测稳定性

三个seed都从同一冻结自有模型初始化，独立训练线性经济状态头。checkpoint只按2023节点宏平均
三分类准确率选择，三者都自然落在epoch 20：

- seed 0：`62.3684%`，RT-at-DA MAE `45.13`；
- seed 1：`62.4036%`，RT-at-DA MAE `45.05`；
- seed 2：`62.3860%`，RT-at-DA MAE `44.88`。

三个seed的分类准确率最大差异不到`0.04`个百分点，说明训练结果对随机初始化/批次顺序不敏感。
新增的混淆矩阵诊断按市场分别记录中性、DA高、RT高三类，后续四市场训练可检查是否由单一多数类
支配；该诊断不参与选模。

## 3. 完整连续MPC收益

固定DA、滚动RT、连续SOC、偏差罚金和退化成本后，只替换RT-at-DA经济状态头：

- seed 0：`296.99 USD/节点·日`；
- seed 1：`298.53`；
- seed 2：`300.30`；
- 三seed无权重方向多数票：`298.53`；
- 稳健XGBoost三任务：`292.29`。

三seed平均为`298.61`，比XGBoost高`6.32`；三个独立seed的点估计全部超过基线。为避免按最终收益
挑随机种子，当前冻结候选使用无权重多数票，而不是收益最高的seed 2。

多数票相对XGBoost：

- 日均净收益差：`+6.24 USD/节点·日`；
- 7日块bootstrap 95%区间：`[-9.35, 24.98]`，仍跨0；
- 最差单日：`-139.76`，好于XGBoost的`-206.07`；
- 最差节点均值：`278.11`，好于XGBoost的`244.44`；
- 偏差罚金：`3.02`，低于XGBoost的`9.87`；
- 正收益日比例：`72.65%`，低于XGBoost的`79.28%`；
- 去掉最好5日后：`209.47`，略低于XGBoost的`210.51`；
- 配对日差中位数：`-1.01`，改善节点为`10/15`。

因此当前准确结论是：**自有三模型首次获得跨三个seed可复现的2023点估计领先，但统计区间与去极值
结果尚不足以证明稳健显著胜出。** 不能把`+6.24`写成已经最终击败基线。

## 4. 不确定性弃权消融

预先定义了一个不含参数的规则：三个seed方向完全一致才允许改变冻结基础方向，否则回退基础模型。
三个方向在`97.13%`的小时上一致，真正改变基础方向的比例为`13.51%`。完整结算为`296.99`，恰好
退化为seed 0的最终方向，低于多数票`298.53`。该规则没有新增价值，直接淘汰；不继续搜索一致率、
置信度或概率门槛。

## 5. 文献核对与适用边界

- *Forecasting Electricity Prices With Decision-Focused Learning for Storage Optimization*（EEM 2025，
  DOI `10.1109/EEM64765.2025.11050194`）支持以储能下游价值而非单一MAE评价电价预测；
- *Multi-Task Decision-Oriented Electricity Price Prediction for Virtual Energy Storage Arbitrage*（PESGM 2025，
  DOI `10.1109/PESGM52009.2025.11225718`）支持把决策相关任务显式纳入多任务预测；
- *Forecasting the Occurrence of Electricity Price Spikes: A Statistical-Economic Investigation Study*（2024，
  DOI `10.3390/forecast6010007`）表明尖峰分类需同时报告统计与经济表现，并保留可解释规则。

这些工作支持“经济状态分类+完整结算”的研究动机，但不证明本项目结构必然跨市场有效；四市场结果
必须独立给出，不能仅凭ERCOT引用文献外推。

## 6. 冻结的四市场验证

已创建CAISO、ERCOT、NYISO、PJM共64节点的同构配置并通过本地CUDA烟雾检查：

- DA仍使用MAE；
- RT-at-DA基础价格仍使用伪Huber；
- 滚动RT仍使用MAE；
- 节点与市场身份统一在编码前进入共享编码器；
- 每市场等权采样，checkpoint按市场宏平均选择；
- 四个ISO使用各自冻结的本地DA起报时刻；
- 模型宽度、层数、学习率、训练轮数、损失和电池阈值均不针对市场修改。

四市场基础三模型现已全部完成正式训练，云端checkpoint与日志已拉回本地并完成SHA256核验：

- DA市场等权MAE：`15.87`；CAISO/ERCOT/NYISO/PJM为`12.20/38.12/7.16/6.01`；
- RT-at-DA市场等权MAE：`19.69`；四市场为`17.04/41.73/10.88/9.12`；
- 滚动RT市场等权MAE：`13.25`；四市场为`10.14/29.48/6.97/6.43`。

该结果证明同一结构可以完成64节点联合训练，但不能证明跨市场盈利领先：三个外部市场尚未完成正式
强基线和各自结算规则审计，经济状态头也尚未在四市场联合骨干上正式训练。下一步只在冻结的多市场
RT-at-DA基础模型上复现同一三分类线性头；外部市场合同冻结前只报告预测、混淆矩阵和价差方向，
不得复制ERCOT偏差罚金规则后宣称真实跨市场盈利。最新快速结论统一见
`43_当前结论与实验总账.md`。

## 7. 机器可读产物

- `data/results/global_economic_regime_multiseed_continuous_mpc_v1.json`
- `data/results/global_economic_regime_vote_paired_robustness_v1.json`
- `data/results/global_economic_regime_unanimous_continuous_mpc_v1.json`
- `data/results/global_economic_regime_multiseed_predictions_v1/`
- `data/checkpoints/global_ercot_economic_regime_rt_at_da_v1/`
- `configs/decision_aware/global_multimarket_entitypre_da_mae_v1.yaml`
- `configs/decision_aware/global_multimarket_entitypre_rt_at_da_pseudohuber_v1.yaml`
- `configs/decision_aware/global_multimarket_entitypre_rt_mae_v1.yaml`
- `data/checkpoints/global_multimarket_entitypre_da_mae_v1/`
- `data/checkpoints/global_multimarket_entitypre_rt_at_da_pseudohuber_v1/`
- `data/checkpoints/global_multimarket_entitypre_rt_mae_v1/`
- `data/results/cloud_global_multimarket_entitypre_remaining_v1/`

