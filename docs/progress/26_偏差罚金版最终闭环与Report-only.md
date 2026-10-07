# 偏差罚金版最终闭环与 Report-only

> 状态：**已闭环**，2026-10-07。
>
> 本文只汇总已经冻结并实际运行的结果。旧无罚金V1/V2文件继续保留为历史对照，
> 不与当前主合同混排，也没有被覆盖。

## 1. 这次补齐了什么

偏差罚金版现在已经追上并超过旧无罚金版的实验节点：

| 环节 | 当前状态 | 证据 |
|---|---|---|
| 固定结算合同 | 完成 | `κ=5.7 USD/MWh`，偏差罚金开启 |
| 训练/验证切分 | 完成 | 1808个训练日、182个验证日 |
| 连续SOC与期末SOC | 完成 | 跨连续日期继承；期末只报MWh，不折现 |
| 纯Huber起点 | 完成 | DA九个候选按新合同重训；RT-at-DA/RT纯预测权重无需因成本重训 |
| 政策与checkpoint重选 | 完成 | 安全策略冻结；active三路在972个组合中重选 |
| 统一横向基线 | 完成 | Seasonal、XGBoost、GRU、Huber、历史V2权重 |
| Oracle/Regret/PCR | 完成 | 182日连续SOC偏差罚金MILP Oracle |
| active联合微调 | 完成 | 同一RTX 4090D环境完成seed 0/1/2 |
| 风险与配对区间 | 完成 | 正收益率、尾部、回撤、7日块Bootstrap |
| 最终留出披露 | 完成 | 冻结后一次读取150日report-only |

因此，当前不再存在“偏差罚金版还少一次联合训练或少一个seed”的欠账。

## 2. 冻结合同

主实验共同使用：

```text
市场/节点：ERCOT / LZ_LCRA
运行与退化成本：κ = 5.7 USD/MWh
偏差罚金：2 × |RT价格| × max(|u_RT_actual-u_DA|-0.03|u_DA|, 0)
电池：1 MW / 4 MWh，效率0.95，SOC范围0.4—3.6 MWh
SOC：连续日期继承；只在真实数据缺口形成的新段重置为2 MWh
期末SOC：只报告MWh，不折算货币
验证：2025-07-01至2025-12-30，共182日
report-only：2026-01-01至2026-05-31，共150日
```

安全横向基线统一使用`da_only + follow_da`、DA K3/3、RT K0/0、阈值6。
这个策略让实际RT动作跟随DA，所以偏差为0、实际偏差罚金也为0；但罚金代码和开关确实开启，
`κ=5.7`退化成本也确实扣除。这是可执行的安全策略结果，不是“忘记计算罚金”。

三路active系统统一使用`spread + plan_track_topk`、DA K4/4、RT K1/1、阈值25。
它允许滚动RT改变动作，因此会实际产生偏差罚金。

## 3. 冻结验证结果

模型、策略和选模规则在查看本次report-only前已经冻结。验证集总表如下：

| 完整系统 | 策略类别 | 验证日均净收益（USD/日） |
|---|---|---:|
| Seasonal | 安全 | 95.5844 |
| **XGBoost（三seed）** | **安全** | **97.3161 ± 0.6523** |
| GRU（三seed） | 安全 | 60.4067 ± 8.8984 |
| Transformer-Huber | 安全 | 83.7028 |
| 历史Transformer-V2权重（三seed） | 安全 | 64.3088 ± 0.0133 |
| Transformer-V3 active epoch0 | 三路active | 49.6733 |
| Transformer-V3 active联合微调（三seed） | 三路active | 50.4550 ± 0.3607 |

按预先规定的主要指标“验证集完整双结算日均净收益”，冻结冠军是 **XGBoost**。
这项选择在读取report-only之后不改变。

## 4. active联合训练三seed

| Seed | 最佳Epoch | 验证收益 | 相对epoch0 | 验证配对95%区间 | report-only收益 |
|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 50.6798 | +1.0065 | `[-3.2683, 5.9704]` | -75.2029 |
| 1 | 2 | 50.6462 | +0.9729 | `[-4.3334, 6.0710]` | -75.5590 |
| 2 | 4 | 50.0389 | +0.3656 | `[-0.4204, 1.2116]` | -67.6512 |
| **三seed** | — | **50.4550 ± 0.3607** | **+0.7817 ± 0.3607** | **逐seed均跨0** | **-72.8044 ± 4.4663** |

active epoch0在report-only为`-67.8049 USD/日`，联合微调三seed均值比它低
`4.9995 USD/日`。所以验证期的小幅均值改善没有形成样本外改善，active联合模型不晋级。

active策略在report-only的偏差罚金约为每日至少`98.98 USD`量级，是净收益转负的主要原因。
这与此前解析结论一致：3%容忍带远小于接近1 MWh的离散RT动作，RT-active容易受到高额罚金。

## 5. 最终report-only披露

| 完整系统 | 验证日均净收益 | report-only日均净收益 |
|---|---:|---:|
| **Seasonal** | 95.5844 | **62.8872** |
| XGBoost（三seed） | **97.3161 ± 0.6523** | 62.4429 ± 3.8625 |
| GRU（三seed） | 60.4067 ± 8.8984 | -6.5255 ± 46.2039 |
| Transformer-Huber | 83.7028 | 47.8037 |
| 历史Transformer-V2权重（三seed） | 64.3088 ± 0.0133 | 21.5198 ± 0.5609 |
| Transformer-V3 active epoch0 | 49.6733 | -67.8049 |
| Transformer-V3 active联合微调（三seed） | 50.4550 ± 0.3607 | -72.8044 ± 4.4663 |

report-only上Seasonal数值略高于XGBoost三seed均值，差约`0.44 USD/日`；但它不参与
回选，所以正式“按验证选择的系统”仍是XGBoost。这个接近反转说明单节点、单时间窗的模型排名
不够稳健，也正是下一阶段必须做多节点、多市场和多时间窗外部验证的原因。

还必须保留一个限制：2026年上半年曾被旧无罚金实验披露过。它没有参与本次偏差罚金候选的
训练、调参或选模，但不能称为整个项目从未查看过的全新测试集。投稿级最终证据仍应使用更晚的
未查看时间段，或预先冻结的其他节点/市场。

## 6. 最终结论

1. **实验闭环已经完成。** 偏差罚金版不再落后于旧无罚金版。
2. **当前验证集数值冠军是XGBoost安全系统，不是联合Transformer。**
3. **三路联合收益微调只在验证期相对active起点小幅改善，且区间跨0；在report-only反而下降。**
4. **问题不是电脑太小。** 三seed在4090D正常完成，三路梯度有限非零；主要矛盾是策略动作粒度与偏差罚金合同。
5. **不把三个Transformer合并成一个大模型。** 三个预测器仍相互独立，只共享最终收益评价。
6. **本节点不再继续看report-only调参。** 下一步转向现有四市场、多个节点的外部验证与泛化实验。

## 7. 可复核文件

| 内容 | 路径 |
|---|---|
| 安全策略冻结配置 | `configs/decision_aware/joint_decision_aware_v3_penalty_safe_kappa57.yaml` |
| active联合训练冻结配置 | `configs/decision_aware/joint_decision_aware_v3_penalty_active_kappa57.yaml` |
| validation统一总表 | `data/results/joint_system_baseline_comparison_validation_penalty_kappa57_final.json` |
| report-only统一总表 | `data/results/joint_system_baseline_comparison_report_only_penalty_kappa57_final.json` |
| active epoch0留出结果 | `data/results/joint_system_active_epoch0_report_only_penalty_kappa57_final.json` |
| active三seed留出结果 | `data/results/joint_decision_aware_v3_penalty_active_kappa57_seed{0,1,2}_report_only_final.json` |
| 最终机器可读闭环摘要 | `data/results/penalty_experiment_final_closure_kappa57.json` |
| 三seed训练结果 | `data/results/joint_decision_aware_v3_penalty_active_kappa57_seed{0,1,2}.json` |
| 三seed最佳checkpoint | `data/checkpoints/joint_decision_aware_v3_penalty_active_kappa57_seed{0,1,2}/best_validation_revenue.pt` |

生成统一总表的脚本现在显式区分`validation`与`report_only`；读取report-only必须同时传入
`--confirm-report-only`。GRU早停始终只使用validation，report-only只做最终预测和结算。

## 8. 下一阶段

不再在LZ_LCRA的2026留出区间继续调模型。下一阶段应预先冻结协议，然后：

1. 在ERCOT多个负荷区/交易枢纽/节点复评XGBoost、Transformer-Huber与active系统；
2. 扩展到CAISO、NYISO、PJM，分别做本地训练、留节点和留市场实验；
3. 报告跨节点/市场均值、最差节点、配对区间与失败案例，而不是只挑最好节点；
4. 若继续研究RT-active，先改变策略动作粒度或显式优化罚金容忍带，再训练模型；不能只加epoch。
