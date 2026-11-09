# TimesFM3完整三腿零样本结算

## 1. 实验目的与边界

文档11只替换DA预测，TimesFM3仍共用decision-aware v4的RT-at-DA与rolling RT，因而不是完整系统。本实验继续回答：若让同一个冻结TimesFM3同时产生DA、RT-at-DA和rolling RT三条预测腿，它能否直接进入同一套15分钟储能LP/MPC结算？

这不是老师榜单中TimesFM3 DA模型的原始结果，而是本项目新增的严格零样本扩展。奥地利数据只用于最终评价，没有参与训练、微调、特征选择、阈值选择或超参数选择；看到结果后也没有调参。

## 2. 三条腿如何产生

- **DA**：复用已冻结的TimesFM3 mainU11-proxy6预测，DA MAE为`18.841 EUR/MWh`；
- **RT-at-DA**：在奥地利本地时间D-1 12:00截断信息，只读取门控前4,096个15分钟RT点，从门控时刻预测至交付日结束，再截取交付日的92/96/100个原生15分钟点；
- **rolling RT**：每个UTC整点重新起报，只读取起报前4,096个15分钟RT点，预测未来16个15分钟点，即4小时；
- **协变量**：三条腿只使用老师mainU11家族中本地确实存在的6个代理量，且目标窗口全部由恰好7天前的负荷、风电、光伏、风速和辐照度构造；未补造缺失的4条MaxBex与TTF，也没有增加老师清单外的组件；
- **结算**：沿用相同电池、退化成本、偏差容忍区间、偏差罚金和原生15分钟LP/MPC。

这里的“完整三腿”表示同一基础模型覆盖三类预测任务，不表示在奥地利训练过；它仍是冻结的零样本系统。

## 3. 预测与经济结果

| 系统/预测腿 | 指标 | 结果 |
| --- | --- | ---: |
| TimesFM3 DA | MAE | 18.841 EUR/MWh |
| TimesFM3 RT-at-DA | MAE | 54.968 EUR/MWh |
| TimesFM3 rolling RT首个可用预测 | MAE | 44.997 EUR/MWh |
| TimesFM3完整三腿 | 平均净收益 | **-143.798 EUR/日** |
| TimesFM3完整三腿 | 中位净收益 | 6.907 EUR/日 |
| TimesFM3完整三腿 | 正收益日比例 | 52.05% |
| TimesFM3完整三腿 | SOC违规 | 0 |
| decision-aware v4完整三腿 | 平均净收益 | **84.509 EUR/日** |

TimesFM3完整三腿比v4低`228.307 EUR/日`。文档11中“TimesFM3只替换DA腿、后两腿固定为v4”时为`-22.067 EUR/日`；换成TimesFM3自己的后两腿后进一步降至`-143.798 EUR/日`，说明主要新增损失来自RT预测和偏差控制。

## 4. 收益分解与解释

TimesFM3完整三腿的逐日平均分解为：

| 分量 | EUR/日 |
| --- | ---: |
| DA交易腿 | +196.529 |
| RT偏差腿 | -36.712 |
| 电池退化成本 | -27.912 |
| 偏差罚金 | -275.704 |
| 净收益 | **-143.798** |

因此，负结果不是DA预测完全无效，而是冻结TimesFM3对RT价格的直接外推误差过大，导致偏差罚金吞掉DA腿收益。即使删除净收益最差的5天，剩余360天平均仍为`-79.717 EUR/日`；不能把它解释为单一异常日造成。

分组结果同样一致：负价日`-46.165`、尖峰日`-401.520`、普通日`-24.250 EUR/日`。尖峰日损失最大，说明RT尾部场景需要专门的门控安全训练或decision-aware目标，不能只依靠DA精度较好的基础模型零样本迁移。

## 5. 当前结论

本实验否定了“DA模型直接复制到另外两条腿即可形成强完整系统”这一假设。TimesFM3的DA MAE明显优于v4，但完整收益远低于v4；v4的优势来自DA、RT-at-DA、rolling RT与结算目标的联合协调，而不仅是某一条价格预测腿。

该负结果不应作为TimesFM3原论文或老师榜单模型的结论，因为老师只给出了DA配置；它只说明本项目采用上述严格因果、零样本扩展时，RT两腿尚不可用。

## 6. 证据

- 执行脚本：`scripts/decision_aware/evaluate_austria_timesfm3_mainu11_full_triplet.py`；
- 机器汇总：`data/results/austria_timesfm3_mainu11_full_triplet_v1/summary.json`；
- 完整逐日结算：`data/results/austria_timesfm3_mainu11_full_triplet_v1/report.json`；
- 三腿预测缓存：`data/results/austria_timesfm3_mainu11_full_triplet_v1/predictions.npz`；
- 时序防泄漏测试：`tests/test_austria_timesfm3_full_triplet.py`。
