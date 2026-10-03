# 储能期末 SOC 与机会价值：待读文献

## 为什么要读这些文献

当前项目比较 DA、RT-at-DA 和滚动 RT 模型时，主要指标是 DA+RT 双结算现金净收益。
但是，不同滚动 RT 模型在回测结束时可能留下不同数量的电量。例如，一个模型期末
剩余 0.5 MWh，另一个模型期末剩余 3.0 MWh。二者的现金收益不能完全代表它们处于
相同的最终状态。

阅读下面文献主要为了回答四个问题：

1. 有限回测结束时，剩余 SOC 是否需要处理；
2. 应该使用期末 SOC 约束、期末价值，还是额外 look-ahead；
3. 滚动优化怎样避免在每个短窗口末尾把电池人为放空；
4. 在 DA+RT 双结算中，SOC 的未来机会价值怎样影响实时决策。

这里的“期末价值”是**电池中剩余能量的继续使用价值**，不是电池设备报废时的残值。

## 当前项目的临时决定

当前阶段暂不把剩余 SOC 换算成美元，也不把它加入市场现金净收益。

每次验证和回测至少单独记录：

- 期初实际 SOC，单位 MWh；
- 期末实际 SOC，单位 MWh；
- `期末SOC - 期初SOC`，单位 MWh；
- 每个因数据缺口而重置的连续时间段，其起点和终点 SOC；
- DA 计划 SOC 与真实执行 SOC 分开记录；
- 市场现金净收益仍只包含 DA 腿、RT 偏差腿、退化成本和已启用的罚金。

当前报告应同时展示“现金净收益”和“期末 SOC”，但不把二者相加。若两个模型的现金
收益很接近而期末 SOC 差异较大，只能写“现金收益暂时更高”，不能直接写成“总体经济
价值一定更高”。

不换算成美元时，可以使用下面的简单解释规则：

- 若模型 A 的现金收益更高，期末 SOC 也不低于模型 B，A 可以视为同时占优；
- 若模型 A 的现金收益更高，但期末 SOC 更低，这是“多赚钱但少留电”的权衡，暂不判定
  总体优胜者；
- 若各模型期末 SOC 基本相同，则现金净收益可以更直接地用于比较。

这个临时方案的优点是简单、透明，不会为了估值而引入未经验证的美元/MWh参数。它的
限制是：仍然无法把 1 MWh 剩余电量与若干美元现金收益直接比较。因此，在正式论文结论
或模型最终部署前，仍需根据下面文献选择并预先固定一种终端处理方法，再做敏感性检查。

## 阅读顺序

### 1. Niet（2020）：先理解“末端效应”是什么

**文献**

Taco Niet. *Storage end effects: An evaluation of common storage modelling assumptions*.
Journal of Energy Storage, 27, 101050, 2020.
[论文页面与 DOI](https://doi.org/10.1016/j.est.2019.101050)

**为什么要读**

这是与当前问题最直接的一篇。它专门比较三类处理方法：要求储能在期末补回能量、增加
look-ahead，以及给期末储能赋值。它说明“完全不处理期末状态”会让模型在边界附近产生
不真实的放空行为，同时也说明没有一种终端处理方式在所有模型中都自动正确。

**阅读时重点看**

- 什么是 storage end effect；
- 三种处理方法各自会引入什么偏差；
- 为什么期末价值的取值会明显影响调度结果。

### 2. Sioshansi 等（2022）：理解为什么必须保留时间连续性

**文献**

Ramteen Sioshansi et al. *Energy-Storage Modeling: State-of-the-Art and Future Research
Directions*. IEEE Transactions on Power Systems, 37(2), 860–875, 2022.
[论文信息与 DOI](https://doi.org/10.1109/TPWRS.2021.3104768)

**为什么要读**

这是一篇较全面的储能建模综述。它解释了储能为什么不能像普通、互不相关的小时样本
那样随意拆开：本小时的 SOC 会限制下一小时，前一天的终点也会成为下一天的起点。
这直接关系到本项目是否可以随机打乱交付日，以及遇到 DST 或缺失日期时能否直接重置
SOC。

**阅读时重点看**

- chronology（时间顺序）为什么重要；
- 将多个运行时段彼此解耦会产生什么误差；
- 储能运行模型和长期规划模型对 SOC 的处理有何不同。

### 3. Xu、Korpås 与 Botterud（2020）：理解 SOC 的“机会价值”

**文献**

Bolun Xu, Magnus Korpås, Audun Botterud. *Operational Valuation of Energy Storage under
Multi-stage Price Uncertainties*. 59th IEEE Conference on Decision and Control, 55–60,
2020.
[IEEE 页面与 DOI](https://doi.org/10.1109/CDC42340.2020.9304081)

**为什么要读**

这篇文献从多阶段价格不确定性出发，研究电池中一单位剩余能量在未来可能带来的运行
价值。它能帮助区分“当前小时卖电得到的现金”和“把电留下来以后再使用的机会价值”。

**阅读时重点看**

- value function 与 SOC 的关系；
- 价格尖峰和负电价如何影响存电价值；
- 为什么机会价值通常不是简单等于最后一个小时的电价。

### 4. Zheng 等（2023）：看机会价值怎样进入实际套利决策

**文献**

Ningkun Zheng, Xiaoxiang Liu, Bolun Xu, Yuanyuan Shi. *Energy Storage Price Arbitrage via
Opportunity Value Function Prediction*. 2023 IEEE Power & Energy Society General Meeting.
DOI: 10.1109/PESGM52003.2023.10253102.
[开放版本](https://arxiv.org/abs/2211.07797)

**为什么要读**

这篇文献不只是讨论期末怎么记账，而是把当前收益和执行动作后 SOC 的未来机会价值
一起用于决策。它与本项目未来的 decision-aware 训练很相关，因为训练目标不能只奖励
当前窗口内的现金收益，否则短窗口模型可能倾向于过早放电。

**阅读时重点看**

- 单步收益后面为什么要接 `V(SOC)`；
- 历史最优机会价值如何生成；
- 机会价值预测与直接价格预测有什么区别。

### 5. Alghumayjan 等（2024）：联系 DA+RT 双结算主线

**文献**

Saud Alghumayjan, Jiajun Han, Ningkun Zheng, Ming Yi, Bolun Xu. *Energy storage arbitrage
in two-settlement markets: A transformer-based approach*. Electric Power Systems Research,
235, 110755, 2024.
[开放版本](https://arxiv.org/abs/2404.17683)；
[期刊 DOI](https://doi.org/10.1016/j.epsr.2024.110755)

**为什么要读**

这是当前项目双结算逻辑最直接的参考。论文把 DA 决策和顺序执行的 RT 决策联系起来，
并在 RT 套利中使用 SOC 机会价值。它有助于确认：`rt_only` 不代表 DA 失去结算作用，
同时实际 SOC 的跨时段价值仍需要由实时决策考虑。

**阅读时重点看**

- DA 与 RT 两阶段收益怎样分解；
- 为什么 RT 决策在该价格接受者假设下可以与 DA 头寸解耦；
- 实时套利模块如何使用 SOC 的机会价值。

### 6. Wang 等（2023）：查看“期末恢复初始 SOC”的另一种做法

**文献**

Luyu Wang, Houbo Xiong, Yunhui Shi, Chuangxin Guo. *Rolling Horizon Robust Real-Time
Economic Dispatch with Multi-Stage Dynamic Modeling*. Mathematics, 11(11), 2557, 2023.
[论文与 DOI](https://doi.org/10.3390/math11112557)

**为什么要读**

这篇滚动调度文献采用另一条路线：要求整个调度过程结束时 SOC 恢复到初始水平。它能
帮助比较“给剩余电量估值”和“强制终点相同”两类方法。后者容易解释和公平比较，但也
可能限制真实盈利，因此不能未经验证就直接加入当前模型。

**阅读时重点看**

- 最终 SOC 约束放在整个过程还是每个短窗口；
- 滚动时域如何传递设备状态；
- 固定终点约束对收益和可行性的影响。

## 阅读完成后需要形成的决定

读完后不要求立刻实现复杂的机会价值网络，只需先回答：

1. 本项目主表是否继续同时报告现金收益和期末 SOC；
2. 正式比较是否要求所有候选具有相同期末 SOC；
3. 若使用期末价值，价值参数只能由训练期资料确定，还是使用训练得到的机会价值函数；
4. 是否用额外 look-ahead 并丢弃尾部结果作为稳健性检查；
5. DST 日和数据缺口是否应保留连续 SOC，而不是免费重置电池状态。

在这些问题冻结前，不使用验证期或 report-only 期的未来真实价格来给期末 SOC 定价，
也不因为某一种估值方式让特定模型获胜就事后选择该方式。

## 当前已执行的临时口径（2026-10-02）

第一步已经完成：主结果同时报告现金净收益和期末实际SOC。SOC只以MWh记录，不折算
成钱，也不加入收益。当前仍按现金收益选模型，不事后要求候选具有完全相同的期末SOC；
期末SOC作为并列的审计指标披露。

这不是在否认剩余电量有价值，而是暂时不假设一个未经验证的价格。以后若研究机会价值、
固定终点约束或额外look-ahead，应作为单独的敏感性实验，并继续遵守“参数不能利用
report-only期未来真实价格”的原则。
