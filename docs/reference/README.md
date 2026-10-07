# 文献与阅读索引

本目录用于保存已经拿到全文、已经做过初步核对的论文，以及下一阶段需要系统阅读的
文献清单。这里的“待读”表示：目前只根据论文摘要、官方页面或已有初步讨论提炼了
阅读目的，**还没有完成逐节精读和代码复现**，不能把下面的简述直接当成论文结论引用。

`chronos2.pdf`、`toto1.pdf`、`toto2.pdf`和`TimesFM.pdf`保留模型原名，不再改成
数字文件名；从文件名已经可以直接判断对应模型。

## 一、已经在本目录中的编号材料

### 1. Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark

Jesus Lago、Grzegorz Marcjasz、Bart De Schutter、Rafał Weron，Applied Energy，2021。
DOI：[10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)

这篇文献给本项目的思路：

- 原文要求有意义的电价预测结论使用**至少一年**的测试期，不是一个月；短测试期容易
  被节假日、季节和极端价格偶然影响。
- 新模型必须与表现良好的简单模型比较，不能只与弱基线比较。本项目中XGBoost胜过
  Transformer不是“基线太强导致实验失败”，而是必须如实面对的研究结果。
- 应使用多个市场、明确写出训练/验证/测试日期，并对模型差异做统计检验。
- 论文的开放基准同时保留LEAR和DNN，提示下一轮不能只比较Transformer家族。
- 正式年度实验应使用滚动起报评价；是否定期重新校准模型也要明确写进实验合同。

对当前项目的直接影响：新一轮年度实验至少要覆盖一个完整测试年，并把单节点结果升级
为多节点或多市场证据。

### 2. Electricity price forecasting: A review of the state-of-the-art with a look into the future

Rafał Weron，International Journal of Forecasting，2014。
DOI：[10.1016/j.ijforecast.2014.08.008](https://doi.org/10.1016/j.ijforecast.2014.08.008)

这篇文献给本项目的思路：

- 公平比较必须使用相同数据、相同误差评价程序和显著性检验。
- 电价具有多重季节性、尖峰、负价格和制度变化，普通时间序列模型的假设不能直接照搬。
- 统计模型、机器学习模型和混合模型都应保留强代表，不能预设深度模型一定最好。
- 提醒我们同时检查预测精度和经济价值，并对尖峰期单独报告失败案例。

### 3. Out-of-sample tests of forecasting accuracy: an analysis and review

Leonard J. Tashman，International Journal of Forecasting，2000。

这篇文献给本项目的思路：

- 单一起报点的固定测试容易被偶然时期支配，滚动起点能产生更多、更加可靠的样本外误差。
- 应区分“更新输入”与“重新校准模型参数”；论文必须明确测试期内模型是否重训。
- 多个测试时期和多条时间序列能够提高结论的可推广性。
- 偷看留出集再选择方法会污染样本外评价；重新初始化模型并不会自动消除研究者已经得到
  的留出信息。

### 4. Forecasting: Principles and Practice 第5.10节读书笔记

Rob J. Hyndman、George Athanasopoulos，2021，Time series cross-validation。
在线教材：[Forecasting: Principles and Practice](https://otexts.com/fpp3/)

这份笔记给本项目的思路：

- 每个测试点的训练数据只能来自它之前，不能让未来观测进入特征、标准化或模型选择。
- expanding window与fixed-length sliding window都可以使用，但必须预先固定。
- 使用多个rolling origin汇总误差，比只切一次训练/测试更可靠。
- 下一轮应把“是否每日、每月或完全不重新训练”作为明确的消融，而不是默认省略。

### 5. Exploring Representations and Interventions in Time Series Foundation Models

Michał Wiliński等，arXiv:2409.12915。

这篇文献给本项目的思路：

- 更深、参数更多不自动带来更多有效信息，时间序列基础模型的相邻层可能高度冗余。
- 可以用CKA等表示相似性检查层是否重复，再决定剪层或扩大容量。
- 模型会逐层形成趋势、季节性等概念，可以用探针判断当前Transformer究竟学到了什么。
- 当前256维模型没有稳定胜过128维，与“容量不是首要瓶颈”的发现方向一致。

### 6. Universal Redundancies in Time Series Foundation Models

Anthony Bao等，arXiv:2602.01605，2026。

这篇文献给本项目的思路：

- 多种TSFM都可能存在可删除的层、MLP和注意力头，盲目增加模型规模不一定有效。
- 文中讨论context parroting、季节性偏置和均值回归等失败模式，可用于检查我们的模型是否
  只是复制168小时历史或过度依赖周周期。
- 在继续堆叠encoder之前，应先做注意力头、层和输入遮蔽消融。

## 二、已有专题阅读清单

- [储能期末SOC与机会价值_待读文献.md](./储能期末SOC与机会价值_待读文献.md)：期末SOC、
  机会价值、look-ahead、终端约束和双结算储能调度。当前项目继续只报告期末SOC的MWh，
  不把它临时折算成钱。

## 三、当前“Transformer打不过基线”的文献化诊断

当前偏差罚金主合同下，XGBoost验证日均净收益约为`97.32 USD/日`，纯Huber
Transformer约为`83.70 USD/日`，三路active联合微调约为`50.45 USD/日`。文献提示
至少要分开处理下面五个问题，不能把所有差距都归因于Transformer容量不够：

1. **单节点样本太少。** 约五年的小时数据只产生约1800个交付日样本，却训练数百万参数；
   树模型在小样本、强日历和滞后特征条件下本来就可能更稳。
2. **token设计可能不适合时间序列。** 原始时间点token容易丢失局部连续性；PatchTST、
   iTransformer和TimeXer分别用时间patch、变量token和外生变量交叉注意力解决这个问题。
3. **F1/F1b不是唯一的来源融合办法。** TFT使用样本相关变量选择门，TimeXer使用内生-
   外生交叉注意力，PriceFM使用MoE和图约束；它们都比继续手工增加一个池化变体更有先例。
4. **多节点不能直接无约束拼接。** 全局模型可以扩大有效样本，但弱相关节点会增加噪声；
   应加入节点ID、市场ID、共享尺度归一化、图或稀疏选择，并做留节点实验。
5. **active系统首先是决策层问题。** 现有接近1 MWh的离散RT动作远大于3%偏差容忍带，
   高额罚金不是靠多训练几个epoch就能消失。应先把HardTopK替换或补充为显式处理罚金、
   SOC和连续动作的LP/DP优化器，再研究可微或decision-focused训练。

因此，“打败基线”应拆成两个可验证目标：先在同一年度预测合同下缩小或反超XGBoost，
再在同一预测器下证明新的罚金感知决策层提高净收益。不能把预测器变化和策略变化混成一个
实验后只报告最终收益。

## 四、待读：优先解决预测结构与强基线差距

以下文献尚未作为本地全文完成系统精读。

### A1. Are Transformers Effective for Time Series Forecasting?

Ailing Zeng等，AAAI 2023。
[论文页面](https://doi.org/10.1609/aaai.v37i9.26317)

为什么要读：论文用很简单的Linear、DLinear和NLinear在多个长期预测数据集上超过复杂
Transformer，直接解释“模型更复杂却打不过XGBoost”并不反常。

给本项目的实验：把DLinear/NLinear加入DA、RT-at-DA和滚动RT统一基线；若它们也能
接近XGBoost，就说明主要瓶颈可能是token和归纳偏置，而不是注意力层数。

### A2. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

Yuxuan Wang等，NeurIPS 2024。
[论文页面](https://openreview.net/forum?id=INAeUQ04lT)

为什么要读：它明确区分目标价格（内生变量）与负荷、风光、天气等外生变量，同时使用
时间patch自注意力和变量交叉注意力，并通过全局内生token传递信息。

给本项目的实验：它是F1/F1b最优先的文献替代方案。DA与RT-at-DA可把价格作为内生流，
负荷、风光、日历作为外生流；只允许使用起报时真正可用的信息。

### A3. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

Yong Liu等，ICLR 2024。
[论文页面](https://openreview.net/forum?id=JePfAI8fah)

为什么要读：它把“每个变量的整段历史”作为token，让注意力主要学习变量之间的关系，
避免在同一时间点过早混合不同物理量。

给本项目的实验：把DA价、RT价、负荷、风、光分别作为变量token，与当前F1/F1b做单变量
替换消融；不要同时更换损失、策略和数据切分。

### A4. A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers（PatchTST）

Yuqi Nie等，ICLR 2023。
[论文全文](https://openreview.net/pdf?id=Jbdc0vTOcol)

为什么要读：用时间patch保留局部模式、降低注意力长度，并通过channel-independent共享
编码器减少过拟合。两结算储能论文也采用了PatchTST类实时价格预测器。

给本项目的实验：先在滚动RT四步预测上比较当前逐时token与patch token；滚动RT是最短
预测任务，适合先验证分块是否比重新设计整个系统更有效。

### A5. Temporal Fusion Transformers for interpretable multi-horizon time series forecasting

Bryan Lim等，International Journal of Forecasting，2021。
[论文页面](https://doi.org/10.1016/j.ijforecast.2021.03.012)

为什么要读：TFT区分静态属性、已知未来变量和只能历史观测的变量，并使用样本相关的变量
选择网络和门控残差模块。

给本项目的实验：节点ID、市场ID可作为静态变量；日历是已知未来变量；价格和实际负荷/
风光是历史观测变量。变量选择权重还能检查哪些输入真正有用。

### A6. Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting

Yunhao Zhang、Junchi Yan，ICLR 2023。
[论文全文](https://openreview.net/pdf?id=vSVLM2j9eie)

为什么要读：它通过二维的“时间段×变量”表示和两阶段注意力，同时学习跨时间与跨变量关系。

给本项目的实验：若iTransformer只做变量token后仍不足，可把Crossformer作为更完整但更贵
的候选；它尤其适合多节点、多价格流，而不是继续简单concat。

### A7. Deep Learning for Electricity Price Forecasting: A Review of Day-Ahead, Intraday, and Balancing Electricity Markets

Runyao Yu等，arXiv:2602.10071，2026。
[论文页面](https://arxiv.org/abs/2602.10071)

为什么要读：该综述按backbone、head和loss统一拆解近期电价深度模型，并区分DA、日内与
平衡市场。它适合用来检查我们是否遗漏了已有结构，避免把通用TSF模型直接套到市场问题。

### A8. Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx

Kin G. Olivares等，International Journal of Forecasting，2023。
[论文页面](https://doi.org/10.1016/j.ijforecast.2022.03.001)

为什么要读：这是直接面向电价预测、允许外生变量、并在多个市场和多年数据上评价的神经
网络结构。论文报告NBEATSx相对原NBEATS约20%的预测精度改善，并相对其他成熟统计/机器
学习电价方法最高约5%的改善；它比继续手工修改F1汇聚更接近本项目需要的正式强候选。

给本项目的实验：将NBEATSx作为DA与RT-at-DA的优先神经候选，并扩展为共享参数的多节点
global版本；滚动RT仅在轻量版本通过内部年度筛选时保留。三个任务仍使用三个独立模型和
checkpoint，不共享权重。

## 五、待读：多节点、多市场与全局模型

### B1. Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality

Pablo Montero-Manso、Rob J. Hyndman，2021。
[开放版本](https://arxiv.org/abs/2008.00444)

为什么要读：论文解释为什么一个共享参数的global model可以同时学习许多相关甚至异质的
序列，并在数据规模扩大时获得比每节点local model更好的泛化。

给本项目的实验：建立“每节点独立模型”与“共享骨干＋节点embedding＋节点输出头”的严格
对照；不能只做多节点拼表后声称完成了全局学习。

### B2. DeepAR: Probabilistic forecasting with autoregressive recurrent networks

David Salinas等，International Journal of Forecasting，2020。
[论文页面](https://doi.org/10.1016/j.ijforecast.2019.07.001)

为什么要读：DeepAR是“在大量相关序列上训练一个global model”的经典实现，并说明如何
处理序列尺度差异和静态类别特征。

给本项目的实验：即使最终不用RNN，其series scaling、node ID和跨序列采样方法也可直接
用于多节点Transformer数据集设计。

### B3. Forecasting day-ahead electricity prices in Europe: The importance of considering market integration

Jesus Lago等，Applied Energy，2018。
[论文页面](https://doi.org/10.1016/j.apenergy.2017.11.098)

为什么要读：论文同时使用相邻市场输入，并联合预测多个市场；在比利时/法国案例中得到
统计显著改进。它证明多市场信息可以有用，但强调使用起报时已经可得的信息。

给本项目的实验：先比较同一市场内相关节点，再比较跨市场；通过输入遮蔽或特征选择剔除
无效节点，避免未来市场信息泄漏。

### B4. Transfer learning for electricity price forecasting

Salih Gunduz、Umut Ugurlu、Ilkay Oksuz，Sustainable Energy, Grids and Networks，2023。
[论文页面](https://doi.org/10.1016/j.segan.2023.100996)

为什么要读：论文在四个DA市场预训练再对目标市场微调，相对强方法在法国和德国分别报告
约7%和3%的改善。

给本项目的实验：比“把四个美国市场直接混在一起训练”更稳妥的第一步是跨市场预训练，
然后使用少量目标节点数据微调，并与只在目标节点训练比较。

### B5. Forecasting day-ahead electricity prices with spatial dependence

Yifan Yang等，International Journal of Forecasting，2024。
[论文页面](https://doi.org/10.1016/j.ijforecast.2023.11.006)

为什么要读：它把不同区域价格建成图节点，并显式建模空间依赖，而不是把所有区域当作
无结构的额外特征。

给本项目的实验：如果美国数据能获得真实电网拓扑或稳定的统计邻接，应比较图模型与简单
共享模型；若没有可信拓扑，不能伪造物理边。

### B6. Day-ahead electricity price prediction in multi-price zones based on multi-view fusion spatio-temporal graph neural network

Anbo Meng等，Applied Energy，2024。
[论文页面](https://doi.org/10.1016/j.apenergy.2024.123553)

为什么要读：该文使用距离、价格相关性和价格分布相似性构造多种图，再做时空融合。

给本项目的实验：当美国节点之间的真实线路关系难以获得时，可预先定义统计邻接作为一个
候选，但必须只用训练期资料建图，不能用测试期相关性。

### B7. PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting

Runyao Yu等，arXiv:2508.04875，2025。
[论文页面](https://arxiv.org/abs/2508.04875)；[代码](https://github.com/runyao-yu/PriceFM)

为什么要读：它联合训练24个国家、38个区域，引入区域MoE投影和输电拓扑稀疏图，并做
留一区域测试，和我们下一阶段多节点/多市场问题最接近。

与F1/F1b直接相关的已有结果：PriceFM并没有比较我们完全相同的“softmax加权和F1”和
“五token拼接F1b”，所以不能声称论文已经替我们判定二者。但它做了很接近的融合消融：

- 价格与外生特征使用concat时，AQL比残差相加高5.3%（越低越好）；
- cross-attention与残差相加表现接近，但参数更多；
- 4个专家比1个专家的AQL低约6%，继续增加到8个没有收益；
- 随机或移除图约束会明显退化，说明无约束混合所有区域可能引入噪声；
- 部分区域（论文提到德国、法国）不使用邻区反而更好，说明多区域不是越多越好。

给本项目的实验：不要再凭直觉增加F1c；优先复现“共享专家＋节点路由”和“有约束的稀疏
跨节点聚合”，并保留local model作为每个节点的竞争基线。

### B8. Global Models for Time Series Forecasting: A Simulation Study

Hansika Hewamalage、Christoph Bergmeir、Kasun Bandara，Pattern Recognition，2022。
[开放版本](https://arxiv.org/abs/2012.12485)

为什么要读：该文系统改变序列数量、长度、异质性、数据生成复杂度和模型复杂度，说明
global是否优于local并不存在一个固定答案，关键在于“可用序列总量”和“共享模型容量”是否
匹配。复杂的global RNN、前馈网络和LightGBM在短序列、异质序列等困难情形也可以有竞争力。

给本项目的实验：可以直接训练多节点global模型，不需要先在新节点完成一轮local训练；但
必须用同一架构的local版本作对照，并把共享模型容量作为正式消融，避免把“global容量太小”
误判成“节点之间不能共享”。

### B9. Day-ahead electricity price forecasting method integrating multi-scale hypergraph features and dual-layer transformer

Applied Energy，2026。
[论文页面](https://doi.org/10.1016/j.apenergy.2026.127396)

为什么要读：这是较新的多节点电价结构，使用价格趋势相似性构造多尺度超图，再分别学习
空间和时间依赖，并用MIC和SHAP筛选及解释外生变量。

给本项目的实验：它支持“节点间统计关系应显式、稀疏地进入模型”，但其公开摘要所述数据期
较短，不能据此跳过本项目的一年期测试与强基线。第一版global先做节点embedding共享模型；
只有简单global确认有效后，才把训练期相关性图/超图作为单变量升级，避免一次改变过多因素。

## 六、待读：实验筛选与算力分配

### D1. Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

Lisha Li等，Journal of Machine Learning Research，2018。
[论文页面](https://www.jmlr.org/beta/papers/v18/16-558.html)

为什么要读：Hyperband先给许多候选少量训练资源，再把更多epoch或样本分配给较有希望的
候选，用提前停止减少完整训练次数；论文在若干深度学习和核方法任务中报告了数量级的搜索
加速。

给本项目的实验：架构和超参数筛选可把epoch、训练年份或训练节点数作为资源，先淘汰明显
无效候选。但电价有强季节性，候选在被淘汰前仍须经过完整年度的内部验证，不能只凭几个
夏季日或冬季日排名。

### D2. BOHB: Robust and Efficient Hyperparameter Optimization at Scale

Stefan Falkner、Aaron Klein、Frank Hutter，ICML，2018。
[论文页面](https://proceedings.mlr.press/v80/falkner18a.html)

为什么要读：BOHB将Hyperband的多保真资源分配与贝叶斯优化结合，用已观察到的结果引导
下一批候选，比完全随机地产生配置更适合昂贵的神经网络搜索。

给本项目的实验：只有在候选超参数较多时才使用BOHB；所有搜索只看内部验证期，不能让
2025最终测试年参与提前停止、候选生成或资源分配。

## 七、待读：决策导向训练与储能交易

### C1. Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-focused Approach

Linwei Sang等，IEEE Transactions on Smart Grid，2022。
DOI：[10.1109/TSG.2022.3166791](https://doi.org/10.1109/TSG.2022.3166791)；
[开放版本](https://arxiv.org/abs/2305.00362)

为什么要读：论文直接针对“电价预测服务于储能套利”，用oracle决策与预测决策之间的regret
构造可训练代理，并联合预测误差和决策误差。

给本项目的实验：这是当前自制两点零阶收益代理最重要的正式对照。先确认其优化问题与本项目
偏差罚金、双结算是否能改写为同类形式，再决定实现SPO式代理还是保留零阶方法。

### C2. Energy Storage Arbitrage in Two-settlement Markets: A Transformer-Based Approach

Saud Alghumayjan等，Electric Power Systems Research，2024。
[论文页面](https://doi.org/10.1016/j.epsr.2024.110755)；
[开放版本](https://arxiv.org/abs/2404.17683)

为什么要读：它直接研究DA＋RT双结算；价格预测采用Transformer/PatchTST思路，实时决策
采用LSTM＋动态规划机会价值，而不是对价格预测做HardTopK。

给本项目的实验：将“更好的RT预测器”和“更好的RT决策器”分开比较；特别检查DP机会价值
能否减少短窗口过早放电和高偏差罚金。

### C3. Smart “Predict, then Optimize”（SPO/SPO+）

Adam N. Elmachtoub、Paul Grigas，Management Science，2022。
[论文页面](https://doi.org/10.1287/mnsc.2020.3922)

为什么要读：SPO直接用下游决策损失评价预测，SPO+提供可训练的凸代理，并适用于目标参数
线性进入优化问题的情形。

给本项目的实验：先把储能动作写成显式LP/MILP，再判断价格、偏差罚金和双结算是否满足
SPO+条件。不能只因为使用了收益损失就把当前方法称为SPO+。

### C4. Learning with Differentiable Perturbed Optimizers

Quentin Berthet等，NeurIPS 2020。
[论文页面](https://proceedings.neurips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html)

为什么要读：它用随机扰动把离散优化器平滑成可微算子，专门处理排序、选择等前向可算、
反向梯度为零或不连续的问题。

给本项目的实验：HardTopK正是这类离散算子。该论文比目前少量随机方向的手写零阶梯度有
更清楚的数学合同，可作为PFY/perturbed optimizer正式实现的依据。

### C5. Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities

Jayanta Mandi等，Journal of Artificial Intelligence Research（JAIR），2024。
[开放版本](https://arxiv.org/abs/2307.13565)

为什么要读：综述系统比较梯度法、代理损失、黑盒/零阶方法和不同可微优化器，并给出统一
benchmark。它能帮助我们选择方法，而不是在SPO+、PFY和自制零阶方法之间混用名称。

### C6. A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage

Ming Yi等，arXiv:2505.01551，2025。
[论文页面](https://arxiv.org/abs/2505.01551)

为什么要读：论文把价格预测、储能优化和市场出清组成三层模型，分别用隐函数梯度和扰动
方法处理两个优化层。

给本项目的实验：它提示最终研究对象可以是“预测后如何报价”，而不只是预测后选择固定
±1 MWh动作；这与解决3%偏差带下的动作粒度问题直接相关。

### C7. Forecasting Electricity Prices With Decision-Focused Learning for Storage Optimization

2025 IEEE European Energy Market会议论文。
DOI：[10.1109/EEM64765.2025.11050194](https://doi.org/10.1109/EEM64765.2025.11050194)

为什么要读：该文把SPO+扩展到非线性神经预测器，并讨论普通SPO+容易过拟合的问题。

给本项目的实验：如果实现SPO+，必须同时设置传统预测预训练、正则化和验证期收益保护，
不能直接从随机参数只追收益。

### C8. Foundation models for electricity price forecasting and battery arbitrage: Can they replace market-specific forecasting models?

Arkadiusz Lipiecki、Rafał Weron，arXiv:2609.00089，2026。
[论文页面](https://arxiv.org/abs/2609.00089)

为什么要读：它在三个市场比较五类基础模型的统计预测和储能经济价值，明确发现统计优势
不会自动转化成经济优势，而且最优模型依赖交易策略与风险偏好。

给本项目的实验：Chronos/Toto/TimesFM即使MAE更好，也必须在统一储能合同下重新排名；
同时保留市场专用模型和树模型，不预设基础模型一定晋级。

## 八、下一阶段建议：直接做多节点global，local只作为对照

文献没有要求“先在新节点训练local，再升级到global”。结合本项目已经有4个市场、64条
完整小时价格序列，下一阶段可以直接从**ERCOT市场内的多节点global**开始。新节点local
不再是前置阶段，而是global实验旁边的一条低成本对照线。

### 第0步：先建立一次全节点数据合同

- 当前数据为CAISO 3个、ERCOT 15个、NYISO 11个、PJM 35个节点，每个节点均有
  2020-01-01至2026-06-01的56255条小时记录。
- CAISO、ERCOT和PJM的负荷、风光、天气等外生变量主要是市场级重复值；NYISO的负荷
  随节点变化。因此64个节点不是64份完全独立的外生信息，global模型的主要新增信息是
  不同节点的价格历史。
- 固定每个市场的本地时区、夏令时、DA/RT起报时点、缺失处理、逐节点缩放和节点采样权重。
  不能因为PJM有35个节点就让它在四市场global模型中天然占一半以上权重。
- `configs/nodes.yaml`中的旧“代表节点”曾参考2025—2026价格统计选择，因此只能用于开发期
  快速调试，不能作为下一轮最终主结果的抽样依据。正式主表使用预先定义的全部节点，并按
  节点等权汇总，避免挑节点偏差。

### 第1步：直接训练ERCOT global DA模型

- 一个训练样本仍是“节点×交付日”，但15个节点共用同一套模型参数。
- 最小实现为：`逐节点归一化 + node embedding + 共享backbone + 共享price head`。
- balanced sampler先等权抽节点再抽日期；归一化统计只用各节点训练期拟合。checkpoint同时保存
  节点映射、scaler和时间/结算合同，防止云端推理时错配。
- 同时跑必要对照：Seasonal/LEAR、每节点XGBoost、DLinear/NLinear和同架构local模型。
  它们不需要先跑完，
  可以与global并行；作用是回答“共享学习是否真的有益”。
- 第一轮只做DA，不同时扩展RT-at-DA、滚动RT和decision-aware。当前偏差罚金主合同下，
  DA预测与安全的`follow_da`策略决定了最主要的可实现收益，先判断预测结构值不值得扩展。

### 第2步：用嵌套年度切分筛选结构

1. `2020—2022`训练、`2023`内部验证：筛选架构和超参数，可使用Hyperband/BOHB节省算力。
2. 选出的少量候选在`2020—2023`重训，使用`2024`作外层验证和最终定型。
3. 冻结全部规则后，只评估一次`2025`完整测试年。
4. `2026上半年`只能作为补充时期，不用于补选失败模型。

最低候选集合为：`Seasonal/LEAR + XGBoost + DLinear/NLinear + 当前F0/F1-global +
NBEATSx-global`。DA与RT-at-DA再加入`TimeXer-global`；滚动RT只保留轻量直接多步候选，
不因DA采用复杂结构就强制复制。第一轮1个seed筛选；前2名才补3个seed。多节点汇总必须
使用“每个节点等权”的macro平均，同时
报告逐节点胜率、最差节点和配对区间，不能把所有小时直接混成一个大平均。

- 预测主表报告MAE、rMAE和相对强基线的DM检验；经济主表报告固定策略下的完整双结算
  日均净收益、正收益日比例、尾部10%、最大回撤、偏差罚金和退化成本分项。
- 检查固定模型在测试年前后半段是否明显退化；若存在漂移，再比较只更新小型head/adapter的
  月度或日度recalibration。不能让某个模型每日重训而其他模型永久冻结后直接排名。

### 第3步：只把优胜结构扩展成三模型完整系统

- 分别训练DA global、RT-at-DA global和滚动RT global，仍然保持三个独立模型，不合并成
  一个大Transformer。
- “global”只表示同一预测任务在多个节点间共享该模型的参数；DA、RT-at-DA和滚动RT之间
  不共享参数、优化器或checkpoint。联合微调时也继续使用三个独立optimizer。
- 每一路先独立选出前1—2个候选，再组合成最多`2×2×2=8`个三元组；不再穷举972组。
- 三元组全部使用同一年度、连续SOC、5.7 USD/MWh单向吞吐成本和偏差罚金合同评价。
- 回测episode必须是“同一节点的连续日期”；每个节点维护自己的SOC链，禁止把节点A的期末
  SOC传给节点B。选模按节点等权的验证净收益汇总，而不是按总样本量加权。

### 第4步：预测层胜出后再优化决策层

- 先固定预测，比较`follow_da`、罚金感知连续LP/DP和HardTopK，不把预测变化与策略变化
  混在一次实验中。
- 只有某个罚金感知策略稳定胜出后，才用SPO+、perturbed optimizer或其他正式DFL方法
  微调三个global模型；不再从离散HardTopK的手写少方向零阶梯度直接开始。

### 第5步：再扩展到四市场，而不是立即无结构混合64节点

- 首选路线是“多市场预训练 + 目标市场/节点微调”，因为四个市场的规则、价格尺度和时区
  不同，已有电价文献也支持这种迁移学习方式。
- 加入`market embedding + node embedding`，并使用逐市场/逐节点缩放；必要时加小型
  market adapter或MoE路由。
- 没有真实线路拓扑时，不伪造物理图。可以先做无图共享模型；统计相关图只能使用训练期
  价格构建，并作为单独消融。

### “留节点”需要特别区分两种问题

- **已见节点的未来泛化：** 节点参加2020—2024训练/验证，只把2025留作时间测试。这是
  当前最稳妥、最容易完成的主实验。
- **全新节点迁移：** global模型先在其他节点预训练，再用目标节点2020—2023微调小型
  adapter/head，2024验证、2025测试。普通可学习node ID不能直接零样本识别从未见过的新
  节点；真正zero-shot还需要节点位置、电网拓扑等元数据，当前数据不足以支持强结论。

这套流程的核心不是“先local后global”，而是：**直接global开发、local同步对照、年度嵌套
筛选、最后一次测试、预测与决策分阶段**。这样既利用现有64条序列，也避免一次把所有模型、
所有节点、三个预测任务和decision-aware训练混在一起，导致结果无法归因。

## 九、阅读与实现纪律

- “待读”条目精读后，再把确认过的结论移入“已读”，并记录页码/章节和适用前提。
- 不根据最终测试结果临时增加只对某个节点有利的结构；新想法进入下一轮预注册实验。
- 不以“必须打败XGBoost”为理由删除负结果。可以把打败强基线设为研究目标，但不能保证
  结果；真正可发表的贡献还包括识别何时复杂模型无效、为什么无效以及怎样修复。
- 新实现优先复现有正式论文的完整结构，再做单变量改动；避免给已有方法换名后重复造轮子。
