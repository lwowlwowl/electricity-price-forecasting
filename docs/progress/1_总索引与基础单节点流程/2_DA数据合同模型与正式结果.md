# DA数据合同、模型与正式结果

> 阶段定位：ERCOT `LZ_LCRA` 单节点早期实验。这里的收益是 DA-only 代理，
> 不能与后续完整偏差罚金、多节点或无 ID 结果直接比较。

## 数据与信息合同

- 在交付日前一天当地时间 10:00 起报，读取此前 168 个完整小时；
- 预测交付日 00:00—23:00 共 24 个小时，23/25 小时夏令时日排除；
- 历史流包括 DA、RT、负荷、风光系统量和日历；目标日只提供已知日历；
- 未来真实负荷、风光、天气和 RT 价格均不可见；
- 归一化统计量只由训练期估计，验证和测试复用；
- 样本数为 1809 个训练日、180 个验证日和 333 个测试日。

数据实现：`src/decision_aware/dataset_da.py`。

## 模型

五条历史流分别编码为 168 步表示，再在同一小时内做来源交互和加权汇聚，随后由
时间 Transformer 建模。24 个可学习 query 结合目标日日历，通过 cross-attention
读取历史 memory 并输出 `p_da [B,24]`。DA 模型没有 RT 输出头，也不与 RT 模型共享参数。

代码：`src/decision_aware/model_da.py`；可编辑架构图：`docs/da_model_architecture.drawio`。

## 实验合同

三个种子都使用完整的1809个训练日、180个验证日和333个测试日。最大10个
epoch，early-stop patience为3；每个checkpoint只按验证集DA-only代理收益选择，
没有根据测试集反向挑模型。模型、HardTopK、BESS参数和时间合同保持相同。

三个结果文件的`device`都为`cuda`。训练期间实测：seed 1的GPU利用率59%、
功耗58.7W；seed 2的GPU利用率49%、功耗52.9W。两次都能看到独立Python训练
进程，因此不是CPU训练伪装成CUDA训练。

## 三个种子结果

- seed 0：最佳验证轮4，验证收益113.52；测试收益102.78，regret 89.69，
  MAE 41.20，RMSE 79.83，正收益日93.39%。
- seed 1：最佳验证轮4，验证收益108.19；测试收益83.10，regret 109.37，
  MAE 18.71，RMSE 73.91，正收益日88.59%。
- seed 2：最佳验证轮9，验证收益115.99；测试收益98.98，regret 93.49，
  MAE 20.78，RMSE 75.71，正收益日92.49%。

三种子测试均值和样本标准差：

- 收益：94.96 ± 10.44；
- regret：97.52 ± 10.44；
- MAE：26.90 ± 12.43；
- RMSE：76.48 ± 3.04。

结果文件为`data/results/da_transformer_seed0.json`、
`da_transformer_seed1.json`和`da_transformer_seed2.json`。seed 0原checkpoint已
另存为`pilot_LZ_LCRA_best_revenue_seed0.pt`；seed 1和2使用各自独立目录。

## 与基线比较

- XGBoost收益114.79；
- 周期基线收益112.06；
- ExtraTrees收益111.75；
- 当前Transformer三种子平均收益94.96；
- GRU收益62.50。

当前没有一个Transformer种子超过XGBoost。不能声称当前来源注意力结构带来了
收益优势。seed 1的MAE最低但收益也是三个Transformer中最低；seed 0的MAE最差，
收益反而最高，再次说明误差和收益排序不同。

## 阶段结论

按验证集收益冻结 seed 2 进入后续结构实验，不按测试收益挑 seed。三个 Transformer
seed 均未超过 XGBoost；seed 1 的 MAE 最低但收益最低，再次证明预测误差与经济价值
不等价。后续 F0/F1/F2 融合消融及完整双结算结论统一归入阶段 2。
