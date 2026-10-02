# 20｜RT-at-DA模型架构图与训练图说明

## 原有图保持不变

原有`docs/model_architecture.drawio`和`docs/training_loop.drawio`已经精确恢复，
继续保留旧Transformer/联合模型的历史结构，不被本次新模型覆盖。

## RT-at-DA模型架构图

新文件为`docs/rt_at_da_model_architecture.drawio`，只描述RT-at-DA内部：

1. 五条历史流分别进入独立StreamEncoder；
2. 堆叠为`[B,168,5,128]`并加入来源embedding；
3. 消融后冻结的F0在每个小时将五来源concat，投影到128维，再经过Pre-LN残差MLP；
4. 时间Transformer生成`Memory [B,168,128]`；
5. 交付日24小时日历经过投影，成为24个query的条件；
6. QueryDecoder查询完整Memory；
7. Linear价格头和反归一化输出`p_rt_at_da [B,24]`。

## RT-at-DA训练图

新文件为`docs/rt_at_da_training_loop.drawio`，只描述新模型训练：

- F0/F1/F1b/F2均使用纯Huber预测损失，收益不反向传播；
- AMP、反向传播、梯度裁剪和AdamW只更新当前RT-at-DA实例；
- 每个epoch在冻结DA参考面板和滚动RT下计算完整双结算验证收益，并据此保存checkpoint；
- 三个seed训练结束后，按验证收益的三seed均值比较各结构；
- F1b保留来源self-attention，但把单一softmax加权和换成五token concat、线性投影
  和残差MLP；其验证收益110.92 USD/日，F0/F1/F2为110.68/110.10/108.69；
- F1b与F0、F1、F2的成对95%区间均跨0，因此只能记为均值最高的结构候选，
  不能声称显著胜出或理论最优；
- 2026-01至06月已经暴露，只作为report-only披露，不用于返回重选。

因此完整双结算收益目前没有反向传播到RT-at-DA；它只用于epoch checkpoint和融合
结构选择。随后用DA F1三个seed、滚动RT F0三个seed和两种RT-at-DA各三个seed完成
下游传播确认：F1b仅比F0高0.24 USD/日，95%区间[-4.36,5.36]跨0。因此当前模型
架构图继续反映实际部署的F0，不因F1b单阶段均值略高而改写部署图。

## 图文件校验

两张RT-at-DA新图都是可编辑draw.io XML。使用项目内drawio skill检查后均为：

```text
0 errors, 0 warnings
score 0: 0 through-vertex, 0 crossings, 0 overlaps
```

本机没有draw.io命令行程序，因此没有自动导出PNG；这不影响用draw.io桌面版或
diagrams.net打开和继续编辑源文件。
