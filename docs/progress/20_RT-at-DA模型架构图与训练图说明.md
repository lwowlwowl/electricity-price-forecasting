# 20｜RT-at-DA模型架构图与训练图说明

## 原有图保持不变

原有`docs/model_architecture.drawio`和`docs/training_loop.drawio`已经精确恢复，
继续保留旧Transformer/联合模型的历史结构，不被本次新模型覆盖。

## RT-at-DA模型架构图

新文件为`docs/rt_at_da_model_architecture.drawio`，只描述RT-at-DA内部：

1. 五条历史流分别进入独立StreamEncoder；
2. 堆叠为`[B,168,5,128]`，在同一小时内做五来源注意力；
3. 可学习权重把五个来源汇聚成168个时间token；
4. 时间Transformer生成`Memory [B,168,128]`；
5. 交付日24小时日历经过投影，成为24个query的条件；
6. QueryDecoder查询完整Memory；
7. Linear价格头和反归一化输出`p_rt_at_da [B,24]`。

## RT-at-DA训练图

新文件为`docs/rt_at_da_training_loop.drawio`，只描述新模型训练：

- 单模型使用Huber预测损失；
- AMP、反向传播、梯度裁剪和AdamW只更新当前RT-at-DA实例；
- 每个seed内部按验证MAE保存最佳epoch；
- 三个seed训练结束后，checkpoint才进入27候选完整双结算验证选择；
- 选中seed 2后，测试集只评价一次。

因此完整双结算收益目前没有反向传播到RT-at-DA；它只用于训练完成后的系统选择。

## 图文件校验

两张RT-at-DA新图都是可编辑draw.io XML。使用项目内drawio skill检查后均为：

```text
0 errors, 0 warnings
score 0: 0 through-vertex, 0 crossings, 0 overlaps
```

本机没有draw.io命令行程序，因此没有自动导出PNG；这不影响用draw.io桌面版或
diagrams.net打开和继续编辑源文件。

