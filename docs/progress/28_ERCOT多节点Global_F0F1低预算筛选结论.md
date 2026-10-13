# ERCOT多节点Global F0/F1低预算筛选结论

> **通用性边界**：本文件使用已知实体身份条件，不能作为无 ID 新市场零样本证据；详见第46号文档。

> 历史状态：本文件只保留低预算链路证据，已被`29`及`43_当前结论与实验总账.md`覆盖；不要据此重复正式训练。

> 日期：2026-10-08  
> 性质：seed 0、3 epoch低预算筛选；不是正式最终结论  
> 数据范围：ERCOT 15节点；2020—2022训练，完整2023验证  
> 当前目标：纯Huber价格预测；尚未进行多节点盈利感知联合训练

## 1. 本次实际运行了什么

本次在AutoDL RTX 4090D 24GB上完成6个完全独立的Global模型训练：

- DA：F0、F1。
- RT-at-DA：F0、F1。
- 滚动RT：F0、F1。

统一条件：

- 使用全部15个ERCOT节点。
- 每个任务和融合方式拥有独立模型、optimizer和checkpoint。
- 训练期为2020-01-01至2022-12-31。
- 验证期为完整2023年。
- seed固定为0。
- 每组只训练3个epoch。
- 每个epoch都计算15节点等权macro MAE。
- 最佳checkpoint按macro MAE选择。
- 本批次清单明确记录`formal_result=false`。

没有运行F2、F1b、seed 1/2、2024外层验证、2025锁定测试或多节点盈利微调。

## 2. 最佳验证结果

### 2.1 DA

- F0：最佳epoch 2，macro MAE = `45.1090`，macro RMSE = `214.5351`。
- F1：最佳epoch 2，macro MAE = `45.6763`，macro RMSE = `213.1790`。
- 按预设macro MAE口径，F0比F1低约`1.24%`。
- 差距很小，而且F1的RMSE反而略低，因此不能认为DA的F0已经稳定胜出。

### 2.2 RT-at-DA

- F0：最佳epoch 1，macro MAE = `39.6310`，macro RMSE = `225.6135`。
- F1：最佳epoch 3，macro MAE = `37.2764`，macro RMSE = `226.7621`。
- 按macro MAE，F1比F0低约`5.94%`。
- F1的epoch 1/2/3 MAE分别为`63.39/113.18/37.28`，波动很大；当前优势必须由更多seed确认。

### 2.3 滚动RT

- F0：最佳epoch 2，macro MAE = `41.5371`，macro RMSE = `194.7546`。
- F1：最佳epoch 3，macro MAE = `45.2928`，macro RMSE = `225.0851`。
- 按macro MAE，F0比F1低约`8.29%`。
- 当前三项中，滚动RT的F0方向最清晰，但仍只有一个seed和三个epoch。

## 3. 当前只能得到的暂定方向

如果只按本批次的2023 macro MAE，暂定组合是：

```text
DA：F0
RT-at-DA：F1
滚动RT：F0
```

这不是最终三模型组合，原因是：

- 只有seed 0，无法估计随机种子不确定性。
- 只有3个epoch，属于低预算筛选。
- 多条验证曲线明显反弹或剧烈波动。
- 还没有完整运行Seasonal、LEAR和XGBoost的15节点年度基线。
- 还没有比较下游电池净收益。
- F0/F1三个任务可以组成8种三元组，当前没有通过同一冻结策略比较这8种组合。
- 没有进行2024外层确认，更没有读取2025锁定回顾性测试。

因此，准确表述只能是：

> 本轮为后续训练提供了方向提示：DA暂时接近打平并略偏F0，RT-at-DA暂偏F1，滚动RT暂偏F0。没有证据证明该组合能够获得最高利润。

## 4. 训练曲线暴露的问题

六组训练的训练Huber总体持续下降，但验证MAE并不单调下降。例如：

- DA-F0：`79.73 → 45.11 → 64.94`。
- DA-F1：`51.95 → 45.68 → 74.81`。
- RT-at-DA-F1：`63.39 → 113.18 → 37.28`。
- RT-F0：`50.90 → 41.54 → 46.74`。

这说明：

- 不能用最后一个epoch代替最佳checkpoint。
- early stopping与`best.pt`保存是必要的。
- ERCOT尖峰会让年度指标产生较大波动。
- 后续需要逐节点指标、尖峰时段指标和多seed，而不是只看一个macro数字。

## 5. 速度与资源

- DA和RT-at-DA每个epoch大约56—66秒。
- 滚动RT-F0每个epoch大约150—161秒。
- 滚动RT-F1每个epoch大约236—244秒。
- F1滚动RT明显比F0更慢；其验证MAE当前还更差。
- 完整批次从首个实测到全部结束约37分钟。
- 按1.88元/小时估算，纯训练时段约1.16元；这不是AutoDL账单的精确金额。

## 6. 本地回收结果

已下载：

- F0 checkpoint：`data/checkpoints/global_ercot_huber_f0_v1/`。
- F1 checkpoint：`data/checkpoints/global_ercot_huber_v1/`。
- 云端日志与状态：`data/results/cloud_global_inner_seed0_screen_v1/`。

本地共有12个正式`.pt`文件，即6个`best.pt`与6个`latest.pt`，加6份`summary.json`，
合计约383.17MB。12个checkpoint均已在本地使用CPU成功反序列化。

所有checkpoint均包含：

- 正式合同哈希`0c5956fb9fd24b5774b6fbdb066a202b17572fa57d012c67bf7354ce7c7d8dd6`。
- 实体映射哈希`2dca4b8cfae5d7d8f88d9d0a23d328aaeea7ba4ea6685b27336d1d01389160e9`。
- 逐节点scaler、映射、训练历史、optimizer和梯度scaler状态。

本地ERCOT源Parquet SHA256仍为：

```text
43c2eb4b4c5c8ce467113f67744d9784ccb591e1febe0423642f363e7369c19d
```

与云端预检和冻结审计一致；本次没有修改`data/markets/**`。

## 7. 与盈利目标的关系

本次训练使用标准化Huber损失，按价格macro MAE选择checkpoint，并没有把电池净收益作为反向传播目标。

三个Global模型仍然可以在下一阶段做盈利感知训练。尚未直接迁移单节点零阶方法的主要原因不是
模型只能追求准确率，而是多节点正式盈利闭环还需要先冻结并验证：

- 连续DA计划与4小时滚动RT MPC。
- 每个节点独立、节点内连续的SOC轨迹。
- 15节点等权的利润目标。
- 三个独立模型通过共同净收益进行联合微调。
- 零阶梯度在多节点下的稳定性、计算成本和外层泛化。

所以本轮checkpoint的角色是“可靠的价格预训练起点”，不是最终盈利模型。

## 8. 下一步

1. 先对6个best checkpoint生成完整2023逐节点预测，并做逐节点、尖峰和负价诊断。
2. 完整运行15节点Seasonal、cutoff-safe LEAR和逐节点XGBoost年度基线。
3. 冻结预测驱动的连续LP/MPC后，对F0/F1形成的8个三元组做同口径2023收益评价。
4. 不因这次单seed MAE直接淘汰DA-F1或RT-at-DA-F0。
5. 确认候选后再增加seed 1/2，并进入2024外层验证。
6. 在纯Huber完整系统稳定后，再从对应checkpoint启动多节点盈利感知联合微调。

在以上步骤完成前，不读取2025锁定回顾性测试，也不声称Global优于local或基线。
