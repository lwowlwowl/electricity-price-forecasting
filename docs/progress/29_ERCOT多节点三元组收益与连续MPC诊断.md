# ERCOT多节点三元组收益与连续MPC诊断

> **通用性边界**：本文件使用已知实体身份条件，收益诊断对 ERCOT 已知节点仍有效，但不能证明新市场零样本泛化；详见第46号文档。

> 日期：2026-10-08  
> 阶段：2023内部验证、seed 0、云端仅训练3 epoch的低预算诊断。  
> 结论边界：本阶段已经完成，但不是最终选模、不是2024外层确认、也不是2025锁定测试。

## 1. 本阶段实际完成了什么

本阶段把第28号文档中的6个Global checkpoint真正接入了盈利评价，而不再只看MAE。
完整流程如下：

```text
云端6个checkpoint
→ 本地逐节点加载合同、映射和scaler
→ 导出F0/F1 × DA/RT-at-DA/滚动RT的完整2023预测
→ 对齐15节点、362个共同交付日和逐小时真值
→ 评价8个F0/F1三元组的旧离散策略
→ 实现预测驱动、偏差罚金感知的连续DA+RT MPC
→ 用连续策略重评8组三元组和Seasonal三模型基线
→ 统计节点等权收益、分项、风险、SOC和置信区间
```

本地预测使用项目根目录`.venv`和RTX 4070 Laptop GPU。结算/LP使用本地CPU，
不需要重新开启AutoDL。

## 2. 新增代码与结果文件

新增或补齐的主要代码：

- `scripts/decision_aware/export_global_f0_f1_predictions.py`：导出6份完整2023预测档案，
  并硬校验checkpoint合同、节点映射和逐节点scaler。
- `src/decision_aware/global_evaluation.py`：按共同交付日对齐三个任务，滚动RT只执行第一步。
- `scripts/decision_aware/evaluate_global_f0_f1_triplets.py`：评价8组三元组的旧离散策略。
- `src/decision_aware/continuous_mpc.py`：预测驱动的连续日前计划和逐小时滚动RT LP/MPC。
- `scripts/decision_aware/evaluate_global_continuous_mpc.py`：15节点并行年度评价、
  7日块Bootstrap和Seasonal同策略对照。
- `tests/test_decision_aware_global_evaluation.py`和
  `tests/test_decision_aware_continuous_mpc.py`：日期/真值对齐、SOC、日放电上限、
  价差信号、信息边界和防真实价格泄漏测试。

机器可读结果：

- 六份预测：`data/results/global_f0_f1_predictions_v1/`
- 旧离散策略：`data/results/global_f0_f1_triplet_backtest_v1.json`
- 连续策略：`data/results/global_continuous_mpc_backtest_v1.json`

## 3. 预测导出与共同评价样本

- DA F0/F1：每份形状`[15, 363, 24]`。
- RT-at-DA F0/F1：每份形状`[15, 363, 24]`。
- 滚动RT F0/F1：每份形状`[15, 362, 24, 4]`。
- 三任务共同评价：15个ERCOT节点、362个交付日。
- 滚动RT每小时只执行4小时预测的第一步；没有把后3步当成额外可执行观测。
- DA、RT-at-DA和滚动RT携带的同小时真实RT价逐元素一致，否则入口会报错。

导出过程中修复了一个字段错误：RT-at-DA档案原本误带空的`executed_timestamps`对象数组。
修复后只有滚动RT保存带时区的执行时间，RT-at-DA重新导出。这个问题没有改模型、checkpoint
或源数据。

## 4. 正式结算口径

本阶段所有收益数字均使用以下固定合同：

- 偏差罚金开启。
- 退化成本为`5.7 USD/MWh × 实际充放电吞吐量`。
- 偏差容忍为`0.03 × |锁定DA计划|`。
- 超额偏差罚金为`2 × |真实RT价| × 超额偏差MWh`。
- 电池为1 MW / 4 MWh，效率0.95，SOC范围0.4—3.6 MWh。
- 初始SOC 2 MWh；同节点连续时段继承SOC，真实缺口才重置。
- 节点间SOC完全独立。
- DA计划提交后锁定，每小时只结算一次。
- 期末SOC只报告MWh，不计入现金收益。

## 5. 旧离散策略诊断

三个策略的8组三元组完整结果如下。

### 5.1 RT-only

8组全部严重亏损。最好组合为`DA F1 + RT-at-DA F1 + RT F0`，节点等权日均净收益
仍为`-1073.73 USD/节点·日`。其平均每日分项约为：DA腿`+205.66`、RT偏差腿`+169.25`、
退化`-33.33`、偏差罚金`-1415.31`。因此亏损主因是偏差罚金，不是退化成本。

### 5.2 follow_da

8组全部为正。最好组合为`DA F1 + RT-at-DA F1`，RT模型在该模式下不改变动作，
所以F0/F1同分，净收益为`171.87 USD/节点·日`。偏差为0，因此罚金为0；代价是滚动RT
预测没有实际参与重调度。

### 5.3 plan_track_topk

8组全部为负。最好组合为`DA F0 + RT-at-DA F1 + RT F0`，仍为
`-101.20 USD/节点·日`。这证明原来的1 MW离散覆盖动作与3%容忍带不匹配。

## 6. 连续罚金感知DA+RT MPC

### 6.1 日前阶段

1. 只读取DA和RT-at-DA预测。
2. 用`预测DA价 − 预测RT-at-DA价`的正负确定该小时允许充电、放电或空闲的方向。
3. 在方向固定后，用连续联合LP决定0—1 MW的具体功率。
4. LP显式包含计划SOC、预期实际SOC、效率、功率、每日放电上限、退化和3%偏差罚金。
5. 计划SOC跨连续日传递；计划提交后锁定。

方向先固定的原因是：完全连续的充/放两个变量可能在同一小时同时充放来钻线性松弛的空子；
完整24小时MILP在本地年度放大时又不具备稳定耗时。当前版本保留连续功率，同时从结构上禁止
同小时充放。它应准确称为“价差方向约束的连续LP”，不能写成无约束LP或Oracle。

### 6.2 实时阶段

1. 每小时只读取当时可见的4小时滚动RT预测和已经锁定的DA计划。
2. 连续LP显式考虑实际SOC、当日剩余放电额度、退化、3%容忍带和预测价格下的偏差罚金。
3. 只执行第一小时动作，然后更新实际SOC并重新求解。
4. 为避免4小时窗口无条件在末端放空，使用最后一个可见预测RT价作为决策用库存机会价值。
5. 真实DA/RT价只在动作确定后进入正式结算；改变真实价格不会改变动作摘要，防泄漏测试已通过。

## 7. 连续策略结果

节点等权日均净收益排名如下：

1. Seasonal/Seasonal/Seasonal：`213.78 USD/节点·日`。
2. DA F1 + RT-at-DA F0 + RT F0：`134.58`。
3. DA F1 + RT-at-DA F0 + RT F1：`115.83`。
4. DA F0 + RT-at-DA F0 + RT F0：`81.43`。
5. DA F1 + RT-at-DA F1 + RT F0：`79.68`。
6. DA F1 + RT-at-DA F1 + RT F1：`64.78`。
7. DA F0 + RT-at-DA F1 + RT F0：`51.88`。
8. DA F0 + RT-at-DA F0 + RT F1：`50.14`。
9. DA F0 + RT-at-DA F1 + RT F1：`21.10`。

### 7.1 当前global最好组合

`DA F1 + RT-at-DA F0 + RT F0`：

- 节点等权日均净收益：`134.58 USD/节点·日`。
- 对节点等权逐日收益做7日块Bootstrap，95%区间：`[44.00, 248.47]`。
- 15个节点的全年平均收益全部为正。
- 最差节点：`LZ_RAYBN = 33.96 USD/日`。
- 最好节点：`LZ_SOUTH = 227.26 USD/日`。
- 节点等权正收益日比例：`41.2%`。
- 每日分项：DA腿`+189.70`、RT偏差腿`-6.29`、退化`-13.07`、偏差罚金`-35.76`。
- 实际吞吐量：`2.293 MWh/节点·日`。
- 节点等权宏观日收益的尾部10%均值：`-98.83 USD/日`。
- 节点等权宏观最大回撤：`2685.83 USD`。
- SOC违规次数：`0`。

同一个F1/F0/F0三元组在旧策略中，`follow_da=126.02`、
`plan_track_topk=-181.23`、`rt_only=-1173.90`。连续MPC相对同三元组的follow_da点估计
提高`8.56 USD/节点·日`，但旧结果没有保存逐日明细，当前不能为该策略差值补做配对区间。

### 7.2 Seasonal同策略基线

- 日均净收益：`213.78 USD/节点·日`。
- 95% 7日块Bootstrap区间：`[54.02, 411.64]`。
- 15个节点全年平均收益全部为正。
- 最差节点：`HB_PAN = 120.35 USD/日`。
- 最好节点：`HB_HUBAVG = 272.79 USD/日`。
- 每日分项：DA腿`+365.36`、RT偏差腿`-32.22`、退化`-21.03`、偏差罚金`-98.33`。
- 实际吞吐量：`3.689 MWh/节点·日`。
- 节点等权宏观尾部10%：`-460.93 USD/日`。
- 节点等权宏观最大回撤：`9733.25 USD`。
- SOC违规次数：`0`。

Seasonal的均值明显较高，但尾部和回撤也明显更差，说明其高均值由少量高收益尖峰日贡献较多；
它不是无风险地全面更好。

### 7.3 global与Seasonal的当前差值

当前global冠军减Seasonal的节点等权日均差为`-79.20 USD/日`。7日块Bootstrap 95%区间为
`[-177.46, 6.14]`，跨0。因此正确结论是：

- 当前3-epoch、seed-0 global低预算系统的平均收益没有超过Seasonal；
- 尚不能声称它统计上稳定落后；
- global的尾部和回撤明显较温和；
- 不能为了均值或风险单独好看而临时更换主指标，后续必须按预先冻结的风险规则共同判断。

## 8. 一个重要反例：最低MAE不等于最高收益

云端预测误差中，RT-at-DA F1的MAE为`37.27`，优于F0的`39.63`；但连续收益冠军使用的是
RT-at-DA F0。滚动RT方面F0的MAE和经济结果方向均优于F1。由此可以确认：三个模型不能按
各自最低MAE直接拼接，必须在同一冻结策略和结算合同下评价完整三元组。

## 9. 测试、源数据与运行问题

- 新增连续策略和对齐测试与相关旧策略测试：`30 passed`。
- 全仓测试：`170 passed in 24.33s`。第一次全测有165项通过、5项因Windows沙箱默认
  临时目录不可写而在fixture建立阶段报错；显式使用项目内临时目录后170项全部通过，
  该临时目录已删除。
- 12个Windows进程会重复加载PyTorch/CUDA DLL并耗尽页面文件；正式年度评价改为3进程，
  结果不受影响，总耗时`1427.07秒`。
- 早期单节点方向LP基准允许同小时同时充放，已被测试发现并废弃；文件
  `global_continuous_mpc_benchmark.json`保留但标为`invalid...do_not_use`，不进入任何排名。
- 四个源Parquet的SHA256与第27号文档完全一致；没有修改`data/markets/**`。
- 没有执行`git add`、`commit`或`push`。

## 10. 当前结论边界

这一部分“本地三元组盈利诊断和连续策略实现”已经完成，但整个多节点研究没有完成。
当前不能把F1/F0/F0或连续策略冻结为最终冠军，原因是：

- 只有seed 0；
- 每个Global模型只训练3 epoch；
- 只使用2023内部验证；
- 尚未完成XGBoost、LEAR等三个任务的完整同策略三元组；
- 尚未完成local/global公平比较；
- 尚未进行2024外层确认；
- 置信区间目前只对节点等权宏观日序列做7日块Bootstrap，正式稿还需节点层级/分层Bootstrap。

## 11. 单节点PRD剩余项

旧单节点主线已经完成独立三模型、融合消融、强基线、偏差罚金版联合训练和冻结后
report-only披露，不建议继续用同一旧留出期反复调参。仍需处理：

1. W02外部合同确认：DA是物理计划还是金融头寸、真实市场结算解释仍需团队确认。
2. 为新年度global协议做严格local/global公平对照：同一2020—2022训练、2023验证、同架构、
   同策略和同节点，旧单节点结果不能直接代替。
3. W11稳健性：成本、效率、功率、预测噪声/延迟的预先情景敏感性尚未形成完整报告。
4. W12论文/组会交付尚未最终收口。
5. W06概率输出、MLP和PriceFM适配仍是可选支线；没有明确收益用途时不应为了“清单齐全”强做。

PRD需要修改的地方：明确区分“旧单节点阶段”和“新多节点年度阶段”；把连续LP/MPC从
“完全由搭档开发”改为当前项目已经实现并负责验证；写入新的嵌套年度切分；其他市场在规则未确认前
只做预测外部验证，不写成真实可交易收益。

## 12. 多节点PRD剩余项与下一步

已完成：正式合同、年度切分、逐节点scaler、稳定映射、balanced sampler、node/market embedding、
三个独立global模型、云端低预算训练、完整2023预测、8组三元组、连续策略和Seasonal同策略基线。

仍未完成：

1. 补齐LEAR、XGBoost、GRU、DLinear/NLinear等三个任务的完整强基线三元组。
2. 入围Global模型补足训练预算和seed 0/1/2；当前3 epoch结果只能筛选。
3. 同架构local/global三seed公平比较。
4. 在2020—2023重训并只用2024做外层确认，冻结结构、epoch规则、策略和三元组。
5. 在冻结策略下补Oracle上界、节点层级Bootstrap、DM/尖峰/负价/半年退化等完整统计。
6. 从冻结Huber冠军开始做多节点三模型联合decision-aware微调。
7. 冻结后用2020—2024五年重训，只读取一次2025锁定回顾性测试。
8. 将冻结预测结构推广到CAISO、NYISO和PJM；结算规则未确认时只报告预测外部验证。
9. 最终论文级总结、老师汇报稿和实验章节框架。

下一步不应直接上2024或2025。先在本地补齐可行的三任务强基线评价入口，并根据计算量决定
哪些训练需要再次开启AutoDL。

## 13. 低收益追因与稳定训练修正

### 13.1 “Global”没有改变三模型边界

这里的Global只表示“同一个任务在15个节点之间共享一套参数”。因此仍然是：

- 一个15节点共享的DA模型；
- 一个15节点共享的RT-at-DA模型；
- 一个15节点共享的滚动RT模型。

三个模型仍不共享Parameter、optimizer或checkpoint。当前并没有把三项任务合并成一个大模型。

### 13.2 为什么价格MAE较低，盈利却落后

新增可复现诊断：`scripts/decision_aware/diagnose_global_screen.py`；机器可读结果：
`data/results/global_screen_failure_diagnosis_v1.json`。

当前Global冠军虽然三项价格MAE均低于Seasonal，但经济决策真正使用的DA减RT-at-DA价差质量很差：

- Global预测价差绝对值均值仅`17.09`，真实值为`34.89`，价差幅度被明显压缩；
- Global价差方向正确率`48.72%`，接近随机；Seasonal为`57.98%`；
- Global预测/真实价差相关系数仅`0.026`；
- 真实价差绝对值99%分位为`721.26`，Global预测只有`61.14`。

因此问题不在“价格MAE完全失效”，而在于三个独立纯Huber模型各自优化单项价格后，没能保留
跨模型的价差方向和极端幅度。该价差恰好又是日前计划最重要的经济信号。

全年均值差也高度集中在少数极端日：Seasonal相对Global的全年总收益差中，最大的10个优势日
解释`97.42%`；Seasonal只在`56.08%`的日期胜出，逐日优势中位数只有`8.24 USD/节点·日`。
这与Seasonal更差的尾部损失和最大回撤一致，不能把当前差值解释为Global每天都全面失败。

### 13.3 三轮训练本身不足以判定收敛

六个run的相邻epoch验证MAE明显震荡。例如F1 DA为`51.95 → 45.68 → 74.81`，F1
RT-at-DA为`63.39 → 113.18 → 37.28`。六路中有两路的最佳点正好落在预算最后一轮，说明
3 epoch只是速度与链路筛选，不是完整收敛训练。

### 13.4 已完成的修正

- 清理`baselines_global.py`中一组无数值影响、但容易造成维护错误的重复定义与重复import。
- Global训练器现在保留验证macro MAE前2个候选checkpoint，而不是只留下一个点。
- 最终checkpoint不再允许由单路MAE直接决定；三路候选必须进入同一冻结连续策略和完整双结算，
  再按收益选择三元组。
- 加入`ReduceLROnPlateau`学习率调度，减少固定高学习率导致的验证震荡。
- 修复断点续训后`stale_epochs`被重置、可能延迟早停的问题，并把scheduler和早停状态写入checkpoint。
- 新增稳定训练配置：`global_ercot_huber_f0_stable_v2.yaml`与
  `global_ercot_huber_f1_stable_v2.yaml`；初始学习率从`5e-4`降为`2e-4`，最多20 epoch，
  patience为7，输出目录与旧三轮结果隔离，不覆盖失败实验。
- 新增AutoDL入口：`scripts/decision_aware/run_cloud_global_f0_f1_stable_v2.sh`。
- 新增候选保存测试；本地2节点/1 epoch稳定配置smoke通过；全仓为`172 passed in 31.65s`。
- 四份源Parquet哈希再次一致，`data/markets/**`没有改动。

下一步需要AutoDL运行这6个稳定seed-0任务。按上一轮4090D实测，若都跑满20 epoch约需
3.5—4.5小时；早停可能缩短，总费用按1.88元/小时约`7—9元`。这仍是2023内部验证，
完成后先重做价差诊断和完整结算，不提前读取2024或2025。

## 14. 稳定v2训练完成与本地复评

### 14.1 云端完成和下载核验

6个稳定seed-0任务于北京时间`2026-10-08 19:51`全部完成。云端产物已下载到：

- `data/checkpoints/global_ercot_huber_f0_stable_v2/`
- `data/checkpoints/global_ercot_huber_f1_stable_v2/`
- `data/results/cloud_global_f0_f1_stable_v2/`

本地逐文件对照云端SHA256，共检查45个checkpoint、清单和日志文件，`0`个不一致。

### 14.2 候选冻结规则

先汇总F0/F1各自保留的epoch候选，再对每个任务按2023节点等权macro MAE保留总体前2个，
不读取收益做这一步：

- DA：F1 epoch 10（MAE `41.51`）、F1 epoch 14（`43.39`）。
- RT-at-DA：F1 epoch 4（`42.81`）、F0 epoch 10（`43.58`）。
- 滚动RT：F1 epoch 8（完整窗口MAE `32.73`）、F0 epoch 8（`33.64`）。

冻结清单为`data/results/global_f0_f1_stable_v2_candidate_manifest.json`。随后只评价
`2×2×2=8`个三元组，符合不做大规模笛卡尔积回选的限制。

### 14.3 稳定Global冠军

完整2023预测和连续MPC复评后，Global冠军为：

- DA：F1 epoch 10；
- RT-at-DA：F0 epoch 10；
- 滚动RT：F1 epoch 8。

主要结果：

- 节点等权日均净收益：`192.83 USD/节点·日`；旧三轮冠军为`134.58`，提高`58.25`。
- 7日块Bootstrap 95%区间：`[74.37, 335.79]`。
- 15个节点全年平均收益全部为正；最差节点`148.68`，最好节点`220.79`。
- 正收益日比例：`45.58%`。
- 尾部10%日均收益：`-97.59`；宏观最大回撤：`1326.55 USD`。
- 每日分项：DA腿`+228.57`、RT偏差腿`-3.94`、退化成本`-14.51`、偏差罚金`-17.30`。
- 吞吐量：`2.545 MWh/节点·日`；SOC违规`0`。

相对Seasonal的日均差从旧三轮的`-79.20`缩小到`-20.95`。配对7日块95%区间为
`[-102.88, 52.41]`，仍跨0。因此仍不能声称Global超过Seasonal，也不能声称稳定落后。
Seasonal均值仍是`213.78`，但其尾部10%和最大回撤仍为`-460.93`与`9733.25`，风险明显更差。

### 14.4 问题是否得到修复

稳定冠军的DA、RT-at-DA和滚动RT第一步MAE分别为`41.59`、`43.66`和`31.40`，均低于
Seasonal的`48.11`、`50.73`和`50.73`。关键价差方向正确率从旧三轮的`48.72%`提高到
`56.58%`，已经接近Seasonal的`57.98%`；预测价差99%分位从`61.14`提高到`189.82`，
说明欠训练造成的价差压缩得到明显缓解，但仍低于真实值`721.26`。

所以正确归因是：降低学习率、完整训练和经济三元组选模确实修复了大部分问题；极端价差幅度
仍被纯Huber明显平滑，是后续其他结构、尖峰损失或联合decision-aware阶段需要继续解决的核心。

机器可读文件：

- 六份预测：`data/results/global_f0_f1_stable_v2_predictions/`
- 完整回测：`data/results/global_continuous_mpc_backtest_stable_v2.json`
- 汇总：`data/results/global_f0_f1_stable_v2_summary.json`

加入通用候选清单、`c0/c1`导出和评价支持后，全仓最终回归为
`173 passed in 43.26s`；唯一warning仍是Windows沙箱不能写默认`.pytest_cache`，不影响测试结果。

## 15. 完整XGBoost三任务强基线与稳健损失修正

### 15.1 先保留原始平方误差负结果

项目根目录`.venv`已经包含`xgboost 3.4.1`，本阶段没有重新安装。最初按常见
`reg:squarederror`训练15个逐节点模型，三任务2023 macro MAE为：DA `46.65`、
RT-at-DA `76.72`、滚动RT `33.92`。同一连续MPC和双结算下，纯XGBoost三元组只有
`55.54 USD/节点·日`，比Seasonal低`158.24`；配对7日块95%区间
`[-309.23, -35.38]`，明确更差。该负结果保留在
`data/results/global_xgboost_continuous_mpc_backtest_v1.json`，不删除也不覆盖。

### 15.2 损失函数只在训练期内部选择

为避免用2023结果反向调参，把inner-train再次固定切成“2020—2021拟合、2022留出”，
对平方误差、绝对误差和伪Huber做15节点macro MAE比较：

- DA：绝对误差`25.70`，平方误差`60.98`，伪Huber`2925.78`；
- RT-at-DA：伪Huber`31.96`，绝对误差`33.33`，平方误差`99.82`；
- 滚动RT：绝对误差`27.58`，伪Huber`32.13`，平方误差`42.63`。

因此冻结为DA/RT使用`reg:absoluteerror`、RT-at-DA使用`reg:pseudohubererror`。正式模型再用
完整2020—2022重训，2023只做一次评价。伪Huber在部分DA节点出现严重数值失稳，因此没有
把同一损失机械套到三个任务。可复现入口为
`scripts/decision_aware/diagnose_xgboost_robustness.py`，三份机器可读结果位于
`data/results/global_baselines_v1/xgboost_robustness_train_internal*.json`。

### 15.3 稳健XGBoost正式预测结果

冻结损失后，2023三任务macro MAE为：

- DA：`35.51`；
- RT-at-DA：`35.23`；
- 滚动RT：`27.14`。

三项都低于稳定Global冠军的`41.59`、`43.66`和`31.40`。逐节点预测、模型和汇总分别保存在
`data/results/global_baselines_v1/<task>_inner/`与
`data/checkpoints/global_baselines_v1/<task>_inner/`。基线运行器新增逐节点断点续跑、缓存合同
校验、结果合并、节点选择和节点级并行；长任务中断后不必从头训练。

### 15.4 同策略完整收益结果

稳健XGBoost纯三元组的2023连续MPC结果为：

- 节点等权日均净收益：`292.29 USD/节点·日`；
- 7日块Bootstrap 95%区间：`[146.00, 469.58]`；
- 相对Seasonal提高：`78.51 USD/节点·日`；
- 配对差值95%区间：`[18.58, 151.62]`，不跨0；
- 15个节点全年平均全部为正，最差节点`244.44`、最好节点`327.57`；
- 正收益日比例：`79.28%`；
- 宏观尾部10%日均收益：`-29.78`；宏观最大回撤：`206.07 USD`；
- 每日分项：DA腿`+333.50`、RT偏差腿`-1.27`、退化成本`-30.07`、偏差罚金`-9.87`；
- 实际吞吐量：`5.276 MWh/节点·日`；SOC违规`0`。

Seasonal仍为`213.78`，稳定Global冠军为`192.83`。所以当前2023内部验证上的排序是
“稳健XGBoost > Seasonal > 稳定Global”。这证明较低的三任务MAE在本次稳健XGBoost中确实
转化为更高收益，但不能推广成“MAE总能决定收益”；此前F0/F1三元组反例仍然成立。
稳健XGBoost正式回测保存在
`data/results/global_xgboost_robust_continuous_mpc_backtest_v1.json`。

### 15.5 结论边界

这是2023内部验证上的强基线，不是最终测试。损失选择完全来自2020—2022内部数据，没有使用
2024或2025样本做训练、选择或报告。下一阶段Global模型若不能在多seed和2024外层确认中超过
该基线，应如实保留XGBoost为当前冠军；不得因为研究主题偏向神经网络而弱化或删除它。

本阶段最终全仓回归为`173 passed in 35.34s`，`git diff --check`无内容错误（仅有现有
Windows CRLF提示）。CAISO、ERCOT、NYISO和PJM四份源Parquet的SHA256均与冻结审计完全一致，
`data/markets/**`没有被修改；没有执行`git add`、`commit`或`push`。

## 16. 强基线差距的逐任务替换诊断（2026-10-09）

为避免把`292.29 - 192.83`的差距笼统归因于“Global不好”，固定同一2023内部验证、
连续DA LP、滚动RT MPC、偏差罚金、κ=5.7和真实结算合同，只在三个任务中逐路替换
稳定Global与稳健XGBoost预测。节点等权日均净收益如下：

- 全稳健XGBoost：`292.29 USD/节点·日`；
- 仅滚动RT改用Global，其余两路XGBoost：`288.03`，下降`4.26`；
- 仅RT-at-DA改用Global，其余两路XGBoost：`252.30`，下降`39.99`；
- 全稳定Global：`192.83`；
- 仅DA改用Global，其余两路XGBoost：`158.90`，下降`133.39`。

因此当前优化顺序冻结为：**DA第一、RT-at-DA第二、滚动RT第三**。滚动RT已经接近强基线，
不应为了“同时优化三路”优先消耗GPU；DA是主要瓶颈。机器可读结果为
`data/results/global_task_replacement_diagnosis_v1.json`。该诊断使用2023内部验证，只用于决定
下一轮内部候选方向，不得写成2024/2025泛化结论。

## 17. 多节点连续MPC收益训练链已打通

新增`src/decision_aware/global_joint.py`，按`节点 + 交付日`严格对齐DA、RT-at-DA和每日24个
滚动RT窗口，并加入节点等权的连续episode sampler。每个节点单独判断真实时间缺口；episode
不会跨节点，也不会跨缺口传SOC。

新增`src/decision_aware/continuous_zo.py`。当前日前求解器先读取预测
`DA - RT-at-DA`的符号来设置充放方向边界，再调用连续LP；这个硬符号门控使完整决策映射
不是普通可微凸层。第一版因此保留已验证的成对零阶估计，但把旧HardTopK收益函数彻底换成
当前正式连续LP/MPC账本：

1. DA与RT-at-DA保持预测中点不变、沿相反方向扰动价差；
2. 滚动RT独立扰动`[日,24,4]`预测；
3. 正负扰动都只用预测作决策，动作锁定后才用真实DA/RT结算；
4. 收益差形成黑盒方向，再用线性代理损失分别传回三套独立参数。

真实checkpoint烟雾试验使用稳定冠军、LZ_LCRA、2023内部验证首个完整日，不更新权重。结果：

- 价差输出梯度范数：`25.70899`；
- 滚动RT输出梯度范数：`27.04863`；
- DA、RT-at-DA、RT参数梯度范数分别为`219470.73 / 234540.98 / 188702.92`；
- 三套参数交集均为0；偏差罚金开启、退化成本为`5.7 USD/MWh`；
- 试验状态为`passed`。

该单日收益为负不用于选模；烟雾试验只证明梯度链、信息边界和参数独立性。结果位于
`data/results/global_joint_continuous_zo_smoke.json`。

## 18. 当前准备好的第一轮优化

Global训练器新增`mae`、`pseudo_huber`、`smooth_l1`和`mse`四种显式训练损失，以及
`--initialize-from`。初始化微调会先把原checkpoint作为epoch 0候选保存，使用全新optimizer；
后续epoch若变差不会覆盖原候选。第一轮配置为
`configs/decision_aware/global_ercot_mae_f1_v1.yaml`，只先训练DA，从稳定F1 epoch 10开始，
使用归一化MAE、`5e-5`学习率和验证macro MAE保留前三候选；最终仍必须与另外两路组成三元组，
用冻结完整结算选择，不能只凭MAE宣布胜出。

两节点CPU smoke已验证初始化、epoch 0保护、MAE反传和候选保存。完整训练需要云GPU；旧AutoDL
地址在2026-10-09返回`Connection refused`，因此尚未启动正式15节点训练。最新全仓测试为
`179 passed in 31.10s`；四市场源Parquet哈希全部与冻结审计一致，源数据未修改。

## 19. DA稳健损失、融合校正与多seed确认（2026-10-09）

### 19.1 云端恢复与实际训练

AutoDL恢复后使用RTX 4090D和`/root/miniconda3/bin/python3.12`。云端没有重新安装
XGBoost；只补齐了当前脚本必需的pandas、scipy、pytest和pyarrow。ERCOT源Parquet云端
SHA256为`43c2eb4b4c5c8ce467113f67744d9784ccb591e1febe0423642f363e7369c19d`，
与本地冻结审计一致。

只优先微调DA，RT-at-DA和滚动RT在收益评价中继续使用稳健XGBoost。三种训练范围为：

- 全模型MAE微调：最佳epoch 5，macro MAE从`41.51`降到`39.59`；
- 只训练价格头：未超过原epoch 0；
- 只训练decoder+价格头：未超过原epoch 0，候选epoch 4 MAE为`46.66`。

这说明从稳定Huber checkpoint继续用MAE训练确实可以降低DA价格误差，但是不能
据此判断收益。

### 19.2 MAE改善没有转化为利润

四个DA候选与冻结的XGBoost RT-at-DA、XGBoost RT组合后，同一连续MPC收益为：

- 原Global DA epoch 0：`158.90 USD/节点·日`；
- MAE最优的epoch 5：`153.47`；
- 全模型epoch 4：`43.19`；
- decoder+head epoch 4：`63.03`。

所以本轮明确拒绝“MAE从41.51降到39.59就是成功”的结论。机器可读结果为
`data/results/global_da_robust_continuous_mpc_v1.json`。

### 19.3 XGBoost主预测 + Global残差校正

为判断Global是否包含XGBoost没有的互补信号，固定RT-at-DA和RT不变，DA候选定义为：

`p_DA = (1-w) × p_XGB + w × p_Global`

正权重`w=2%—40%`全部使收益下降；`w=2%`时为`290.25`，低于纯XGBoost的
`292.29`。这表明普通正向平均不能赢。

随后只在2023内部验证做了一次分阶段反向残差筛选：先`-1%—-15%`，再
`-20%—-75%`，最后`-16%—-22%`定位峰值。峰值在`w=-20%`，即：

`p_DA = 1.20 × p_XGB - 0.20 × p_Global`

其内部验证节点等权日均收益为`298.28 USD/节点·日`，比纯XGBoost高`5.996`。
15个节点年均收益仍全部为正，最差节点`244.89`，最好节点`316.28`。`-16%`和
`-18%`也分别达到`298.25`和`298.23`，因此不是只有单个小数点才成立。

但这仍是内部候选：`-20%`相对纯XGBoost的7日块配对Bootstrap 95%区间为
`[-3.27, 16.28]`，仍跨0；`-15%`平均为`298.13`，配对区间为`[0.31, 12.08]`。
项目按事先确定的平均净收益目标冻结`-20%`为均值冠军，但不把它写成已有稳定
外部优势。详细结果为`global_da_antiblend_peak_continuous_mpc_v1.json`。

### 19.4 并行评价修复

第一次Linux多进程评价因直接向`ProcessPoolExecutor`传递PyTorch CPU Tensor，每个storage
创建文件描述符，触发`Too many open files`。现在提交进程前把只读Tensor转成NumPy，
12进程的60个年度节点任务从长时间串行降到`89.03秒`，数值合同不变。

### 19.5 多seed门槛结果

`-20%`权重冻结后，使用相同Global F1配置补训DA seed1和2。seed1最佳epoch 5、
MAE `43.06`；seed2最佳epoch 19、MAE `40.60`。三个seed及集成的完整收益为：

- seed0：`298.28`，相对XGBoost `+6.00`，配对区间`[-3.27, 16.28]`；
- seed1：`279.78`，相对XGBoost `-12.51`，配对区间`[-21.74, -4.50]`；
- seed2：`278.78`，相对XGBoost `-13.51`，配对区间`[-21.34, -7.34]`；
- 三seed Global预测等权平均后的校正系统：`291.28`，相对XGBoost `-1.01`，
  配对区间`[-4.59, 3.07]`。

三个单seed收益均值为`285.62`，seed间样本标准差为`10.98`。因此seed0的胜出没有
通过多seed稳定性门槛，三seed集成也没有超过`292.29`。当前内部冠军仍是稳健
XGBoost，不进入2024/2025外层年份。详细结论和本地文件索引见
`docs/progress/30_ERCOT_DA校正多seed结论.md`。

下一轮先做小时级符号翻转审计，再比较受约束残差门控、TimeXer风格的内生价格patch+
外生变量cross-attention，以及已打通的连续MPC黑盒收益微调。

当前全仓回归为`181 passed in 24.56s`。四市场源Parquet的SHA256全部与冻结审计
一致，`data/markets/**`没有被修改。
