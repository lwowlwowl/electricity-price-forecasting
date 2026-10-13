# ERCOT多节点任务专属损失正式训练准备

> 历史状态：准备记录，正式结果已由`39_ERCOT任务专属损失结论与关键事件加权启动.md`覆盖；不要按本文件重复启动训练。

> 日期：2026-10-10  
> 阶段：2020—2022训练、2023内部验证；2024不用于本轮开发，2025完全未读取。  
> 状态：本地实现和验收完成，等待AutoDL seed-0正式训练。

> **结果适用范围**：本文件的任务专属损失实验保持节点/市场 embedding，结论只适用于已知 ERCOT
> 节点的身份条件化 Global 模型；不能作为无 ID 通用模型或未见市场零样本证据。

## 1. 为什么先做这一轮

现有稳定Global三模型全部使用平滑Huber，2023三任务MAE约为`41.59 / 43.66 / 31.40`，
完整收益为`192.83 USD/节点·日`；稳健XGBoost使用DA/RT绝对误差、RT-at-DA伪Huber，
三任务MAE为`35.51 / 35.23 / 27.14`，收益为`292.29`。

此前神经模型的任务专属损失实验不是干净对照：一部分从旧Huber checkpoint继续微调，另一部分
同时加入lag168锚点。第34号文档已经证明lag168残差学习会混淆损失函数效果并破坏RT-at-DA。
因此本轮先回答一个必要而尚未回答的问题：在同一Global结构、同一15节点数据和同一训练预算下，
仅把三任务损失改成各自适合的稳健损失，能否缩小与XGBoost的预测和收益差距。

## 2. 冻结训练协议

三个模型均从随机初始化开始，互不共享Parameter、optimizer和checkpoint：

- DA：F1，MAE；
- RT-at-DA：F0，伪Huber；
- 滚动RT：F1，MAE；
- `global_residual_anchor: none`，不使用lag168；
- 15节点等权采样、逐节点scaler、节点/市场embedding保持不变；
- 最多20 epoch，初始学习率`2e-4`，平台降学习率，patience 7；
- 每个任务按2023节点等权macro MAE保留前2个epoch，再对`2×2×2`三元组做完整结算；
- seed 0不通过门槛时不补seed 1/2，不读取2024。

本轮没有使用XGBoost预测作为输入、残差锚点、门控或教师信号。XGBoost只在相同结算中作为基线。

## 3. 新增流水线

- 三份任务配置：
  - `configs/decision_aware/global_ercot_taskloss_da_mae_v1.yaml`
  - `configs/decision_aware/global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml`
  - `configs/decision_aware/global_ercot_taskloss_rt_mae_v1.yaml`
- 候选冻结：`scripts/decision_aware/prepare_global_taskloss_candidates.py`
- 云端训练、导出和完整结算：`scripts/decision_aware/run_cloud_global_taskloss_v1.sh`
- 预测、价差、收益和风险汇总：`scripts/decision_aware/summarize_global_taskloss_v1.py`

云端脚本固定复用既有`/root/autodl-tmp/epf-venv/bin/python`，不创建新环境、不安装依赖；预检
会核对CUDA、项目路径和ERCOT源数据SHA256。任务中断时复用各自`latest.pt`继续训练。

## 4. 晋级规则

seed-0汇总同时报告：三任务MAE、重要价差方向、价差幅度分位数、完整日均净收益、配对7日块
95%区间、正收益日、偏差罚金、去掉最好5日后的收益以及最大回撤。

只有以下硬门槛全部满足才自动标记为可补多seed：

1. 日均净收益点估计超过稳健XGBoost；
2. 配对95%区间下界大于0；
3. 最大回撤不差于XGBoost；
4. DA、RT-at-DA和滚动RT三项MAE均不差于XGBoost。

该门槛故意严格。若DA或RT-at-DA预测明显改善但尚未全部通过，则结果用于定位下一轮共享编码器、
节点条件化和尖峰分层训练的优先级，不直接宣称最终胜出。

## 5. 本地验收

- 项目现有`.venv`：PyTorch `2.12.0+cu126`，本地CUDA可用；没有创建新环境。
- 三个正式配置分别完成2节点、1 epoch CUDA smoke；训练损失、验证、梯度、checkpoint和候选清单正常。
- 新增任务专属配置、候选冻结、收益汇总与最大回撤测试。
- 针对性测试：`16 passed`。
- 全仓测试：`201 passed in 31.22s`。
- 云端Bash入口通过语法检查。

smoke验证数值不参与模型排名。下一步是打开AutoDL后同步本轮代码与配置，先运行只读预检，再启动
三套seed-0正式训练；预计4090D约`2—3小时`，随后自动导出预测并完成年度结算。

## 6. GPU等待期间完成的结构候选

检查发现现有Global节点/市场embedding只在五条历史流完成编码和融合以后才加入memory与未来query。
因此共享历史编码器在提取DA、RT、负荷、风光和日历模式时并不知道当前节点身份。这可能限制
同一市场内节点基差、尖峰响应和局部价格动态的表达。

新增可开关的`global_entity_conditioning_mode`：

- `post_fusion`：历史默认，保持旧checkpoint结构和行为；
- `pre_encoder`：将同一节点/市场身份向量加入每条流的输入投影，再进入各自Transformer/MLP，
  并继续条件化未来query；融合后的memory不再重复加一次身份。

该结构仍是三套独立模型，只在各任务内部跨15节点共享编码器，不把DA、RT-at-DA和滚动RT合成
一个checkpoint。为保持严格消融，三份`entitypre`配置相对任务专属损失配置只改变条件化位置和
输出目录，损失、融合、宽度、层数、学习率和训练预算完全相同：

- `configs/decision_aware/global_ercot_entitypre_da_mae_v1.yaml`
- `configs/decision_aware/global_ercot_entitypre_rt_at_da_pseudohuber_v1.yaml`
- `configs/decision_aware/global_ercot_entitypre_rt_mae_v1.yaml`

三个结构候选均完成2节点CUDA smoke。新增测试证明：`pre_encoder`下，相同连续输入但不同节点ID
在流编码输出阶段已经产生不同表示；`post_fusion`旧模式在同一阶段仍保持节点无关。全仓回归更新为
`205 passed in 30.74s`，旧模型默认行为未变。云端入口
`scripts/decision_aware/run_cloud_global_entitypre_v1.sh`也已通过Bash语法检查并固定复用现有环境。

冻结运行顺序：先跑第2节的干净任务专属损失基线；读取其2023结果后，再决定是否运行
`entitypre`严格消融。不会在证据产生前同时烧GPU跑两套，也不会用2024回选结构。
