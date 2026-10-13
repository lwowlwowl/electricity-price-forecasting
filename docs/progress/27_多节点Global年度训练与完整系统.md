# 多节点 Global 年度训练与完整系统

> 创建日期：2026-10-07  
> 当前阶段：仓库审计、完整测试、四市场数据合同检查和 global 缺口分析已完成。  
> 重要边界：本文记录新年度协议；第25、26号文档中的单节点旧结果继续保留，不覆盖、不删除。

> **通用模型口径说明（后续修订）**：本文件首次正式引入可学习的 `node_id` / `market_id`
> embedding，用于“已知节点/已知市场的多节点 Global 模型”。这对已见实体的共享预测仍然有效，
> 但不能证明训练时未出现的新市场可以直接使用。自本文件起，凡配置中
> `global_use_entity_embeddings: true` 的结果，统一标记为“身份条件化 Global 结果”，不得写成
> “任意新市场通用模型”证据。通用模型必须使用 `global_use_entity_embeddings: false`、
> `global_normalization_mode: context_robust`，并通过留一市场零样本验证。

## 1. 本阶段目标

新协议按以下顺序推进：

```text
四市场全节点数据合同
→ ERCOT 15节点的三个独立global预测模型
→ 纯Huber完整系统
→ 罚金感知连续LP/MPC
→ 三模型联合decision-aware微调
→ CAISO/NYISO/PJM预测外部验证
→ 2025锁定回顾性年度测试
→ 论文级总结
```

“global”只表示同一个预测任务在多个节点之间共享参数。DA、RT-at-DA、滚动RT始终是
三个独立模型，不共享可训练Parameter、optimizer或checkpoint。

## 2. 2026-10-07 仓库起点审计

执行命令：

```powershell
git status --short --branch
git log -8 --date=iso --pretty=format:'%h%x09%ad%x09%s'
& external/toto/.venv/Scripts/python.exe -X utf8 -m pytest -q
```

审计结果：

- 当前分支为`main`，跟踪`origin/main`。
- 起点提交为`818ccc6`：`feat(decision-aware): 闭环偏差罚金实验并冻结多节点global研究协议`。
- 开始时已有用户修改：`docs/reference/README.md`。本阶段保留该修改，没有覆盖或清理。
- 完整测试结果：`133 passed in 8.56s`（收尾回归；首次审计运行同样为133项全通过）。
- `data/checkpoints`共有104个文件，约2.442 GB；`data/results`共有619个文件，约0.552 GB。
- 没有删除旧checkpoint、失败实验、日志或结果。
- 没有修改`data/markets`中的Excel或Parquet。
- 没有执行`git add`、`commit`或`push`。

## 3. 四市场全节点数据合同

只读审计命令：

```powershell
& external/toto/.venv/Scripts/python.exe -X utf8 `
  scripts/decision_aware/audit_multimarket_global_data.py --summary-only
```

机器可读结果：

`data/results/multimarket_global_data_audit.json`

审计脚本：

`scripts/decision_aware/audit_multimarket_global_data.py`

### 3.1 数据范围与完整性

- CAISO：3个节点，168,765行；每节点56,255小时。
- ERCOT：15个节点，843,825行；每节点56,255小时。
- NYISO：11个节点，618,805行；每节点56,255小时。
- PJM：35个节点，1,968,925行；每节点56,255小时。
- 合计：4个市场、64个节点、3,600,320行。
- 各节点本地时间均覆盖`2020-01-01 00:00`至`2026-06-01 23:00`。
- 四个市场都是完整矩形面板：同一市场内每个节点拥有同一组小时。
- 必要字段缺失率均为0；节点×UTC时间重复数为0；非1小时时间缺口数为0。
- `timestamp_local`与市场IANA时区转换一致；`hour_local`与本地时间小时一致。
- 实际节点集合与`configs/nodes.yaml`中的`all`列表完全一致。

### 3.2 原始Parquet哈希

- CAISO：`8766a56c0557e8222334a9cfe88886101c2c0791eab46efb6120adb3ca469031`
- ERCOT：`43c2eb4b4c5c8ce467113f67744d9784ccb591e1febe0423642f363e7369c19d`
- NYISO：`dfb4eb2c1863250afbd2509dc93e2aab637668d2df83527549773c13261c722b`
- PJM：`d7a27f9453f18f99c464b7fb34bab52af154dbdc357241d4bc196ccda84628db`

后续云端同步必须重新计算并匹配这些哈希。

### 3.3 时区

- CAISO：`America/Los_Angeles`。
- ERCOT：`America/Chicago`。
- NYISO：`America/New_York`。
- PJM：`America/New_York`。
- UTC序列连续，但本地交付日仍有夏令时23/25小时日。当前日级数据集明确排除非24小时日，
  不能静默补成普通24小时日。

### 3.4 字段和外生变量层级

四市场均有31列，包括DA价、RT价、实际负荷、实际风光、本地日历、实际天气和名为
`forecast_*`的天气列。

跨节点逐小时比较结果：

- CAISO、ERCOT、PJM：负荷、风、光和所有天气列在同一市场同一时刻对所有节点完全相同，
  属于市场级重复变量。
- NYISO：`actual_load_mw`随节点变化；风、光和天气仍是市场级重复变量。
- 节点间真正稳定变化的主要信息是DA/RT价格；NYISO另有节点负荷。
- 重复15次或35次的市场级变量只能作为一份市场信息使用，不能描述为15份或35份独立信息。
- `forecast_*`列没有独立的`forecast_issue_time`字段和发布时间元数据。正式模型在核实实际
  发布边界前不得启用这些未来天气列。当前三个数据集也没有使用这些天气列。

### 3.5 为什么必须逐节点/逐市场缩放

仅看内部训练期`2020—2022`，价格尺度已经明显不同：

- CAISO节点DA标准差约57.69—66.31，RT标准差约59.99—66.03。
- ERCOT节点DA标准差约518.86—523.27，RT标准差约559.43—576.13；期间最高价超过
  9,000/10,000 USD/MWh。
- NYISO节点DA标准差约28.41—44.76，RT标准差约44.53—77.32。
- PJM节点DA标准差约27.63—42.08，RT标准差约55.35—90.53。

因此，多市场共用一个价格均值/标准差会被ERCOT尖峰严重支配。正式实现必须使用训练期拟合的
逐节点价格scaler；验证和测试只能复用，不能重新拟合。

## 4. 冻结年度时间切分

新协议只使用以下切分：

1. 内部筛选：训练`2020-01-01—2022-12-31`，内部验证`2023-01-01—2023-12-31`。
2. 外层确认：重训`2020-01-01—2023-12-31`，外层验证`2024-01-01—2024-12-31`。
3. 最终训练：`2020-01-01—2024-12-31`完整五年。
4. 锁定回顾性测试：`2025-01-01—2025-12-31`。
5. `2026-01-01—2026-06-01`只作补充/report-only，不参与回选。

旧`PilotConfig`默认仍是`2020—2024训练、2025上半年验证、2025下半年至2026测试`，与新协议
不一致。正式global入口不能依赖这些旧默认值，必须使用显式阶段配置并硬校验。

## 5. 唯一正式结算合同（待代码化，数值已冻结）

- `use_deviation_penalty = true`。
- `bess_kappa = 5.7 USD/MWh`。
- 退化成本=`5.7 × 实际充电MWh + 5.7 × 实际放电MWh`。
- 偏差罚金=`2 × |真实RT价| × max(|u_RT_actual-u_DA|-0.03|u_DA|, 0)`。
- 功率1 MW；容量4 MWh；效率0.95。
- SOC下限0.4 MWh；上限3.6 MWh；初始SOC 2 MWh。
- 同一节点连续日期继承SOC；只在该节点真实缺口形成新连续段时重置。
- 不同节点绝不能继承SOC。
- DA计划锁定；每个交付小时只结算一次。
- 期末SOC只报告MWh，不货币化。

当前代码能计算上述单节点账本，但尚未形成一个全入口共用、带哈希的正式
`Experiment/Settlement Contract`对象。

## 6. 当前代码已经具备的能力

- 三个独立数据集：DA、RT-at-DA、滚动RT。
- 三个独立模型类/实例/checkpoint。
- 联合训练正式配置可启用三个独立optimizer，并检查模型之间无共享参数。
- 单节点训练期scaler可由验证/测试复用。
- 单节点连续日期SOC、锁定DA、偏差罚金、退化成本和完整双结算已有测试。
- 单节点MILP Oracle可显式处理偏差罚金，但它使用真实价格，是评价上界，不是可执行预测策略。
- 旧单节点偏差罚金实验、负结果、report-only结果和三seed日志均保留。

## 7. Global 缺口分析

### 7.1 P0：正式训练前必须补齐

- 没有“节点×交付日”的多节点数据集；现有builder每次只接受一个`cfg.node`。
- 样本中没有`node_id`、`market_id`、节点名或市场名张量。
- 模型中没有node embedding或market embedding。
- 没有逐节点scaler字典；当前checkpoint只保存一个节点的`norm_stats`。
- 没有先等权抽节点、再抽日期的balanced sampler；跨市场等权采样也不存在。
- 联合数据集只按单节点日期排序，尚未证明或强制SOC不跨节点。
- 没有稳定节点/市场映射及其checkpoint恢复测试。
- 没有正式合同对象、合同哈希和所有正式入口的统一硬校验。
- `PilotConfig.use_deviation_penalty`默认仍为`false`。旧普通训练入口不会因正式运行误关罚金而
  统一硬失败。
- 旧checkpoint没有合同哈希、节点映射、市场映射、Git commit和数据范围等完整字段。
- 新年度切分没有独立配置/阶段枚举；旧默认切分不适用。

### 7.2 P1：第一轮global跑通后补齐

- 没有DLinear/NLinear、NBEATSx-global、TimeXer-global和轻量PatchTST-global实现。
- LEAR当前位于旧通用评估路径，尚未接入新多节点年度三任务统一接口。
- 没有节点等权macro汇总、逐节点胜率、最差节点、高波动节点和节点类别报告。
- 没有年度分小时、尖峰、负价、前后半年退化和节点层级Bootstrap统一报告。
- 没有可复现的local/global同架构预算比较入口。

### 7.3 P1：决策层

- 当前可执行active策略仍是离散HardTopK/`plan_track_topk`，动作粒度与3%容忍带冲突。
- 现有连续MILP使用真实价格，是Oracle；不能直接当成预测驱动的可执行LP/MPC。
- 需要新增预测价格驱动的连续DA计划与每小时4小时RT滚动MPC，并在验证期冻结。
- 在该策略通过前，不启动新的联合decision-aware正式训练。

### 7.4 P2：多市场

- ERCOT的D-1 10:00合同已有依据；CAISO、NYISO、PJM不能静默沿用该时点。
- 其他市场真实偏差结算规则未冻结，因此第一阶段只做预测外部验证，不声称真实可交易收益。
- 没有真实电网拓扑；第一版使用无图global。统计图只能由训练期价格构建并单独消融。

## 8. 旧checkpoint兼容性抽查

抽查DA F1 seed1和联合active seed0：

- DA checkpoint含`model_state/config/norm_stats/evaluation_contract`，但没有统一`contract_hash`、
  `node_mapping`或`market_mapping`。
- 联合bundle含三个`model_states`和`optimizer_states`，但没有新的global scaler、映射和统一合同哈希。
- 因此旧权重可作为历史单节点初始化或结构参考；正式global运行必须生成新格式checkpoint，
  不能给旧文件补标签后直接使用。

## 9. 文献核对对第一版设计的约束

本阶段只依据论文页、期刊页、OpenReview和作者官方代码核对：

- Montero-Manso与Hyndman支持把多序列作为一个global任务，但并不保证global必胜；必须保留
  同架构local对照。
- DeepAR明确使用跨序列共享模型、序列尺度处理和静态类别embedding；这支持逐节点scaler与
  node ID，但本项目按节点等权，而不是照搬其按序列尺度加权采样。
- NBEATSx是电价与外生变量的直接候选，优先用于DA和RT-at-DA。
- TimeXer把内生价格patch与外生变量交叉注意力分开，适合DA/RT-at-DA，但不能使用未确认
  发布时间的未来外生量。
- DLinear/NLinear是必须加入的简单强基线。
- PatchTST优先作为小型滚动RT候选；iTransformer作为变量token候选，均不与第一版简单
  global同时大规模堆叠。
- Hyperband用于按epoch分配资源；BOHB只在配置空间确实较大时使用，不与Hyperband重复堆叠。
- SPO+和perturbed optimizer有不同数学前提。当前手写两点零阶代理继续按原名保留，不能改名。

结论：第一版global只实现`逐节点scaler + node embedding + 共享backbone + 共享输出头 +
节点等权采样`，暂不加入图、MoE或超图。

## 10. 冻结候选与seed规则

- 强基线：Seasonal、LEAR、XGBoost、GRU、DLinear/NLinear。
- DA：当前F0/F1 local/global、NBEATSx-global、TimeXer-global、DLinear/NLinear-global、XGBoost。
- RT-at-DA：当前F0/F1 global、F1b一次诊断、NBEATSx-global、TimeXer-global、
  DLinear/NLinear-global、XGBoost。
- 滚动RT：当前F0/F1 global、F1b一次诊断、小型NBEATSx-global、小型PatchTST-global、
  DLinear/NLinear-global、XGBoost。
- 所有必要候选先跑seed 0；每任务约前2名再补seed 0/1/2。
- 最终胜出架构必须完成local/global三seed公平比较。
- 三元组每路最多2个checkpoint，总数最多8组。
- F2不作为默认正式候选；没有新的global证据不重新做F0/F1/F1b/F2三seed穷举。

## 11. 失败实验与负结果登记

- 旧单节点XGBoost在偏差罚金安全策略验证期胜过Transformer-Huber。
- 旧三路active联合微调相对自身epoch0只小幅改善，区间跨0；report-only下降。
- 旧256维容量没有稳定改善，第一轮global保持约128维，不直接扩大到512维。
- 这些结果全部保留，但不直接外推为新年度、多节点global结论。

## 12. 当前验收与下一步

本阶段已经完成：

- 指定仓库和文档审计。
- Git状态、最近提交、旧输出和checkpoint盘点。
- 初始完整pytest记录`133 passed`；Global基础设施完成后为`157 passed`。
- 四市场64节点数据范围、缺失、时区、字段、重复外生变量和原始哈希审计。
- 当前global、正式合同、策略和checkpoint格式缺口分析。
- 第一版global结构依据的一手文献核对。

下一步本地实现顺序：

1. [x] 正式合同对象、哈希与硬校验。
2. [x] 多节点数据集、稳定映射、逐节点scaler和balanced sampler。
3. [x] node/market embedding与三个独立global模型。
4. [x] 防泄漏、采样平衡、SOC隔离、参数/optimizer独立、checkpoint恢复测试。
5. [x] 2节点、数日、1 epoch CPU smoke。
6. [x] Seasonal、cutoff-safe LEAR、逐节点XGBoost强基线与统一年度评估入口。
7. [x] 预测驱动、罚金感知的连续DA LP与逐小时RT MPC及测试；当前为内部验证诊断，尚未外层冻结。

首批global-Huber云端任务还需完成第6项和任务清单冻结；在此之前不启动AutoDL。

## 13. 2026-10-08 Global基础设施实现与本地验收

本轮已经实际加入：

- `src/decision_aware/contracts.py`：唯一正式年度/结算合同、稳定SHA256合同哈希、
  checkpoint防篡改和旧checkpoint拒绝逻辑。
- `src/decision_aware/global_data.py`：稳定节点/市场映射、逐节点scaler保存、节点等权
  sampler、跨市场“先市场后节点”sampler，以及三个任务的多节点Dataset构造。
- 滚动RT不再把单个小时窗口当作global采样单位；现在先把同一节点同一当地交付日的
  24个滚动窗口组合为`节点×交付日`样本，DST 23/25小时日明确排除。
- `src/decision_aware/entity_conditioning.py`：node embedding与market embedding；同时注入
  历史memory和目标query。旧单节点配置默认关闭，因此旧checkpoint结构不变。
- 三个独立模型均支持global身份条件，但仍是三个独立实例，不共享Parameter。
- `locked_dual_backtest_by_node`逐节点调用完整结算，每个节点从自己的SOC链开始，
  不把一个节点的终点SOC传给另一个节点。
- `scripts/decision_aware/train_global_forecaster.py`：正式ERCOT global-Huber训练入口；
  支持inner/outer年度阶段、完整年度验证、macro节点MAE选模、latest断点续训、AMP、
  数据哈希硬校验、合同/映射/scaler/Git提交写入checkpoint。
- `configs/decision_aware/global_ercot_huber_v1.yaml`：ERCOT 15节点共同基础配置。

本地真实数据smoke：

- 节点：冻结`all`清单字母序前两个`HB_BUSAVG`、`HB_HOUSTON`；只作连通性测试，
  不作正式节点抽样或模型优劣结论。
- 三个任务均完成1 epoch、每节点3个交付日的CPU smoke。
- DA、RT-at-DA、滚动RT的loss和梯度均为有限值。
- 三个模型Parameter集合互斥，三个optimizer参数集合互斥。
- 三个checkpoint均已保存并在CPU重新加载；正式训练入口的三条task路径也分别通过。
- DA latest checkpoint已额外通过`--resume`恢复测试。
- smoke结果：`data/results/global_ercot_two_node_smoke.json`。
- smoke checkpoint：`data/checkpoints/global_ercot_smoke/`以及
  `data/checkpoints/global_ercot_huber_v1/*_smoke/`。它们全部带`smoke_only=true`，
  不能进入正式结果表。

回归验收：

- 完整pytest：`157 passed in 8.82s`。
- 四份源Parquet重新计算的SHA256与2026-10-07审计完全一致；本轮没有修改
  `data/markets/**`。

强基线入口随后已经补齐，见第15节。当前尚未开始正式训练；2023内部验证、2024外层验证和
2025锁定回顾性测试仍未产生正式结果。

## 14. 云端任务、seed、验证和最终测试占位

- 云端任务：已完成F0/F1乘以三个任务的seed-0、3 epoch低预算筛选，详见第28号文档。
- 正式seed：仅完成seed 0低预算筛选；seed 1/2尚未开始。
- 2023内部验证：已完成本批6组完整年度预测误差、8组三元组盈利评价和Seasonal三任务
  同策略评价；LEAR/XGBoost等其余完整三任务强基线尚未完成。
- 2024外层验证：尚未开始。
- 2025锁定回顾性测试：尚未读取。
- 是否超过Seasonal/LEAR/XGBoost：当前3-epoch seed-0 global冠军均值未超过Seasonal；
  配对7日块区间跨0，不能声称稳定落后。LEAR/XGBoost完整三任务尚无同策略证据。
- 当前可写入论文的结论：只可写数据合同与旧单节点负结果，不能写global优于local。
- 必须披露的限制：其他市场起报/结算规则未冻结；未来天气缺发布时间元数据；2025/2026过去
  曾被旧协议查看，因此2025只能称为新协议冻结后的锁定回顾性测试。

## 15. 2026-10-08 强基线、本地环境与首批云任务冻结

### 15.1 本地Python环境

项目根目录虚拟环境已经确认可直接使用：

```text
D:\【科研】\electricity-price-forecasting\.venv\Scripts\python.exe
```

其中包含`xgboost 3.4.1`、`torch 2.12.0+cu126`、`pandas 3.0.6`、
`scikit-learn 1.9.1`、`pyarrow 23.0.1`和`pytest 9.1.1`。系统Python没有XGBoost；
`external/toto/.venv`也不是本项目正式运行环境。后续本地命令统一显式使用根目录`.venv`，
不再重复安装XGBoost。

### 15.2 强基线统一入口

新增：

- `src/decision_aware/baselines_global.py`：逐节点Seasonal、cutoff-safe LEAR和逐节点XGBoost。
- `scripts/decision_aware/run_global_baselines.py`：按inner/outer年度阶段运行三类基线并保存逐节点预测。
- `tests/test_decision_aware_global_baselines.py`：检查截止时点、防目标泄漏、滚动校准和指标。

LEAR使用公开方法的中位数/MAD加`asinh`稳健变换、AIC选正则强度和逐小时Lasso，但由于当前
Python 3.12环境与旧版`epftoolbox`依赖不兼容，这里是独立的cutoff-safe复现，不冒充某个
旧版本包的逐字节输出。每天预测完成后才允许把当天真实值加入后续校准窗。

2节点极小smoke已经连通：DA与RT-at-DA均通过Seasonal/LEAR/XGBoost，4小时滚动RT通过
Seasonal/XGBoost；这些数值样本太小，只能证明代码能跑，不能进入模型优劣结论。

完整回归测试在两个环境中均通过：

- 项目根目录`.venv`：`163 passed in 13.48s`。
- `external/toto/.venv`：`163 passed in 9.75s`。

### 15.3 首批云端低预算筛选

新增F0配置、机器可读清单和可断点运行脚本：

- `configs/decision_aware/global_ercot_huber_f0_v1.yaml`
- `configs/decision_aware/cloud_global_inner_seed0_screen_v1.json`
- `scripts/decision_aware/run_cloud_global_inner_seed0_screen.sh`

首批只跑seed 0、内部阶段和3个epoch：F0/F1乘以DA、RT-at-DA、滚动RT，共6个完全独立run。
每个epoch都验证完整2023年，按节点等权macro MAE选模。该批次只是低预算筛选与真实速度测量，
`formal_result=false`，不能作为最终结论。各run保留`latest.pt`，中断后可续训；GPU任务串行，
不会同时争抢显存。

云端预检只要求这批Huber训练实际使用的PyTorch、pandas和pyarrow，并核对ERCOT Parquet
SHA256；它不要求XGBoost。Linux脚本已由Git for Windows Bash完成`bash -n`语法检查；F0配置
也已通过正式合同校验和2节点CPU smoke。

AutoDL实例冻结为RTX 4090D 24GB、1.88元/小时。启动后先只运行预检和一个真实run测时，
再更新整批耗时/费用估计；未经该实测不直接放大到20 epoch或多seed。

## 16. 2026-10-08 首批云端筛选完成

F0/F1乘以DA、RT-at-DA、滚动RT的6个seed-0低预算run已经全部完成并下载到本地。
12个`best/latest` checkpoint均已通过本地CPU加载。按3 epoch内最佳2023 macro MAE，当前仅可暂记
`DA偏F0、RT-at-DA偏F1、滚动RT偏F0`；该结果不包含盈利目标、强基线、多seed或外层验证，
不能作为最终模型结论。完整数值、训练曲线、文件位置和限制见
`docs/progress/28_ERCOT多节点Global_F0F1低预算筛选结论.md`。

## 17. 2026-10-08 三元组盈利诊断与连续策略

已导出F0/F1乘以DA、RT-at-DA和滚动RT的6份完整2023预测，并按15节点、362个共同
交付日对齐真值。旧离散策略显示：`rt_only`与`plan_track_topk`的8组全部为负，主要原因是
3%容忍带之外的偏差罚金；`follow_da`全部为正，但没有真正使用滚动RT重调度。

随后实现“预测DA-RT-at-DA价差确定方向 + 连续联合日前LP确定功率 + 每小时4小时固定DA
连续RT LP/MPC + 只执行第一步”的策略。真实价格只在动作锁定后结算，偏差罚金、κ=5.7、
SOC、效率、每日放电上限和跨日连续状态均显式处理。

连续策略下，8组global中的冠军为`DA F1 + RT-at-DA F0 + RT F0`，节点等权日均净收益
`134.58 USD/节点·日`，15个节点全年平均均为正，SOC违规为0。Seasonal三任务同策略为
`213.78`；global减Seasonal为`-79.20`，7日块Bootstrap区间`[-177.46, 6.14]`跨0。
Seasonal均值更高，但宏观尾部10%和最大回撤分别为`-460.93`与`9733.25`，明显差于
global冠军的`-98.83`与`2685.83`。因此本轮只能写“global均值未超过Seasonal，但风险较温和”，
不能写稳定胜出或稳定落后。

本阶段完整方法、逐组合结果、失败尝试、测试和PRD余项见
`docs/progress/29_ERCOT多节点三元组收益与连续MPC诊断.md`。全仓测试在显式可写临时目录下为
`170 passed in 24.33s`；源Parquet哈希保持不变。

## 18. 2026-10-08 低收益归因与稳定训练v2准备

“Global”再次确认只表示同一任务跨15节点共享参数，DA、RT-at-DA和滚动RT仍是三个独立模型。
当前F1/F0/F0冠军的三项价格MAE均优于Seasonal，但预测DA-RT-at-DA价差被明显压缩，方向
正确率仅`48.72%`，而Seasonal为`57.98%`。Seasonal相对Global的全年收益差有`97.42%`
来自最大的10个优势日，说明均值差主要由少数稀缺尖峰日驱动；与此同时Seasonal尾部和回撤
明显更差。

已新增`global_screen_failure_diagnosis_v1.json`和可复现诊断脚本。训练流程已改为保留每路
macro MAE前2个候选，并要求最终三元组使用冻结完整结算选模；同时加入验证平台期降学习率、
修复resume早停状态、清理基线重复定义。稳定v2将旧结果隔离，使用`2e-4`初始学习率、最多
20 epoch、patience 7。2节点CPU smoke、shell语法和正式合同校验均通过，全仓测试更新为
`172 passed in 31.65s`。下一步在AutoDL执行6个F0/F1稳定seed-0任务，预计4090D约3.5—4.5小时。

## 19. 2026-10-08 稳定v2云端任务启动

- AutoDL：RTX 4090D 24GB，PyTorch 2.8.0+cu128，CUDA可用，数据盘剩余约49GB。
- ERCOT源Parquet SHA256：`43c2eb4b4c5c8ce467113f67744d9784ccb591e1febe0423642f363e7369c19d`，
  与本地冻结审计一致。
- 云端关键测试：`29 passed in 4.83s`。第一次测试发现云端`policy.py`仍是旧版；训练尚未开始时
  即停止，随后同步完整`src/decision_aware/*.py`并重测通过。
- 后台任务于北京时间`2026-10-08 17:18`启动；第一项为F0 DA inner seed 0。
- 运行脚本：`scripts/decision_aware/run_cloud_global_f0_f1_stable_v2.sh`。
- 总日志：`data/results/cloud_global_f0_f1_stable_v2/orchestrator.nohup.log`；独立日志与状态表在同目录。
- checkpoint：`data/checkpoints/global_ercot_huber_{f0,f1}_stable_v2/<task>_inner_seed0/`。
- 任务已由`nohup`托管，不依赖本地SSH或本地电脑；AutoDL实例必须保持开机。

## 20. 2026-10-08 稳定v2完成、回收与完整结算

6个稳定F0/F1 inner seed-0任务已全部完成并下载。45个云端/本地文件SHA256逐一一致。
每个任务从已保存候选中按macro MAE汇总后，总体保留2个，形成8个三元组；最终Global冠军为
`DA F1 epoch10 + RT-at-DA F0 epoch10 + RT F1 epoch8`。

冠军2023节点等权日均净收益为`192.83 USD/节点·日`，相对旧三轮冠军提高`58.25`；
15个节点全年平均全部为正，尾部10%为`-97.59`，最大回撤`1326.55`，SOC违规0。
Seasonal仍为`213.78`；Global减Seasonal为`-20.95`，配对7日块95%区间
`[-102.88, 52.41]`跨0。价差方向正确率由`48.72%`提高到`56.58%`，证明三轮欠训练
和学习率震荡确实是旧结果偏低的重要原因；但极端价差仍被平滑，尚未稳定超过Seasonal。

本阶段仍只有F0/F1历史结构和seed 0，只能用于内部候选晋级。下一步先补完整强基线与必要
候选的同策略评价，再决定每任务约前2名的seed 1/2；不能直接进入2024或2025。
本地最终回归测试为`173 passed in 43.26s`。

## 21. 2026-10-08 稳健XGBoost完整三任务强基线

已完成15节点DA、RT-at-DA和滚动RT的逐节点XGBoost。原始平方误差版本的三任务MAE为
`46.65 / 76.72 / 33.92`，连续MPC收益只有`55.54 USD/节点·日`，显著低于Seasonal。
问题主要来自尖峰RT价格下平方损失不稳健。

损失函数只在inner-train内部选择：2020—2021拟合、2022留出，2023不参与。最终冻结
DA和滚动RT使用绝对误差，RT-at-DA使用伪Huber。用完整2020—2022重训后，2023 MAE降至
`35.51 / 35.23 / 27.14`。

同一连续DA LP、滚动RT MPC、偏差罚金和退化成本合同下，稳健XGBoost达到
`292.29 USD/节点·日`，相对Seasonal提高`78.51`，配对7日块95%区间
`[18.58, 151.62]`不跨0；15节点年均全部为正，宏观尾部10%为`-29.78`，最大回撤
`206.07`，SOC违规0。该结果也高于稳定Global的`192.83`，因此稳健XGBoost是当前
2023内部验证冠军。完整方法与限制见第29号文档第15节；机器可读主结果为
`data/results/global_xgboost_robust_continuous_mpc_backtest_v1.json`。

这不是2024外层确认或2025锁定回顾测试。后续Global、local/global和decision-aware方法都必须
与此强基线在相同策略下比较；若不能超过，应保留该负结论。

本阶段全仓测试为`173 passed in 35.34s`；四市场源Parquet哈希与冻结审计逐一一致，源数据未改。

## 22. 2026-10-09 强基线差距定位与下一轮优化入口

逐任务替换诊断确认：相对全稳健XGBoost的`292.29`，只换Global滚动RT为`288.03`，
只换Global RT-at-DA为`252.30`，只换Global DA则降至`158.90`。所以DA是首要瓶颈，
RT-at-DA次之，滚动RT暂不优先重训。完整表见第29号文档第16节和
`data/results/global_task_replacement_diagnosis_v1.json`。

多节点三任务已经接入正式连续MPC的成对零阶收益梯度；真实checkpoint烟雾试验三路参数梯度
均有限且非零，三套模型仍不共享参数。另已给Global训练器加入稳健MAE微调和epoch 0原模型保护，
准备先运行DA F1稳健微调，再根据完整结算决定是否进入连续MPC收益微调。旧AutoDL连接地址已
失效，正式15节点GPU训练尚未启动；当前本地全仓回归为`179 passed`，源数据哈希未变。
