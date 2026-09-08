# Todo 4：V8 学习后执行路径

> 目的：先理解当前 v8 的真实实现，再用一个口径干净的 DA-only（只做日前）版本，判断 RT 联合训练是否真的有价值。
>
> 依据：W11 要求近期聚焦日前市场；当前 v8 仍同时训练 DA 与 RT，因此需要可比对照，而不是直接假定联合建模正确。
>
> 主线：先完成 DA（日前）模型与验证，再把 RT（实时）作为独立第二阶段；最后才验证二者在策略和结算层的协同价值。

## 为什么不能现在直接把 DA 和 RT 合并训练

DA 和 RT 都是电价，但不是同一个预测任务。DA 是日前固定时点一次性给出未来全天计划，价格相对平滑；RT 是日内短窗口滚动问题，更受拥塞和突发状态影响，尖峰更显著。两者的预测时点、当时可获得的信息、预测窗口、误差形态和调度用途不同。

当前 v8 的 `use_dual_split: true` 只是一个模型内的多分支：DA 与 RT 仍共享 5 流 encoder 和 CrossModalFusion，训练时还共同更新这些共享参数。其 DA 动作实际依赖 `p_da - p_rt_da`，RT 分支也通过 `p_rt` 与零阶代理损失影响训练。因此如果训练结果变好或变差，无法判断是 DA 模型本身、RT 预测、共享表示干扰、双结算口径，还是实现细节造成的。

先分开做的目的不是否认 RT 的价值，而是建立可解释的因果顺序：先证明“只靠日前信息能否做出可靠的日前计划”，再单独证明“RT 模型能否抓住尖峰并在受约束下改善该计划”，最后才把两者接到同一个策略/结算层。若 DA-only 已经稳定且收益不差，联合复杂度未必值得；若独立 RT 的增益可重复验证，再合并协同才有证据。

## 0. 先学 v8，不急着改代码

按下面顺序阅读，并从代码而不是 Drawio 图确认逻辑：

1. `configs/decision_aware/formal_ercot_v8.yaml`：确认 v8 的开关、数据、模型和 BESS 参数。
2. `src/decision_aware/dataset_v3.py`：确认一个 batch 中实际有哪些输入、目标和归一化字段。
3. `src/decision_aware/model.py`：理解 5 条输入流如何编码、融合，以及 `p_da`、`p_rt_da`、`p_rt`、`p_rt_windows` 四类输出。
4. `src/decision_aware/policy.py`：理解 HardTopK、BESS 的 SOC/循环限制、`forward_dual` 双结算收益。
5. `src/decision_aware/zero_order.py` 与 `src/decision_aware/loss.py`：理解不可微 HardTopK 如何通过零阶梯度与 `L_proxy` 训练。
6. `scripts/decision_aware/train_formal.py`：把数据、模型、损失和评估串成完整训练流程。

学习时要记住：`use_dual_split: true` 是一个联合模型内的 DA/RT 多分支，并不是 W11 所说的“两个独立模型”。

## 1. 固定当前 v8 的事实基线

先不改任何逻辑，记录当前 v8 的实际口径：

- 配置：`configs/decision_aware/formal_ercot_v8.yaml`。
- 数据：`data/unified/ERCOT_统一小时数据_20200101_20260601.parquet` 中的 ERCOT LZ_LCRA，1 小时粒度，168 小时上下文；当前实际读取 DA/RT、实际负荷、实际风光和本地日历。
- 模型：5 流输入；输出 48 小时 `p_da`、48 小时 `p_rt_da`、24 个 RT 滚动窗口及其首步信号。
- 决策：`u_da = HardTopK(p_da - p_rt_da)`；`u_rt` 由 RT 信号决定，并在启用偏差罚金时优先跟踪 `u_da`。
- 结算：DA 计划腿 + RT 偏差腿 - 退化成本 - 可选偏差罚金。
- 训练：Huber 预测损失 + 零阶 `L_proxy`；前 8 个 epoch 纯预测，之后逐步加入决策代理损失。

注意：当前代码实际实例化出的 v8 可训练参数量约为 10.71M；不要沿用旧文案中的 9.92M。

## 2. 先定义 DA-only 对照，不凭感觉删代码

目标是保持 v8 的模型质量修复不变，只移除 RT 联合任务。先写清楚下列边界，再实现：

- 运行时输入：只使用日前起报时可获得的历史数据和日前可用特征；不能使用未来 RT 真实值，也不能把未来实际负荷/风光或 `actual_*` 天气当作日前可知信息。
- Xweather 的数据条件已具备：统一表提供 8 列 HRRR t+24 预报天气；但现有 Dataset/Model 尚未使用。接入前必须先定义“每个 DA 起报时刻对应哪些 forecast run/lead、覆盖哪 24 个目标小时”，再做单独消融。统一表中 268 个预报缺失 decision hour 经线性插补，需加质量标记或进行剔除敏感性分析。
- 模型输出：只输出未来 24 小时 DA 价格 `p_da`。
- 运行时决策：只从 `p_da` 生成日前计划 `u_da`；不预测 RT，不根据 RT 信号主动修正动作。
- 模型结构：尽量保留 v8 的 5 流 encoder、modality embedding、CrossModalFusion、QueryDecoder、final LayerNorm 等架构修复，确保比较的核心变量是“是否引入 RT 联合任务”。
- 训练目标：只保留 DA 的预测损失和对应的 DA 决策代理损失。

DA-only 版本不等于忽略真实市场结算。需要分开报告两种结果：

- DA 计划收益：用于判断日前预测是否能生成有效日前套利计划。
- 真实结算诊断收益：若实际执行出现被动物理偏差，可使用事后真实 RT 价格计算影响；但不能让 RT 预测参与运行时决策。

## 3. 实现 DA-only 的最小对照配置

建议新建独立配置和训练入口，避免直接覆盖 v8：

- 新配置建议命名为 `configs/decision_aware/formal_ercot_v8_da_only.yaml`。
- 新训练入口可复用 `scripts/decision_aware/train_formal.py` 的数据、优化器、checkpoint 与日志框架。
- 需要验证模型输出只有 `p_da`，损失不再访问 `p_rt_da`、`p_rt`、`p_rt_windows`，策略也不再以价差或 RT 信号生成主动实时动作。
- 保持训练、验证、测试切分，batch size，优化器，BESS 约束和随机种子与 v8 相同。

不要在这一步同时改成 15 分钟粒度或 125kW/261kWh 设备参数。当前 v8 是 1 小时、24 点、1MW/4MWh；W11 的 15 分钟数据口径仍未解决。先隔离“DA-only vs DA/RT 联合”这一个变量。

## 4. 先跑小规模正确性检查

先用少量 epoch 或少量样本确认流程正确，再开始完整训练。检查：

- 输出 `p_da` 的形状是 `[B, 24]`。
- 训练和验证损失可正常下降，不出现 NaN/Inf。
- HardTopK 产生的动作满足预期方向：低价充电、高价放电。
- BESS 的 SOC 上下限、每日放电上限和退化成本都实际生效。
- LP Oracle、`R_model`、regret、PCR 的量纲和符号合理。
- 测试集不能泄漏未来数据；DA-only 运行时不得读取未来 RT。

## 5. 做可解释的完整对比

DA-only 与 v8 使用完全相同的数据切分、随机种子、评估窗口与 BESS 参数。至少并排比较：

- DA MAE、DA RMSE。
- DA 计划收益。
- 真实结算诊断收益。
- `R_model`、LP Oracle、regret、PCR。
- 训练稳定性，例如是否爆炸、验证指标是否剧烈波动、最佳 epoch。

判断规则：如果 DA-only 更稳定且收益不差，优先沿 W11 的日前主线推进；如果 v8 明显更优，继续做消融，确认收益来自 RT 联合信息而不是数据泄漏、结算口径或实现差异。

## 6. 明确的后续顺序：DA → RT → 协同

第一阶段完成 DA-only：只用日前起报时已知的信息预测 DA，生成并验证日前计划。这一步先回答“DA 本身是否足以支持可靠套利”。

第二阶段再做独立 RT：为 RT 单独定义起报时刻可见的数据、短期滚动预测窗口、尖峰指标和模型。RT 的目标是识别日内突发变化，并提出受 SOC、偏差罚金和计划跟踪约束限制的修正建议；它不能偷看未来 RT 真实值。

第三阶段最后做 DA/RT 协同：DA 模型提供基准计划 `u_da`，RT 模型只在独立验证后用于有限修正；统一策略与结算层用真实 DA/RT 价格评估增益，并与“严格执行 DA 计划”进行对照。只有 RT 修正带来的收益超过偏差罚金、额外复杂度和稳定性损失时，才保留协同方案。

15 分钟与真实设备参数迁移也应单独立项：先解决日前数据只有 1 小时、却要求 96 个点的颗粒度冲突，再迁移到 W11 的 125kW / 261kWh / SOC 0.05~0.95 参数。

## Drawio 图核对规则

以后查看 `docs/model_architecture.drawio` 或 `docs/training_loop.drawio` 的截图时，图只作为说明线索。以以下代码为准：

- 模型结构：`src/decision_aware/model.py`。
- 数据字段：`src/decision_aware/dataset_v3.py`。
- 策略与结算：`src/decision_aware/policy.py`。
- 损失与零阶梯度：`src/decision_aware/loss.py`、`src/decision_aware/zero_order.py`。
- 训练流程：`scripts/decision_aware/train_formal.py`。

若图、文档和代码不一致，记录差异，但不要为了贴合图而直接修改代码。当前 `docs/model_architecture.drawio` 的 5 流图与代码一致，不能因为统一表有天气字段就直接加出第 6 条 Weather 流；只有天气真实接入 Dataset、Model 与训练验证后，再同步修改 Drawio。
