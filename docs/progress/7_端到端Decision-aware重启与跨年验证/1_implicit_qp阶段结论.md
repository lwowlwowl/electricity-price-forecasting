# Decision-aware Implicit-QP 阶段结论

> 更新时间：2026-10-24  
> 状态：2023 开发阶段；尚未进入 2024 时间外确认。  
> 原则：完整记录正负结果；禁止挑选最好 seed；禁止利用 2024/2025 反向调参。

## 1. 固定评价口径

- 训练市场：CAISO、NYISO、PJM，完全排除 ERCOT 训练标签。
- 训练年份：2020—2022。
- 开发评价：ERCOT 2023，15 节点。
- 收益：节点等权宏平均 USD/节点·日。
- 正式结算：DA LP + rolling RT MPC，连续 SOC，计入退化成本和偏差罚金。
- 当前最强同数据公平基线：pooled LOMO XGBoost，`418.0648633`。
- Transformer 三 seed 固定等权集成：`407.1667076`。
- 所有候选必须使用相同精确结算器；训练代理收益不得替代正式收益。

完整基线定义见仓库根目录 `BASELINES.md`。

## 2. 本阶段方法

实现了强凸平滑电池 QP 与 KKT 隐式微分层。预测链路为：

```text
历史特征 -> DA / RT-at-DA / rolling RT 预测
         -> 可微电池 QP
         -> DA计划、RT动作、SOC与偏差
         -> 使用真实源市场价格计算收益
         -> 收益梯度穿过优化器返回预测网络
```

训练目标为价格锚点与负收益的加权和。训练 QP 仅作为可微代理；候选最终统一进入原精确 LP/MPC 结算器。

## 3. Seed 0 方法筛选

| 候选 | 可训练范围 | 收益 | 相对XGBoost | 改善节点 | 去最佳5日后的差异 | 配对7日块95%区间 |
|---|---|---:|---:|---:|---:|---|
| price-head balanced | price head | 421.8223 | +3.7575 | 11/15 | -4.5724 | [-9.34, 20.62] |
| price-head decision | price head | 423.3223 | +5.2574 | 13/15 | -3.1014 | [-8.19, 22.47] |
| price-head sharp | price head | 417.3823 | -0.6826 | 7/15 | -9.0633 | [-16.11, 16.68] |
| **decoder-balanced** | **decoder + price head** | **426.9884** | **+8.9235** | **14/15** | **+0.6451** | **[-3.88, 25.92]** |
| decoder-decision | decoder + price head | 414.8255 | -3.2394 | 2/15 | -11.6884 | [-21.18, 14.89] |

阶段判断：`decoder-balanced` 是 seed 0 下最强候选，点估计、节点覆盖和去极端日检查较好，但配对区间仍跨 0，不能宣称统计显著。

## 4. 冻结配置三 seed 验证

在查看 seed 1/2 收益前冻结 `decoder-balanced`：decision weight `0.60`、学习率 `2e-5`、decoder+head、二次正则 `0.25`、障碍温度 `0.10`、3 epochs、每节点每 epoch 2 episodes。三个 seed 使用相同微调采样种子 `20261023`；差异只来自原始 Transformer checkpoint。

| 系统 | 原始Transformer | Decision-aware后 | 相对XGBoost | 改善节点 | 去最佳5日后的差异 | 配对7日块95%区间 |
|---|---:|---:|---:|---:|---:|---|
| seed 0 | 377.8381 | **426.9884** | +8.9235 | 14/15 | +0.6451 | [-4.66, 25.38] |
| seed 1 | 266.1547 | 387.3219 | -30.7430 | 0/15 | -40.4029 | [-68.22, 0.60] |
| seed 2 | 403.0974 | 399.4066 | -18.6583 | 1/15 | -28.6958 | [-50.64, 10.46] |
| 固定等权集成 | 407.1667 | **424.7690** | +6.7041 | 13/15 | -2.0552 | [-9.52, 25.08] |

- 三 seed 收益均值：`404.5723`。
- 三 seed 样本标准差：`20.3315`。
- 固定集成点估计较 XGBoost 高 `1.60%`，但区间跨 0，去最佳 5 日后为负。
- 不允许只报告 seed 0；当前方法尚未满足稳定胜出条件。

## 5. 已确认的失效机制

1. 原始同架构 Transformer 本身具有很大的 seed 波动：`377.84 / 266.15 / 403.10`。
2. 三个微调运行的源市场训练收益轨迹接近，但 ERCOT 收益分化，说明源市场平均训练收益不能稳定代表目标市场零样本泛化。
3. 电池决策对小时价格排序敏感，小预测差异会翻转充放电动作并放大为收益差异。
4. 独立微调每个 seed 后再平均只能部分抵消波动；固定集成仍由少数高贡献日期支撑。

## 6. 下一阶段冻结路线

下一阶段只使用源市场证据开发“跨市场稳健 decision-aware 集成”，不再依据 ERCOT 2023 逐候选调参：

1. 对已有 seed 做预测、动作、收益分量和源市场梯度审计。
2. 以三个 seed 的预测集成为基础，训练共享的小型校准/残差层。
3. 损失同时约束源市场平均收益、跨市场收益方差/最差市场风险和价格锚点。
4. 模型选择只使用 CAISO、NYISO、PJM 的留出验证。
5. 冻结后仅做一次 ERCOT 2023 验收。
6. 只有通过点估计、配对区间和去极端日检查后，才冻结进入 2024 时间外确认。

## 7. 主要证据产物

- `data/results/global_implicit_qp_summary_v1.json`
- `data/results/global_implicit_qp_multiseed_summary_v1/summary.json`
- `data/results/global_implicit_qp_multiseed_summary_v1/summary.md`
- `data/results/global_implicit_qp_multiseed_continuous_mpc_v1.json`
- `data/results/global_implicit_qp_run_v1/status.tsv`
- `data/results/global_implicit_qp_multiseed_run_v1/status.tsv`
- `src/decision_aware/implicit_battery_qp.py`
- `scripts/decision_aware/train_global_joint_implicit_qp.py`

