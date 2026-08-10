# 正式版 TODO（todo3_2）

> 日期：2026-08-05（v7 代码完成后更新）
> 关联：`docs/正式版实验结果.md`、`docs/team_notes/w10/decision_aware_multimodal_tsfm_modeling(3).pdf`
> 基于：w10 原文逐节对照 + v1-v6 实验发现 + v7 代码改动
> v7 配置：`configs/decision_aware/formal_ercot_v7.yaml`

---

## 四类分类

### 一、w10 要求 + 不需要云 GPU

#### ✅ 已做对（v7 完成）

| # | w10 要求 | 出处 | 现状 |
|---|---|---|---|
| 1 | 双点零阶高斯梯度 ĝ | §6.1 | ✅ |
| 2 | ε = ρ·σ_train（每任务独立 ε_m）| §6.1 | ✅ v7：ε_DA=ρ·σ_DA，ε_RT=ρ·σ_RT |
| 3 | α-β 退火 + 预训练 | §6.2 | ✅ |
| 4 | TopK/BotK 候选选择 + 价差门控 | §4 | ✅ v7：HardTopKPolicy 加 spread_threshold（自动 κ/η）|
| 5 | SOC 约束 s_min ≤ s ≤ s_max | §4 | ✅ 在 simulator 里 |
| 6 | BESS 物理参数（P/E/η/SOC/κ/E_cyc）| §7 | ✅ v7：加 E_cyc=4 |
| 7 | 数据按时序划分 | §7 | ✅ |
| 8 | TSFM 编码历史价格 + 适配层 + 融合 | §3 | ✅ |
| 9 | DA/RT 分离决策 forward_dual | §4.3 | ✅ v7：训练时零阶梯度也调 forward_dual |
| 10 | 48h 联合预测（pDA + pRT\|DA 两条曲线）| §2 | ✅ v6 实现 |
| 11 | fD 输出双曲线 + 预测损失 LD | §3 | ✅ v7：支持 MSE/Huber 切换（默认 Huber，MSE 在尖峰价上数值爆炸）|
| 12 | L_proxy 分三项扰动（用 forward_dual）| §6.2 | ✅ v7：三项都走双结算 |
| 13 | DA 策略用价差 d̂=p̂DA−p̂RT\|DA | §4.1 | ✅ v7 |
| 14 | LP Oracle 双结算 | §5.2 | ✅ v7：lp_oracle_revenue_dual（DA腿+RT腿）|
| 15 | 日循环上限 E_cyc + q_t | §4.2 | ✅ v7：forward/forward_dual/LP 都加 |
| 16 | 48h SOC 连续 + 计划/实际 SOC 分离 | §4.3 | ✅ v7：forward_dual 重写，DA 48h 计划 SOC + RT 24h 实际 SOC |
| 17 | 偏差罚金（3% 容忍 + 双倍 RT 价）| §5.1 | ✅ v7：config 开启 use_deviation_penalty |
| 18 | 计划跟踪优先规则 | §4.3 | ✅ v7：plan_track_override |
| 19 | RT 滚动预测 24 窗口×H=4 | §2/§3 | ✅ v7：rt_decoder + head_rt_windows [B,24,4]（一次性前向简化版，完整逐小时重预测待云 GPU）|
| 20 | 预测损失 LR = 24 窗口取平均 | §3 | ✅ v7：unfold 构造 24×H 真实窗口 |
| 21 | fD/fRT 结构分离 | §3 | ✅ v7：RT 独立 decoder（信息截止完整分离待云 GPU）|
| 22 | PCR 在 Oracle 非正时不报告 | §7 | ✅ v7：≤0 报 N/A |
| 23 | RMSE 指标 | §7 | ✅ v7：metrics + 训练日志 + test 报告 |

#### ⚠️ 做了但不完整（本地简化版，完整版待云 GPU）

| # | w10 要求 | 出处 | 差什么 | 优先级 |
|---|---|---|---|---|
| 1 | 完整 H=4 滚动预测（每小时重预测+只执行第一个动作）| §2 | v7 是一次性前向出 24 窗口（结构合规、LR 公式合规），但非每小时用更新信息重新预测。完整版每 batch 24 次前向，48h/次 | 中（待云 GPU）|
| 2 | fD/fRT 信息截止完整分离 | §3 | v7 RT 用独立 decoder 但共享上下文。完整分离需逐小时不同截止时点 | 中（待云 GPU）|

#### ❌ 还没做（不需云 GPU）

| # | w10 要求 | 出处 | 差什么 | 优先级 |
|---|---|---|---|---|
| 1 | 负荷/风光用预报版 | §2 | 当前用实际值，w10 要求预报版 | 中 |
| 2 | 统一输入 X 六模态 | §2 | 缺 Xweather 和 Xnews | 中 |
| 3 | 温度协变量（Xweather）| §2 | XGBoost 证明 MAE 降 0.37，需拉 2020-2026 数据 | 中 |
| 4 | 日前信息截止时点逻辑 | §2 | 没实现时点截止 | 低 |
| 5 | 最少 4 模型对比 | §7 | 有 B 对比（vs foundation），但没有标准 TSFM 和 XGBoost | 低 |
| 6 | 标准 TSFM 消融（β=0）| §7 | pretrain 是 β=0 但没作为消融对比报告 | 低 |
| 7 | 退化成本敏感性 {10,25,50} | §7 | 只用了 27 | 低 |
| 8 | 多模态输入消融 | §7 | XGBoost 做了，TSFM 没做 | 低 |
| 9 | 消融：K 值 / 策略参数 / 偏差罚金开关 | §7 | 没做 | 低 |
| 10 | ρ 搜索 {0.02,0.05,0.1,0.2} | §7 | 只试了 0.05 | 低 |
| 11 | β 搜索 {0,0.01,0.05,0.1,0.5} | §7 | 只试了退火到 0.5 | 低 |
| 12 | K 搜索 {2,4} | §7 | 只试了 2 | 低 |
| 13 | XGBoost 决策感知路径（三回归器+G_j/H_j）| §6.3 | 当前只做了协变量筛选 | 低 |
| 14 | 实施顺序：先 XGBoost → 策略测试 → 零阶 → TSFM | §7 | 直接做了 TSFM | 低 |
| 15 | 新闻协变量（Xnews）| §2 | GDELT 只有 2025-2026，扩展到 2020 需 1-2 天 | 低 |
| 16 | A 方案消融（等参基线 vs DA）| — | 论文用 | 低 |

---

### 二、w10 要求 + 需要云 GPU

| # | w10 要求 | 出处 | 说明 | 需要云原因 |
|---|---|---|---|---|
| 1 | 多节点联合训练（15 节点 650K 样本）| todo3 | 单节点 43K 样本，本地能跑但 15 节点需大内存+长训练 | 650K 样本×100M 参数 |
| 2 | 模型放大 d=512/4层/~100M | todo3 | 4.36M 本地能跑，100M 本地 15 小时但调参成本高 | 100M 参数训练慢 |
| 3 | 完整 H=4 滚动预测训练 | §2 | v7 简化版已合规（结构+LR 公式），完整版每 batch 24 次前向，训练慢 24 倍 | 48 小时/次 |
| 4 | fD/fRT 信息截止完整分离 | §3 | v7 结构分离已做，完整逐小时不同截止时点需逐小时前向 | 同上 |

---

### 三、后续发现 + 不需要云 GPU

| # | 改动 | 依据 | 预期效果 | 工作量 |
|---|---|---|---|---|
| 1 | 减小 K（4→2）| 退化成本 216 > Oracle 164，少做少亏 | 低波动日也能赚 | 改 config 一行 |

---

### 四、后续发现 + 需要云 GPU

| # | 改动 | 依据 | 说明 |
|---|---|---|---|
| 1 | 100M + 多节点验证"赚得多" | 4.36M B 对比已赢，100M 预期更优 | 需云 GPU |
| 2 | 多节点泛化性验证（NYISO）| w10 默认 NYISO，可做跨市场验证 | 需云 GPU + NYISO 数据 |

---

## 执行顺序

### 不需要云 GPU（第一步到第五步，本地 M3 Pro 完成）

```
✅ 第一步：口径问题 + 价差门控（v7 已完成）
  → DA 策略改用价差 d̂=p̂DA−p̂RT|DA ✅
  → LP Oracle 改双结算 ✅
  → 零阶梯度改用 forward_dual ✅
  → ε 分任务独立（DA/RT|DA/RT 各用各自 σ）✅
  → TopK 加价差门控（价差覆盖成本才保留候选）✅

✅ 第二步：偏差罚金 + 计划跟踪（v7 已完成）
  → 开启 use_deviation_penalty ✅
  → 加计划跟踪优先规则 ✅

✅ 第三步：架构完善（v7 已完成，完整逐小时滚动待云 GPU）
  → 循环约束 E_cyc + q_t ✅
  → 48h SOC 连续 + 计划/实际 SOC 分离 ✅
  → fD/fRT 结构分离 ✅（信息截止完整分离待云 GPU）
  → 预测损失改 MSE + 除以 96 ✅
  → RT 滚动 24 窗口×H=4 ✅（一次性前向简化版，完整逐小时重预测待云 GPU）
  → LR = 24 个滚动窗口取平均 ✅
  → RMSE 指标 + PCR 非正时不报告 ✅

⬜ 第四步：数据（未做）
  → 温度协变量（Xweather）
  → 负荷/风光用预报版
  → 新闻协变量（Xnews，需扩展 GDELT 到 2020）

⬜ 第五步：消融+调参+XGBoost（未做，全部本地）
  → ρ 搜索 {0.02,0.05,0.1,0.2}
  → β 搜索 {0,0.01,0.05,0.1,0.5}
  → K 搜索 {2,4}
  → 标准 TSFM 消融（β=0）
  → 退化成本敏感性 {10,25,50}
  → 多模态输入消融
  → 消融：K值/策略参数/偏差罚金开关
  → XGBoost 决策感知路径（三回归器+G_j/H_j）
  → A 方案消融（等参基线 vs DA）
  → 最少 4 模型对比（标准/DA TSFM + 标准/DA XGBoost）
  → 日前信息截止时点逻辑
```

### 需要云 GPU（第六步，等本地全部改完后上云）

```
⬜ 第六步：上云 GPU
  → 完整 H=4 滚动预测（每小时重新预测 4h，只执行第一个动作，训练慢 24 倍）
  → fD/fRT 信息截止完整分离（逐小时不同截止时点）
  → 模型放大 d=512/4层/~100M
  → 多节点联合训练（15 节点 650K 样本）
  → NYISO 泛化验证
```

---

## v7 完成的改动清单

| # | 项 | 文件 | 改动 |
|---|---|---|---|
| 1 | DA 用价差 d̂ | `loss.py` | uDA = policy_hard(p_da − p_rt_da) |
| 2 | LP Oracle 双结算 | `policy.py` | `lp_oracle_revenue_dual`（DA腿 spread 无κ + RT腿 pRT 有κ）|
| 3 | ZO 用 forward_dual | `zero_order.py`+`loss.py` | `estimate_zo_gradient_dual` + action_fn 闭包 |
| 4 | ε 分任务 | `train_formal.py`+`loss.py` | eps_da=ρ·σ_DA，eps_rt=ρ·σ_RT |
| 5 | TopK 价差门控 | `policy.py`+`config.py` | HardTopKPolicy spread_threshold（自动 κ/η）|
| 6 | 预测损失 MSE + /96 | `loss.py` | `half_se_loss` + `use_mse_loss` |
| 7 | RT 滚动 24×H=4 | `model.py`+`loss.py` | rt_decoder + head_rt_windows [B,24,4] |
| 8 | LR 24 窗口平均 | `loss.py` | unfold 构造真实窗口 |
| 9 | fD/fRT 结构分离 | `model.py` | RT 独立 decoder |
| 10 | E_cyc + q_t | `policy.py` | forward/forward_dual/LP 全加循环约束 |
| 11 | 48h SOC 连续 + 分离 | `policy.py` | forward_dual 重写：DA 48h 计划 SOC + RT 24h 实际 SOC |
| 12 | 偏差罚金 | `v7.yaml` | use_deviation_penalty: true |
| 13 | 计划跟踪 | `policy.py`+`loss.py` | `plan_track_override` |
| 14 | PCR 非正不报告 | `train_formal.py` | ≤0 报 N/A |
| 15 | RMSE | `loss.py`+`train_formal.py` | metrics + 日志 |
| 16 | E_cyc config | `config.py`+`v7.yaml` | bess_e_cyc: 4.0 |

---

## 【架构缺陷】模型架构问题清单

> 来源：两个 agent 交叉审查 `src/decision_aware/model.py` 后合并去重
> 日期：2026-08-05
> 涉及文件：`src/decision_aware/model.py`、`src/decision_aware/config.py`、`configs/decision_aware/formal_ercot_v7.yaml`
> 验证：v7 dual_split / v3 non-dual_split / v1-v2 向后兼容三条路径 forward + backward + AMP 全通过
> 参数量：4.36M → 9.92M，本地 M3 Pro 可跑

### 🔴 必改（会让模型学错）

| # | 状态 | 问题 | 说明 | 改法 |
|---|------|------|------|------|
| N1 | ✅已改 | Pre-LN 缺 final LayerNorm | Pre-LN 残差主干不归一化，加层后会训练不稳定。已加在 StreamEncoder/CrossModalFusion/QueryDecoder 每个栈出口 | 每个出口加 `nn.LayerNorm(d_model)`，B1 加层前必须先做 |
| A1 | ✅已改 | 融合层开了 RoPE | 840 token 是 5 流拼接的，RoPE 把同时刻不同流的 token 当成距离很远，抑制跨流注意力 | `CrossModalFusion.__init__` 中 `use_rope=False` |
| A2 | ✅已改 | 5 流 concat 无 modality 信号 | 模型不知道哪些 token 属于哪条流。计算量是 25 倍（O(N²)）不是 5 倍 | 加 `self.modality_emb [n_streams, d]` 到每条流所有 token。patching（降 16× 计算量）未做，属可选优化，留到 100M 时评估 |

### 🟠 建议改（限制上限）

| # | 状态 | 问题 | 说明 | 改法 |
|---|------|------|------|------|
| B1 | ✅已改 | 编码器只有 1 层 | 1 层只能"看一眼全局"。v7 已设 `n_layers_enc=2`，4 个 transformer 流×2 层 | `StreamEncoder.blocks = ModuleList(...)`，层数可配（依赖 N1 先做） |
| B2 | ✅已改 | Decoder 缺 query 间 self-attn | 24 个 query 互不知道对方，预测曲线容易跳变。RT 滚动窗口更严重 | cross-attn 前加 `self.q_self = RotaryMHA(..., use_rope=True)`，RT decoder 同受益 |
| B3 | ✅已改 | query 无位置信息，也不条件化于输入 | 24 个 query 随机初始化、全局共享、与样本无关，pretrain 期间效率低 | 加 `query_pos` 位置编码 + `ctx_proj(mean(memory))` 输入条件化 |
| B4 | ✅已改 | RT 复用 DA 前 4 个 query token（非 dual_split） | 4 个 token 同时预测 DA 和 RT，任务冲突。v7 dual_split 不受影响 | 非 dual_split：`n_queries = horizon_da + horizon_rt`，RT 用后段 |
| B5 | ⬜未改 | DA 和 RT 共享同一份 memory | 共享 fusion 输出可能限制各自发挥。但 w10 §3 共享编码器是合规设计，真正的分离是信息截止时点不同 | 待云 GPU，与 ⚠️#2 信息截止分离合并处理 |

### 🟡 可选改（一致性与细节）

| # | 状态 | 问题 | 说明 | 改法 |
|---|------|------|------|------|
| C1 | ✅已改 | 融合层单头 | 跨模态融合最该用多头。`config.py` 默认 1→4，v7=4 | `n_heads_fusion: 4` |
| C2 | ✅已改 | GRU 流无残差 | `x = x + out`。v3 已无 GRU，仅 v1/v2 生效 | `StreamEncoder.forward` gru 分支加残差 |
| C3 | ✅已改 | CrossModalFusion 冗余 LayerNorm | 先给各流出口加了 `final_norm` 统一尺度，再删 fusion 的 `self.norm`。顺序不能反 | 删 `self.norm`，直接 `self.block(tokens)` |
| C4 | ✅已改 | Calendar 用纯 Linear | Linear 学不了 hour 与价格的非线性周期关系。v3/v1-v2 都改了 | `kind="linear"` → `kind="mlp"` |
| C5 | ✅已改 | GRU 和 Transformer 输出不一致 | v3 system 流已从 GRU 换 Transformer。v1/v2 保留 GRU 向后兼容 | `enc_system = StreamEncoder(2, d, "transformer", ...)` |
| C6 | ✅已改 | GRU 不兼容 AMP 混合精度 | v3 已无 GRU 自动失效。v1/v2 保留 fp32 workaround | 换 Transformer 后自动消失，v1/v2 保留兼容 |

### 非架构缺陷（性能优化）

| 问题 | 状态 | 说明 |
|------|------|------|
| BESS forward Python for 循环 | 不改 | 可用 torch.cumsum 向量化，属实现优化 |
| LP Oracle 逐样本 for 循环 | 不改 | 可用 scipy.sparse 批量化，属实现优化 |
| N3 预测损失不约束价差 d̂ | 不改 | L_pred 分曲线 MSE 符合 w10 §3 规范，不是 bug。建议（未实施）：加价差正则项 λ·MSE(p̂DA−p̂RT|DA, pDA−pRT)，留待调参阶段 |

---

## 【代码正确性 bug】交叉审查发现（2026-08-06）

> 来源：另一个 agent 审查 `loss.py` / `policy.py` / `train_formal.py` 后提出 12 条问题
> 逐条验证后：4 条是真实 bug（已修），2 条是死代码（1 条已删），6 条不构成问题
> 涉及文件：`src/decision_aware/loss.py`、`src/decision_aware/policy.py`、`scripts/decision_aware/*.py`

### 🔴 已修复的真实 bug

| # | 状态 | 问题 | 代码位置 | 说明 | 修法 |
|---|------|------|---------|------|------|
| D1 | ✅已改 | **L_proxy 对 p̂RT\|DA 和 p̂RT 的梯度被 detach 截断**（致命）| `loss.py:171-177,209,218` | `p_rt_da_full` 和 `p_rt_24` 在传入 `compute_l_proxy_scaled` 前被 `.detach()`，而 `compute_l_proxy_scaled = (p_hat * grad.detach()).mean()`——p_hat 若已 detach 则整项无 grad_fn，梯度恒为零。结果：dual_split 模式下只有 p̂DA 收到决策感知梯度，p̂RT\|DA 和 p̂RT 两条曲线的 L_proxy 项完全失效，只靠 L_pred 训练。**RT 决策感知完全失效**。验证方法：β=1/α=0 + 大 ε 下 `head_rt_action.weight.grad` 为 None | 保留有梯度的原版 `p_rt_da_orig`/`p_rt_24_orig` 给 `compute_l_proxy_scaled`，detach 版只给 action_fn 闭包（闭包在 no_grad 下跑，用哪个都一样）。ZO 估计内部已 detach，传原版无害 |
| D2 | ✅已改 | `lp_oracle_revenue`（单结算 Oracle）收益没减 κ | `policy.py:316` | LP 目标函数 `c` 含 κ（优化出的 dis/chg 考虑了 κ），但算最终收益时 `np.sum((dis-chg)*p)` 没减 `κ·Σ(dis+chg)` → R* 被高估、regret 被放大。对比 `_lp_revenue_one` 第 360 行就写对了。仅影响非 dual_split 路径（v7 dual_split 走 `lp_oracle_revenue_dual` 不受影响） | 直接复用 `_lp_revenue_one`（已正确含 κ 和 E_cyc），删除重复的 LP 求解代码 |
| D3 | ✅已改 | `lp_oracle_revenue`（单结算 Oracle）缺 E_cyc 循环约束 | `policy.py:284-306` | 只建了 SOC 上下限约束，没建 `Σ dis_t ≤ E_cyc` → Oracle 可超日循环上限放电 → R* 进一步高估。同 D2，仅影响非 dual_split 路径 | 同 D2：复用 `_lp_revenue_one` |
| D4 | ✅已改 | `train_formal.py` 等 6 处没传 `e_cyc=cfg.bess_e_cyc` | `train_formal.py:120` 等 | `BESSSimulator` 构造默认 `e_cyc=energy_mwh`。当前 `bess_energy_mwh=4.0` 和 `bess_e_cyc=4.0` 恰好相等，碰巧没出问题——但改任一个参数 E_cyc 就静默失效。这是 latent bug | 所有 6 处 BESSSimulator 构造补 `e_cyc=cfg.bess_e_cyc`：`train_formal.py`、`eval_v3_da_oracle.py`、`compare_baselines_formal.py`、`train_pilot_v3.py` |

### ✅ 死代码清理

| # | 状态 | 问题 | 代码位置 | 说明 | 处理 |
|---|------|------|---------|------|------|
| D5 | 不改 | `BESSSimulator.forward` 中 `prt_t` 赋值但未使用 | `policy.py:75` | `prt_t = price[:, t]` 赋了值，但收益公式只用 `pda_t`。`forward()` 是单结算模式（price_da 传入时用 DA 价），RT 价确实不该参与——逻辑正确，只是多了个死赋值。不影响正确性 | 保留（无害，删了反而让 reader 困惑"为什么没有 RT 价变量"） |
| D6 | ✅已删 | `forward_dual` 的 plan SOC 轨迹是死代码 | `policy.py:117-132` | `soc_plan`/`q_plan` 算了 48 步 Python 循环，但从不参与收益计算（第 134 行起的结算用 `soc_act`/`q_act` 从 s0 重新起跑）。48 步循环白跑，浪费 ~40% `forward_dual` 时间。w10 §4.3 的"计划 SOC 检查提交可行性"当前未实现约束/罚金 | 删除整段 plan SOC 循环。若后续需要可行性检查可重新加入。实测 `forward_dual` 1.4ms/call |

### ❌ 不构成问题（逐条说明原因）

| # | agent 主张 | 为什么不构成问题 |
|---|-----------|----------------|
| D7 | CUDA 下 AMP 没用 GradScaler | **正确但仅影响 CUDA（云 GPU）**。MPS 不需要也不支持 `GradScaler`，当前本地训练不受影响。上云到 CUDA 时再加 `torch.cuda.amp.GradScaler`。归入「需要云 GPU」类别 |
| D8 | 没有学习率调度器 | **是优化建议，不是 bug**。常数 lr 对 10M 小模型可工作。Transformer 放大到 65M 时建议加 warmup+cosine，但属调参范畴。todo3_2 第五步「调参」可包含 |
| D9 | K=2 零阶梯度方差高 | **是超参数选择，不是 bug**。w10 §7 明确 K∈{2,4}，todo3_2 第五步已列「K 搜索 {2,4}」。K=2 是下界但 w10 允许 |
| D10 | HardTopKPolicy Python for 循环 | **性能优化，非正确性问题**。已在上方「非架构缺陷」表列出。向量化 topk+scatter 有难度，当前 64×4=256 次循环/step 可接受 |
| D11 | train.py 与 v3/dual_split 不兼容 | **是设计如此，非 bug**。`train.py` 是先行版（v1/v2/v3）入口，用 `batch["price_tgt"]` 和 `total_loss`（旧接口）。`train_formal.py` 是正式版入口。两者分工明确，v7 配置应用 `train_formal.py`。可加 assert 但优先级低 |
| D12 | `modality_emb` 初始化偏小（×0.02） | **是超参数选择，非 bug**。d=256 时每个 emb 范数 ≈0.32，LayerNorm 后 token 范数 ≈16，modality 信号占 ~2%。模型可通过学习放大。若验证发现流区分不明显可调大到 0.1，但当前不构成正确性问题 |

### 验证

| Fix | 验证方法 | 结果 |
|-----|---------|------|
| D1 | β=1/α=0 + 大 ε(50) → 检查 `head_rt_action.weight.grad` | ✅ 从 None → 0.87（梯度正常流过） |
| D2 | κ=27 的 R* ≤ κ=0 的 R* | ✅ 48.7 ≤ 133.8 |
| D3 | e_cyc=0.5 的 R* ≤ e_cyc=4 的 R* | ✅ 30.1 ≤ 133.8 |
| D4 | `sim.e_cyc == cfg.bess_e_cyc` | ✅ 4.0 == 4.0 |
| D6 | `forward_dual` 输出 shape + 无 NaN | ✅ [B] 无 NaN，1.4ms/call |

---

## 【代码正确性 bug 第二轮】交叉审查发现（2026-08-06）

> 来源：另一个 agent 第二轮审查 `loss.py` / `policy.py` / `zero_order.py` 后提出 3 条问题
> 逐条验证后：3 条全部是真实 bug（已修）
> 涉及文件：`src/decision_aware/zero_order.py`、`src/decision_aware/loss.py`、`src/decision_aware/policy.py`

### 🔴 已修复的真实 bug

| # | 状态 | 问题 | 代码位置 | 说明 | 修法 |
|---|------|------|---------|------|------|
| E1 | ✅已改 | **L_proxy `.mean()` 使 DA 项梯度比 w10 小 48×** | `zero_order.py:100` `compute_l_proxy` | w10 §6.2：DA/RT\|DA 项是点积 `(p̂DA)^T ĝ^DA`（不除 horizon），RT 项是 `(1/24)Σ`（除 24）。代码统一用 `.mean()` = `Σ/(B·H)`：DA 项（H=48）被多除 48 → 梯度小 48×；RT 项（H=24）恰好匹配。`proxy_scale=80` 是全局标量，无法修正 DA/RT 相对权重。验证：code/w10 梯度比 DA=0.0208（=1/48），RT=1.0000 | `compute_l_proxy` 加 `per_sample_sum` 参数：DA/RT\|DA 项用 `per_sample_sum=True`（per-sample 点积再 mean over batch），RT 项保持默认 `.mean()`（匹配 w10 的 1/24）|
| E2 | ✅已改 | **LP Oracle 不含偏差罚金**（v7 开了罚金但 Oracle 没扣）| `policy.py:348-353` `lp_oracle_revenue_dual` | `lp_oracle_revenue_dual` 解两个独立 LP 相加，偏差罚金 `P_dev = Σ 2|pRT|·[|Δu|−0.03|uDA|]+` 耦合 uDA/uRT，独立 LP 无法表达。v7 `use_deviation_penalty=true`，模型 `forward_dual` 正确扣了罚金，但 Oracle R* 没扣 → R* 虚高、regret 虚高、PCR 虚低。实测：Oracle=280 vs 可达−125（虚高 405）。**分析**：罚金倍数 2×\|pRT\| 使偏差收益 `Δu·pRT − 2|pRT|·|Δu| ≤ −|pRT|·|Δu| < 0`（任意 Δu 符号），即偏差永远不划算 → 最优 uRT=uDA → R* 退化为单结算 LP `max Σ pDA·uDA − κ|uDA|`。这与模型 plan_track 行为一致，是公平的真上界 | `lp_oracle_revenue_dual` 加 `use_deviation_penalty` 参数：True 时走单结算 LP（DA 价），False 时走原双 LP。`loss.py` 调用处传 `use_dev_penalty` |
| E3 | ✅已改 | **forward_dual RT 偏差腿用未裁剪的 uRT 意图值** | `policy.py:131,139-141,155-157` | `delta_u = urt_t - uda_t` 用原始意图值，但实际执行 `d_act/c_act` 被 SOC 裁剪（§4.2）。市场按实际计量值结算偏差。实测：uRT=+1（放电）但 SOC 空→全 clip 到 0，代码仍算 `rt_leg=2400`（应为 0）。SOC 裁剪越频繁错误越大 | rt_leg 和 penalty 改用 clipped actual：`actual_rt_net = d_act - c_act`（已裁剪净放电），`delta_actual = actual_rt_net - da_net`，`rt_leg = delta_actual · prt_t`，penalty 阈值/超额也用 actual |

### 验证

| Fix | 验证方法 | 结果 |
|-----|---------|------|
| E1 | DA/RT proxy 梯度比 head_da/head_rt_action | ✅ 修复前 DA/RT≈0.02（48× 偏小），修复后 155（DA 主导，符合 w10 设计） |
| E2 | `lp_oracle_revenue_dual(pen=True) ≤ (pen=False)` | ✅ 48.7 ≤ 247.8 |
| E3 | SOC 空 + uRT=放电 → rt_leg 应为 0 | ✅ 修复前 2400，修复后 0.00 |

### 与第一轮（D1-D12）的关系

- **E1 与 D1 互补**：D1 是 RT 项梯度被 detach 完全归零（已修），E1 是 DA 项梯度被 .mean() 缩小 48×（本轮修）。两者叠加时 DA 项实际梯度 = `ĝ^DA/48`（D1 让 RT=0，E1 让 DA 偏小）。D1 修复后 RT 恢复，E1 修复后 DA 放大 48× 到正确尺度。
- **E2 独立**：Oracle 不含罚金，与 D2/D3（单结算 Oracle 缺 κ/E_cyc）是不同函数的不同 bug。
- **E3 独立**：forward_dual 的 RT 偏差腿用意图值，与 D6（plan SOC 死代码）在同一函数但不同问题。