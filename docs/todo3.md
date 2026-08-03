# 模型搭建决策清单（todo3）

> 日期：2026-07-22（2026-07-31 根据 w10 PDF 全面更新）
> 状态：先行版 v1/v2/v3 实验完成，正式版规划对齐 w10
> 关联文档：w10/decision_aware_multimodal_tsfm_modeling(3).pdf（数学建模规范）、w9/模型相关.md、w8/研究方案.md、w8/two_settlement_bess_external_model.pdf、docs/model_architecture_v2.drawio

---

## 一、训练回路层面（端到端可微）

### 1. 策略 π 的实现方式 ⭐ 最高优先级

端到端可微的核心瓶颈：策略 π 把预测价格 p̂ 变成充放电动作 u，这一步涉及离散判断，导数为零。

| 选项 | 方案 | 优点 | 缺点 | 依赖 |
|------|------|------|------|------|
| ~~A~~ | ~~Greedy + 代理梯度~~ | ~~实现简单~~ | ~~无理论保证~~ | ~~无~~ |
| B | 凸优化层(LP) + 扰动法 | 理论干净 | 每 step 多一次 LP | cvxpy/Gurobi |
| **w10** | **TopK/BotK + 零阶双点高斯梯度** | w10 规范，有理论引用 [7,8] | 每 step 跑 K 次策略仿真 | 无（scipy 足够）|
| 先行版实际 | STE（先行版 v1/v3-ste）/ soft TopK（v2/v3）| 实现简单，无需额外仿真 | 近似梯度，无理论保证；soft TopK 在高波动 DA 价上会爆炸 | 无 |

决策：**正式版用 w10 的 TopK + 零阶双点梯度**（Nesterov-Spokoiny 高斯平滑，w10 第 6 节）。
先行版实证：
- v1 STE 占 LP oracle 52%（先行版最优）
- v2/v3 soft TopK 占 33% / -77%（soft TopK 正反馈爆炸，验证了 w10 用零阶梯度的必要性）
参考论文：Nesterov & Spokoiny [7], Duchi et al. [8]（w10 参考文献）

### 2. Loss 配比 α 和 β

L = αL_pred + βL_bus

决策：退火策略（B）：前期 α=1,β=0 → 逐步 α→0,β→1。
先行版实测：**β≈0.8 最优**（不是 β=1）；β=1 平台期或恶化。monitor=val regret 自动选 ckpt。
w10 第 6.2 节：TSFM 用 L = αL_pred + βL_proxy，其中 L_proxy 是零阶梯度注入的局部代理。

### 3. L_pred 的具体损失函数

决策：先行版用 Huber，正式版可换 Quantile Loss。
w10 第 3 节：日前阶段同时约束两条 48h 曲线（pDA + pRT|DA），实时阶段对 24 个滚动窗口取平均。

### 4. 双结算收益公式 ⭐ w10 新增

w10 第 5 节定义：
```
R_base = Δt·Σ_t ( pDA_t · uDA_t + pRT_t · Δu_t − κ|u_t| )
```
- 日前腿按 DA 价结算、实时偏差按 RT 价结算、扣退化成本 κ
- Δu = uRT − uDA（实时实际动作相对日前计划的偏差）

先行版实现：
- v1/v2：退化版（只有 RT 价，无 DA 价，无 κ）→ 两腿都用 realized RT
- v3：简化版（有 DA 价 + κ，但 uDA=uRT，不分离决策）→ DA 腿用全量，无 RT 偏差腿

**正式版需补**：DA/RT 分离决策（先报 uDA 计划，实时再纠偏 Δu），需要先有实时滚动预测（#7）。

出处：w8/two_settlement_bess_external_model.pdf（双结算收益公式 + FERC/ISO 手册引用）、w10 第 5 节等价数学式 + 学术文献 [1] Alghumayjan 2024、[2] Krishnamurthy 2018。

### 5. 偏差罚金（可选）⭐ w10 新增

w10 第 5.1 节：偏差超 3% 容忍阈值按双倍 RT 价考核：
```
P_dev = Δt·Σ 2|pRT|·max(0, |Δu| − 0.03|uDA|)
R = R_base − P_dev
```
决策：正式版实现（先行版未做）。

### 6. Oracle ⭐ w10 新增

w10 第 5.2 节：LP Oracle R* = max R(u; p)，约束含功率/SOC/效率/循环。
先行版已实现（scipy.linprog）。
指标 PCR = ΣR_i / ΣR*_i（先行版用"占 LP %"替代，等价）。

---

## 二、预测阶段 ⭐ w10 新增

### 7. 日前联合预测 + 实时滚动预测

w10 第 2 节：
- **日前联合预测**：预测交易日及后一日共 48h 的 pDA + pRT 两条曲线
- **实时滚动预测**：每个执行小时预测未来 H=4h 的 pRT，只执行第一个动作，下一小时更新

先行版：只做 24h 单曲线日前预测。**正式版需补 48h + 滚动。**

### 8. 各 Encoder 的基础结构

决策：混合方案 C（Price/Load Transformer, Weather/System GRU, Econ MLP, Calendar Embedding）。
先行版已实现。w10 第 3 节：TSFM 编码历史价格，其余模态由适配层编码，融合模块形成共享表示。

### 9. Attention / 位置编码

MHA + RoPE（跟 TimesFM 对齐）。先行版已实现。

---

## 三、Decoder 层面

### 10. Learnable Queries 数量

- 小时级 → 24 queries（先行版）
- w10：48h 联合预测 → 需要 48 queries（正式版）

### 11. Decoder 结构

DETR-like（Learnable Queries + Cross Attention + FFN）。先行版已实现。

---

## 四、输出头层面

### 12. Head_DA / Head_RT 共享 Decoder

决策：共享 Decoder + 分叉 Head（A）。
w10 第 3 节：日前模块输出两条曲线（pDA + pRT|DA），实时模块输出一条（pRT）。

### 13. 额外输出头

第一轮只做电价预测（Head_DA + Head_RT）。跑通后加 Load Head。
先行版已实现 DA/RT 双头。**正式版需补**：日前头输出 48h pDA + 48h pRT 两条曲线。

---

## 五、BESS 参数 ⭐ w10 新增

### 14. 储能物理参数（w10 第 7 节）

| 参数 | 先行版 v1/v2 | 先行版 v3 | w10 规范 |
|---|---|---|---|
| P_max | 1 MW | 1 MW | 1 MW |
| E_max | 4 MWh | 4 MWh | 4 MWh |
| η | 0.9 | 0.95 | 0.95 |
| SOC min/max | [0, 4] | [0.4, 3.6] | [0.4, 3.6] MWh |
| κ（退化成本）| 0 | 27 USD/MWh | 27 USD/MWh |
| E_cyc（日循环上限）| 无 | 无 | 4 MWh/天 |
| 偏差罚金 | 无 | 无 | 3%, 双倍 RT 价 |

先行版 v3 已对齐 η/SOC/κ；**正式版需补** E_cyc + 偏差罚金。

### 15. 上下文窗口

7 天历史（168h）。先行版已实现。

### 16. 市场与数据

ERCOT LZ_LCRA（先行版）。ERCOT 统一小时表 2020-2026（6.5 年 DA+RT）已接入。
NYISO 也有 DA+RT 6.5 年数据（618K 行），可做泛化性验证。

### 17. 参数配置

| 版本 | d_model | n_layers | 参数量 | 训练样本 | 训练时长 |
|------|---------|----------|--------|----------|----------|
| 先行版 v1/v2 | 128 | 1 | 1.13M | 3,259 | 17 min |
| 先行版 v3 | 256 | 1 | 4.36M | 43,640 | 3 h |
| **正式版** | **512** | **4** | **~100M** | **~650,000（15 节点）** | **云 GPU** |

---

## 六、待确认 / 正式版待补清单

### 先行版已完成的
1. ✅ 混合 Encoder（Price/Load Transformer, Weather/System GRU, Econ MLP, Calendar Emb）
2. ✅ Cross-Attn 融合 + Query Decoder（DETR-like）
3. ✅ STE 可微 π + α-β 退火
4. ✅ LP Oracle（scipy.linprog）
5. ✅ 6.5 年 ERCOT DA+RT 双价数据
6. ✅ 真双结算收益公式（简化版 uDA=uRT）
7. ✅ w10 BESS 参数（η=0.95, κ=27, SOC 0.4-3.6）
8. ✅ 真节假日
9. ✅ TopK 策略（soft 近似，验证了需换零阶梯度）
10. ✅ B 对比（STE 模型 vs TimesFM/Chronos/Toto 零样本）

### 正式版待补的
1. **零阶双点梯度**（w10 第 6 节，替代 STE/soft TopK）— 需云 GPU
2. **48h 联合预测**（pDA + pRT 两曲线）— 架构改动
3. **实时滚动预测**（H=4）— 架构改动
4. **DA/RT 分离决策**（先报 uDA，实时纠偏 Δu）— 依赖 #2/#3
5. **偏差罚金**（3% 双倍 RT 价）— 一行公式
6. **日循环上限 E_cyc** — TopK K=4 已隐式约束，影响小
7. **多节点联合训练**（15 节点 650K 样本）— 支撑 100M 参数
8. **模型放大**（d=512, 4 层, ~100M）— 依赖 #7
9. **XGBoost 并列路径**（w10 第 6.3 节）
10. **PCR 指标**（表述形式，等价于"占 LP %"）
