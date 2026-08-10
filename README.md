# 电价预测研究：Decision-aware 多模态 TSFM

> 当前方向：**Decision-aware 多模态时序基础模型（从零训练 + 业务损失闭环）**。
> 目标不是"预测得准"，而是"按预测调度储能（BESS）能多赚钱"——把预测损失和调度收益（零阶梯度代理）联合训练。
>
> 旧方向（冻结零样本 TimesFM + 消融驱动的 ElecFM 融合模型）已废弃并清理，历史结论仍见 `docs/archive/`（v1.0 参数消融、v2.0 结构消融的方法论与数据洞察，指导当前模型的输入设计）。

---

## 核心成果（v7，最新有完整结果的正式版本）

在 ERCOT LZ_LCRA 节点、单节点小模型（4.36M 参数）上，v7 是第一个 **w10 方案口径完全一致**的版本（价差决策、双结算 LP Oracle、零阶梯度全部对齐同一套结算公式）：

| 指标 | 含义 | v4 | v6 | **v7** |
|------|------|----|----|--------|
| MAE | 预测误差（美元）| 49.5 | 20.2 | **21.1** |
| RMSE | 均方根误差（美元）| — | — | 53.2 |
| R_model | 模型实际调度收益 | -81.9 | -149.0 | **-19.0**（少亏 77%）|
| R*_LP | 双结算 LP Oracle 上界 | — | — | 276.0 |
| PCR | 收益占 Oracle 比例 | -54% | -99% | **-6.9%**（接近盈亏平衡）|

> regret 指标口径在 v4/v6/v7 间不一致（v7 是双结算 Oracle），不可直接横向比较，仅 MAE/R_model/PCR 可比。详细训练动态、验证标准、MSE 数值爆炸的教训见 `docs/正式版实验结果.md`。

**下一步 v8**（`configs/decision_aware/formal_ercot_v8.yaml`，配置已就绪，尚未训练出结果）：在 v7 基础上修复了两轮交叉代码审查发现的架构缺陷（Pre-LN 缺 final LayerNorm、融合层误开 RoPE、多流缺 modality embedding 等）和正确性 bug（L_proxy 梯度被 detach 截断、LP Oracle 缺 κ/E_cyc 等），参数量增至 9.92M，详见 `docs/todo3_2.md`。

---

## 项目结构

```
school/
├── configs/
│   ├── nodes.yaml                    # 节点分组配置（volatility/spikes/stable）
│   ├── decision_aware/               # 当前方向：Decision-aware 训练配置
│   │   ├── pilot_ercot.yaml / pilot_ercot_v3.yaml   # 先行版（小规模验证）
│   │   └── formal_ercot.yaml ~ v8.yaml              # 正式版迭代（v7 最新有结果，v8 待跑）
│   └── archive/                      # 旧范式配置（v1.0 参数消融、v2.0 结构消融）
│
├── src/
│   ├── data_processing/              # 数据加载与节点分组
│   │   ├── loader.py                 # 从 raw 长表按需切片
│   │   └── build_nodes_config.py
│   ├── decision_aware/                # 当前方向核心代码
│   │   ├── model.py                   # 多流 Encoder + CrossModalFusion + DA/RT 双 Decoder
│   │   ├── policy.py                  # BESS 模拟器 + HardTopK 策略 + LP Oracle（单/双结算）
│   │   ├── loss.py                    # 预测损失（Huber/MSE）+ L_proxy 决策感知损失
│   │   ├── zero_order.py              # 零阶梯度估计（双点高斯扰动）
│   │   ├── dataset.py / dataset_v3.py # 滑窗数据集
│   │   └── train.py                   # 先行版训练循环
│   ├── evaluation/                    # 指标 / 统计检验 / 回测
│   ├── models/                        # 预测器层：统计/树基线 + 外部 TSFM 适配器
│   │   └── workers/                   # TimesFM / Chronos-2 / Toto 子进程 worker
│   └── archive/                       # 旧范式代码（v1.0/v2.0 消融），已废弃不维护
│
├── scripts/
│   ├── decision_aware/
│   │   ├── train_formal.py            # 正式版训练入口（v3+，支持 dual_split）
│   │   ├── train_pilot.py / train_pilot_v3.py  # 先行版训练入口
│   │   ├── compare_baselines_formal.py / compare_baselines.py  # 基线对比
│   │   ├── eval_v3_da_oracle.py       # Oracle 评估
│   │   └── covariate_screen_xgb.py    # XGBoost 协变量筛选
│   └── covariates/                    # 协变量数据管线（下载/清洗/合并/特征构建，15 个步骤脚本）
│
├── docs/
│   ├── todo3.md / todo3_2.md          # 当前决策清单（模型搭建 + v7 改动清单 + 架构/bug 审查记录）
│   ├── 正式版实验结果.md               # 正式版 v4~v7 完整实验记录（本 README 摘要来源）
│   ├── 先行版实验结果.md / 先行版汇报材料.md
│   ├── model_architecture_v2/v3.drawio  # 架构图
│   ├── covariate_conclusions.md       # 协变量结论（多模态输入设计依据）
│   ├── specs/                         # 实验方法论手册
│   ├── concepts/                      # 概念说明（起报点与滚动回测等）
│   ├── reference/                     # 论文 PDF（TimesFM/Chronos/Toto 等）
│   ├── team_notes/                    # 团队周会纪要与研究方案（w1, w7~w11）
│   └── archive/                       # 旧范式文档归档（v1.0/v2.0 消融结论，仍有复用价值）
│
├── data/
│   ├── raw/                           # 原始数据源（EIA/ERCOT/NYISO/PJM/CAISO/weather，不入库）
│   ├── covariates/                    # 协变量派生数据（gas/oil/steel/storm/news/generation_mix）
│   ├── results/                       # 实验输出（parameter_ablation/structural_ablation 为旧范式产物）
│   ├── checkpoints/                   # 训练 checkpoint（不入库，本地产物）
│   └── market_hourly.parquet          # 主数据表
│
└── external/                          # 三个基础模型，各含独立 .venv
    ├── timesfm/                        # TimesFM-2.5（Google）
    ├── chronos-forecasting/            # Chronos-2（Amazon）
    └── toto/                           # Toto-1.0 & Toto-2.0（Datadog）
```

---

## Decision-aware 训练框架

**核心思路**：不满足于预测准确，而是让模型直接对储能调度收益负责。用一个共享 Encoder-Decoder 同时预测日前价格 p̂DA 和日前口径下的实时价格 p̂RT|DA，再用价差 d̂ = p̂DA − p̂RT|DA 驱动 HardTopK 调度策略，通过零阶梯度（双点高斯扰动，模拟器不可导）把调度收益的代理损失 L_proxy 传回预测头，和预测损失 L_pred 加权（α/β 退火）联合训练。

```
历史价格/日历/系统特征等多流输入
  → StreamEncoder（每流独立编码）
  → CrossModalFusion（跨流注意力）
  → QueryDecoder（DA 48h 双曲线 + RT 24×4h 滚动窗口）
  → p̂DA, p̂RT|DA, p̂RT
        ↓                          ↓
   L_pred（Huber）          价差 d̂ → HardTopK 策略 → BESS 模拟器
        ↓                          ↓
        └──────→ α·L_pred + β·L_proxy ←──────┘
              （零阶梯度估计 L_proxy 对预测头的梯度）
```

### 快速运行

```bash
# v7 训练（~8 小时，M3 Pro MPS）
external/chronos-forecasting/.venv/bin/python \
    scripts/decision_aware/train_formal.py \
    --config configs/decision_aware/formal_ercot_v7.yaml --no-early-stop --no-oracle-train \
    2>&1 | tee /tmp/v7_train.log

# 基线对比
external/chronos-forecasting/.venv/bin/python \
    scripts/decision_aware/compare_baselines_formal.py \
    --config configs/decision_aware/formal_ercot_v7.yaml
```

---

## 旧范式：三阶段消融研究（已归档，结论仍有效）

在切换到 Decision-aware 方向之前，项目经历了"参数消融 → 结构消融 → ElecFM 融合模型"三阶段递进研究，用于回答"时序基础模型在电价预测中哪些组件真正有效"。ElecFM 融合模型本身（冻结零样本骨干 + 挂尖峰检测头）与当前"从零训练 + 业务闭环"范式架构不可调和，代码与实验产物已删除；消融阶段的**结论**被沉淀为文档，仍用于指导当前模型设计。

| 阶段 | 规模 | 核心结论 | 详情文档 |
|------|------|---------|---------|
| v1.0 参数消融 | 11 模型 × 5 维度 × 3 窗口 | 协变量加全提升 Spike-F1；720h context 在稳定期最优；单变量已够，跨节点增益有限 | `docs/archive/参数消融汇报材料.md` |
| v2.0 结构消融 | 36 次组件消融 + 32 次逐层消融 | FFN 是所有模型最关键组件；精度通路与尖峰通路功能分离；TimesFM 40% 层可安全移除 | `docs/archive/结构消融汇报材料.md` |
| ElecFM 融合模型 | v1→V6 共 9 个版本 | 冻结骨干 + spike head 在消融定位的分叉层接入，可用极少参数（334K）提升尖峰检测；已废弃 | `docs/archive/README.md`（索引） |

```bash
# 旧范式脚本仍保留，仅供查阅（run_experiment.py 依赖已删除的部分配置，不保证可直接运行）
external/timesfm/.venv/bin/python src/archive/parameter_ablation/run_ablation.py configs/archive/parameter_ablation/baseline.yaml
```

---

## 可用模型（预测器层，跨阶段复用）

| 模型 | 类型 | 运行位置 |
|------|------|----------|
| Naive / SeasonalNaive / ETS / Theta | 统计基线 | 进程内 |
| RandomForest / LightGBM / XGBoost | 树模型 | 进程内（每起报点重训）|
| TimesFM-2.5（Google）| 时序基础模型 | 独立 venv 子进程 |
| Chronos-2（Amazon）| 时序基础模型 | 独立 venv 子进程 |
| Toto-1.0 / Toto-2.0（Datadog）| 时序基础模型 | 独立 venv 子进程（共用 venv）|

---

## 文档导航

| 文档 | 内容 |
|------|------|
| `docs/todo3.md` | 当前决策清单（Decision-aware 模型搭建）|
| `docs/todo3_2.md` | v7 改动清单 + 两轮架构缺陷/代码 bug 交叉审查记录 |
| `docs/正式版实验结果.md` | 正式版 v4~v7 完整实验记录（训练动态、口径说明、下一步计划）|
| `docs/先行版实验结果.md` / `docs/先行版汇报材料.md` | 先行版验证结果 |
| `docs/covariate_conclusions.md` | 协变量结论（多模态输入设计依据）|
| `docs/新闻特征初步汇报材料.md` | LLM 新闻特征提取（Event Encoder 可选输入）|
| `docs/team_notes/w8/研究方案_Decision-aware_多模态TSFM.md`、`w9/模型相关.md` | 方向切换的研究方案原始记录 |
| `docs/archive/README.md` | 旧范式文档归档索引 |

---

## 注意事项

- **基础模型 venv 独立**：TimesFM / Chronos / Toto 依赖冲突，各自使用 `external/<model>/.venv`，不要在主环境 import
- **数据范围**：ERCOT 实时电价 2025-01-01 ~ 2026-06-02，约 17 个月
- **测试隔离**：W1（稳定期）/W2（负电价）/W3（极端尖峰）测试窗口及前 168h buffer 已严格排除于训练集之外
- **运行目录**：所有脚本从项目根目录运行
- **checkpoint 与日志不入库**：`data/checkpoints/`、`logs/` 均已 gitignore，为本地训练产物，清理前如需保留请自行备份

---

## 参考

- TimesFM: https://github.com/google-research/timesfm
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Toto / Toto-2.0: https://github.com/DataDog/toto
- 详细方法论见 `docs/specs/experiment_manual_v2.md` 与 `docs/archive/结构消融汇报材料.md`
