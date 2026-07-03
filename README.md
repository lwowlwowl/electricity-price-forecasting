# 电价预测研究：消融驱动的时序基础模型融合

基于 ERCOT 实时电价，通过**三阶段递进研究**构建面向尖峰检测的专用融合模型：

```
v1.0 参数消融  →  v2.0 结构消融  →  ElecFM 融合模型（消融驱动架构设计）
  （11模型对比）     （68次实验）       （v1→V6，共9个版本）
```

研究问题：时序基础模型在电价预测中哪些组件真正有效？消融结论能否驱动更好的融合架构？

---

## 核心成果

| 指标 | TimesFM 零样本 | **ElecFM v5a** | **ElecFM V6** |
|------|--------------|--------------|--------------|
| W1 SMAPE | 27.67 | **27.55** ✅ | 28.90（3节点口径）|
| W1 Spike-F1 | 0.329 | **0.4159**（+26%）| **0.4439**（+35%）|
| 可训练参数 | — | 334K | 664K |
| 架构来源 | — | 逐层消融→spike head 分叉点 | + Chronos skip_variate 消融 |

---

## 项目结构

```
school/
├── configs/
│   ├── parameter_ablation/        # v1.0 参数消融配置（5维度×3窗口）
│   ├── structural_ablation/       # v2.0 结构消融配置（36+32次实验）
│   └── fusion/                    # ElecFM 训练配置
│       ├── electfm_ercot_full_v5.yaml          # v5a：纯 spike head，15节点
│       ├── electfm_ercot_full_v5b.yaml         # v5b：720h context
│       ├── electfm_ercot_full_v6.yaml          # V6：CrossNodeAttention
│       ├── electfm_ercot_full_v6_allgroups.yaml # V6 15节点全分组
│       ├── electfm_ercot_full_v7.yaml          # V7：SwiGLU adapter
│       └── ...（其他变体）
│
├── src/
│   ├── data_processing/
│   │   ├── loader.py              # 从 raw 长表按需切片
│   │   └── build_nodes_config.py # 节点分组（volatility/spikes/stable）
│   ├── models/                   # v1.0 参数消融用到的模型框架
│   │   ├── base.py               # Forecaster 抽象基类
│   │   ├── forecasters.py        # 7 个统计/树模型基线
│   │   ├── foundation.py         # 4 个基础模型的子进程适配器
│   │   └── workers/              # 各基础模型 worker
│   ├── parameter_ablation/       # v1.0 实验执行器
│   ├── structural_ablation/      # v2.0 消融模块（14 种手术操作）
│   └── fusion_model/             # ElecFM 融合模型
│       ├── model.py              # ElecFM + CrossNodeAttention + SwiGLUAdapter
│       ├── train.py              # 两阶段训练（spike_head_only / cross_node_only）
│       ├── dataset.py            # 滑窗数据集 + 3节点同步数据集
│       ├── evaluate.py           # 三窗口回测 + τ* 搜索
│       ├── run_fusion.py         # 一键入口（Step1验证 → 训练 → 评估）
│       └── loss.py               # Pinball + BCE Spike 联合损失
│
├── docs/
│   ├── fusion/                   # ElecFM 当前活跃文档
│   │   ├── README.md             # 文档导航入口
│   │   ├── design.md             # 架构设计 + 全版本实验汇总
│   │   ├── experiments.md        # 详细实验记录（按时间线）
│   │   └── implementation.md    # 代码实现指南
│   ├── archive/elecfm/           # 历史文档归档
│   ├── 参数消融汇报材料.md        # v1.0 关键发现与结论
│   ├── 参数消融实验结果与问答.md  # v1.0 完整分析
│   ├── 结构消融汇报材料.md        # v2.0 + ElecFM 完整研究记录
│   └── 结构消融实验结果与问答.md  # v2.0 完整分析
│
├── scripts/                      # 辅助脚本
│   ├── run_lora_quick_test.sh    # LoRA 快速测试
│   └── run_fusion_sweep.sh       # 参数扫描
├── run_overnight.sh              # 多实验串行脚本
├── CHANGELOG.md                  # 版本变更日志
└── external/                     # 四个基础模型（各含独立 .venv）
    ├── timesfm/                  # TimesFM-2.5（Google）
    ├── chronos-forecasting/      # Chronos-2（Amazon）
    └── toto/                     # Toto-1.0 & Toto-2.0（Datadog）
```

---

## 第一阶段：v1.0 参数消融（已完成）

**目的**：确定时序基础模型在电价预测中的最优输入配置，回答"给模型什么"的问题。

**实验规模**：11 个模型（7 基线 + 4 基础模型）× 5 消融维度 × 3 测试窗口 = 15 组，60+ 次回测

| 消融维度 | 扫描范围 | 关键结论 |
|---------|---------|---------|
| A 协变量 | 无 → 负荷 → +温度 → +风光 | Chronos 加全协变量 Spike-F1 从 0.317→0.417 |
| B 上下文长度 | 168h / 336h / 720h | TimesFM 对长度不敏感；720h 在 W1 SMAPE=26.84（最优）|
| C 单/多变量 | 单变量 / 多变量 | 跨节点增益有限（≤3%），不引入 |
| D 预测步长 | 24h / 48h / 168h | 所有模型随步长恶化，Toto2 步长鲁棒性最好 |
| F 数据频率 | 1h / 15min | 15min 对 Spike-F1 有微弱增益，对 MAE 无帮助 |

**测试窗口**：

| 窗口 | 区间 | 市场特征 |
|------|------|---------|
| W1 | 2025-08-01 ~ 08-31 | 夏季稳定期（基准）|
| W2 | 2025-03-01 ~ 03-31 | 春季负电价 |
| W3 | 2026-01-01 ~ 01-31 | 冬季极端尖峰 |

```bash
# 运行单个基准实验
python src/parameter_ablation/run_experiment.py configs/parameter_ablation/baseline.yaml

# 一键运行全部消融
bash run_all_ablations.sh
```

---

## 第二阶段：v2.0 结构消融（已完成）

**目的**：打开模型内部，定位每个基础模型的关键组件，回答"模型哪里在干活"的问题。

**实验规模**：
- 组件级消融：**36 次**（3 模型 × 12 类操作，含 FFN / 注意力 / 位置编码 / 输出头）
- 逐层消融：**32 次**（Toto2 6层 + Chronos2 6层 + TimesFM 20层）
- 统计检验：Wilcoxon signed-rank + Bonferroni 校正

**核心发现**：

| 发现 | 数据依据 |
|------|---------|
| **FFN 是所有模型最关键组件** | 移除后 TimesFM +486%、Toto2 +419%、Chronos2 +33% |
| **精度通路与尖峰通路功能分离** | 部分层专精度、部分层专尖峰，甚至互相抑制 |
| **TimesFM L7 是纯尖峰检测层** | ΔSpike-F1 = −6.7%，ΔSMAPE = +1.6%（精度几乎不变）|
| **Chronos skip_variate 退化 +6.7%** | 跨变量注意力贡献显著 |
| **TimesFM 40% 层可安全移除** | 逐层消融，双指标严格标准 |

```bash
# 运行结构消融
python src/structural_ablation/run_structural_ablation.py configs/structural_ablation/full_timesfm.yaml
```

图表见 `data/results/structural_ablation/report/`。

---

## 第三阶段：ElecFM 融合模型（已完成）

**核心思路**：把消融结论转化为架构决策，构建同时具备零样本精度和显式尖峰检测能力的专用模型。

### 架构

```
原始电价序列 [B, 168h]
  → Tokenizer → 15层 Transformer（TimesFM，剪枝5层，冻结）
                          ↓ 新L6=原L7（"尖峰检测层"，消融发现）
              ┌───────────────────────────┐
              │  CrossNodeAttention（V6）  │  ← Chronos skip_variate 消融动机
              │  [B, 3, 1280] → [B, 3, 1280]│
              └───────────────────────────┘
                          ↓
              Spike Head（自研，334K）→ spike_logits [B, 24]
              Quantile Head（TimesFM原版，冻结）→ q[0.1..0.9] × 24步
```

### 实验历程

| 版本 | 核心改动 | W1 SMAPE | W1 Spike-F1 | 关键教训 |
|------|---------|---------|------------|---------|
| v1 | quant_head 可训 | 31.95 | 0.370 | quant_head 必须冻结 |
| v2 | quant_head 冻结 | 29.19 | 0.335 | 数据太少 |
| v3 | 全 ERCOT 非 LoRA | 未收敛 | — | 79M 参数 vs 125K 样本 |
| v4 | LoRA | 29.39 | 0.351 | 骨干修改仍损害 SMAPE |
| **v5a** | **纯 spike head，15节点** | **27.55** ✅ | **0.4159** ✅ | **冻结骨干是正确路线** |
| v5b | 纯 spike head，720h | 27.52 | 0.3779 | 720h 稀释尖峰信号 |
| **V6** | **+ CrossNodeAttention，3节点** | **28.90**\* | **0.4439** ✅ | **消融预测闭环验证** |
| V7 | + SwiGLU adapter | 27.55 | 0.3980 | 冻结骨干表征已足够，SwiGLU 冗余 |

\* V6 评估口径为 LZ_LCRA/LZ_WEST/LZ_RAYBN 三个高波动节点，零样本基准即 28.90。

### 最终最优模型

**日常精度（广泛节点）**：v5a（15节点，W1 SMAPE=27.55）

**尖峰检测**：V6（3节点，W1 Spike-F1=0.4439）

### 快速运行

```bash
# Step 1：零样本验证（确认剪枝基准）
external/timesfm/.venv/bin/python src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_v5.yaml --step1-only

# v5a 完整训练（约 30 分钟）
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_v5.yaml \
    2>&1 | tee run_v5a.log

# V6 完整训练（约 30 分钟）
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_v6.yaml \
    2>&1 | tee run_v6.log
```

---

## 可用模型（v1.0 回测框架）

| 模型 | 类型 | 运行位置 |
|------|------|---------|
| Naive / SeasonalNaive / ETS / Theta | 统计基线 | 进程内 |
| RandomForest / LightGBM / XGBoost | 树模型 | 进程内（每起报点重训）|
| TimesFM-2.5（Google）| 时序基础模型 | 独立 venv 子进程 |
| Chronos-2（Amazon）| 时序基础模型 | 独立 venv 子进程 |
| Toto-1.0 / Toto-2.0（Datadog）| 时序基础模型 | 独立 venv 子进程（共用 venv）|

---

## 方法论贡献

### 三步递进关系

```
参数消融（v1.0）
  → 确认精度瓶颈在模型内部，不在输入配置
  
结构消融（v2.0）
  → 发现精度通路/尖峰通路功能分离
  → 定位 TimesFM L7 为纯尖峰检测层
  → 量化 Chronos 跨变量注意力贡献 (+6.7%)

ElecFM 融合模型（v3.0）
  → 消融结论→架构决策：spike head 在 L7 分叉
  → 冻结骨干：SMAPE 保持零样本水平
  → V6 CrossNodeAttention：实测 +6.7%，与消融预测完全吻合
```

### 核心工程决策

- **quant_head 永久冻结**：小样本无法维持预训练分位数校准（Coverage 从 0.775 降至 0.582）
- **骨干完全冻结**：任何骨干权重修改都导致 SMAPE 退化（4 轮实验一致）
- **spike head 接入 L7**：消融数据直接给出，非拍脑袋
- **CrossNodeAttention 使用同质节点**：跨价格区间节点混合导致数值不稳定（NaN 实验证实）

---

## 文档导航

| 文档 | 内容 |
|------|------|
| `docs/fusion/README.md` | ElecFM 入口 |
| `docs/fusion/design.md` | 架构设计 + 全版本结果汇总表 |
| `docs/fusion/experiments.md` | 所有实验的详细记录 |
| `docs/fusion/implementation.md` | 代码结构与使用指南 |
| `docs/结构消融汇报材料.md` | v1.0+v2.0+ElecFM 完整研究记录 |
| `docs/参数消融汇报材料.md` | v1.0 结果与分析 |
| `CHANGELOG.md` | 版本变更日志 |

---

## 注意事项

- **基础模型 venv 独立**：TimesFM / Chronos / Toto 依赖冲突，各自使用 `external/<model>/.venv`，不要在主环境 import
- **数据范围**：ERCOT 实时电价 2025-01-01 ~ 2026-06-02，约 17 个月
- **测试隔离**：W1/W2/W3 测试窗口及前 168h buffer 已严格排除于训练集之外
- **运行目录**：所有脚本从项目根目录运行

---

## 参考

- TimesFM: https://github.com/google-research/timesfm
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Toto / Toto-2.0: https://github.com/DataDog/toto
- 详细方法论见 `docs/specs/experiment_manual_v2.md` 与 `docs/结构消融汇报材料.md`
