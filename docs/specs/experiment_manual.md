# 电价预测多模型消融实验手册

> 版本：v1.0 ｜ 适用项目：`school/`（电价预测）
> 读者：负责模型方向的科研同学
> 目标：建立一套**可控变量、可复现、可统计**的实验体系，对 TimesFM / Chronos-2 / Toto 等时序基础模型做消融实验，识别各自的强项板块，最终设计一个融合模型。

---

## 0. 这份手册要解决什么

在动手之前，先明确一个核心认知：**当前卡点不是"不会用模型"，而是"实验还没被定义成一个可控变量的标准单元"。** 你提的三个困惑——数据怎么取、训练类与零样本类怎么统一、单变量与多变量怎么处理——本质上是同一个问题：实验的"旋钮"还没有被抽象出来。

一旦把实验定义清楚，这三个问题会同时消失。本手册就是把这套定义、流程、指标、实验矩阵一次性写清楚，让任何人（包括三个月后的你自己）都能照着推进。

### 核心抽象：一次实验由若干"旋钮"完全确定

```
一次实验 = f(市场, 节点, 频率, 上下文长度, 预测步长, 起报点集合, 协变量集合, 模型, 是否微调)
```

**消融实验 = 固定其它所有旋钮，只转动其中一个，观察指标如何变化。**

例如：固定 `ERCOT + 节点A + 1h + 上下文168 + 步长24 + Toto + 零样本`，只改协变量集合 `{无} → {负荷} → {负荷+天气}`，看 MAE 与 Spike-F1 是否提升——这就回答了"协变量对 Toto 有没有用"。

---

## 1. 实验设计哲学（务必先读懂）

### 1.1 三个公平性原则

整个体系建立在三条不可妥协的原则上，违反任何一条，结论都不成立：

1. **同起跑线原则**：在任意一个起报点，所有模型（无论零样本还是需训练）都只能看到该起报点**之前**的数据。需训练模型的 `fit` 也只能用历史数据，绝不允许"偷看"未来。
2. **多点统计原则**：任何结论都不能基于单个起报时间。必须在测试期内设置多个起报点，对误差取平均并报告标准差。单点结论没有统计意义。
3. **能力诚实原则**：模型支持什么能力（协变量、多变量）就用什么；不支持的自动降级为单变量，并在结果中**显式标记**，让"因能力差异带来的优势"可被看见，而非被掩盖。

### 1.2 为什么要统一框架

当前三个 `forecast_*.py` 各写各的、参数硬编码、节点选择逻辑略有差异、且都被人为压成"单变量零样本"在比。这种对比无法体现模型各自的强项板块，也无法做受控消融。

统一框架的目标：让"训练/零样本""单/多变量""有/无协变量""不同上下文长度"都成为**配置项**，而非分散在不同脚本里的硬编码。所有模型走同一条评估路径，唯一的差异来自旋钮本身。

---

## 2. 数据：从"取全部"到"按需切片"

### 2.1 数据现状盘点

项目内有两套数据，目前是脱节的，需要厘清：

| 数据集 | 位置 | 时间范围 | 频率 | 协变量 | 用途定位 |
|---|---|---|---|---|---|
| ENLITEN | `data/processed/lmp_processed.csv` | 2020 单年 | 1h | 无 | 仅供框架跑通验证，**做不了协变量实验** |
| 市场数据 | `data/raw/{CAISO,ERCOT,NYISO,PJM}/processed/` | 2025–2026 | 5min/15min/1h | 负荷、风电、光伏 | **主力数据**，支持协变量消融 |
| 天气 | `data/raw/weather/processed/by_ba/*_weather_hourly.csv` | 2020–2026 | 1h | 温度等 | 协变量来源 |
| EIA | `data/raw/EIA/` | 2020–2026 | 1h | 系统运行数据 | 备用协变量 |

命名高度规整：`{市场}/processed/{变量}_{频率}.csv`，例如 `ERCOT/processed/actual_price_hourly.csv`、`actual_load_15min.csv`、`wind_actual_hourly.csv`。

### 2.2 市场选择策略（先深耕一个，再泛化）

| 顺序 | 市场 | 选择理由 | 对应实验 |
|---|---|---|---|
| 第一 | **ERCOT** | 价格波动极大、含负电价与尖峰，最能拉开模型差距；已有价格画像分析 | 主消融战场（协变量、上下文、单/多变量） |
| 第二 | **CAISO** | 风光数据最全 | 验证"协变量提升"能否**跨市场泛化** |
| 第三 | **NYISO** | 有 2020–2026 长历史 | 支撑"零样本 vs 微调"（微调需长训练数据） |

不必一次全做，先把 ERCOT 跑透，结论成型后再扩展。

### 2.3 统一数据加载器 `load_slice()`

核心：**永远只取一次实验需要的那一小片对齐数据**，而不是把大文件全部读进内存。

```python
load_slice(
    market="ERCOT",
    nodes=["<节点A>", "<节点B>"],   # 单个或多个
    freq="1h",                       # 1h / 15min / 5min
    target="price",                  # 预测目标 = 电价
    covariates=["load", "wind", "solar", "temperature"],  # 可为空
    start="2025-06-01", end="2025-09-01",
) -> DataFrame[timestamp, price, load, wind, solar, temperature, ...]
```

实现要点：

- **时间对齐**：以电价的 `timestamp` 为主轴，把负荷、风光、天气按时间戳 `join`。频率不一致时（如天气只有 1h，电价有 15min），低频协变量做前向填充或重采样到目标频率。
- **缺失处理**：沿用数据交接说明里的标记列（`value_was_imputed_by_cleaning` 等），实验时可选择是否剔除被插补的行，并在结果中记录。
- **协变量分两类**：
  - *历史协变量*：起报点之前已知（如历史负荷、历史温度实测）。
  - *未来协变量*：预测窗口内可获取（如负荷预测、天气预报）。电价预测中负荷/天气**预报**通常是可得的，这对 Chronos-2、TimesFM 的协变量接口至关重要。手册中默认用"未来真值"做上界分析，并标注这是乐观估计。

### 2.4 节点选择：固化成可复现清单

当前 `nlargest(std)` 的思路正确，但需固化。先对每个市场跑一次统计，把代表性节点写进 `configs/nodes.yaml`，之后所有实验从固定清单选，保证可比性：

```yaml
ERCOT:
  volatility:  [节点1, 节点2, 节点3]   # 标准差最大
  spikes:      [节点4, 节点5, 节点6]   # 尖峰次数最多
  stable:      [节点7, 节点8, 节点9]   # 最平稳（对照组）
```

---

## 3. 统一实验框架

### 3.1 统一的 Forecaster 接口

用统一协议把"训练/零样本""能力差异"全部藏到接口之后：

```python
class Forecaster:
    name: str
    needs_training: bool          # True=需fit, False=零样本
    supports_covariates: bool     # 是否支持协变量
    supports_multivariate: bool   # 是否原生支持多变量

    def predict(self, context_df, future_covariates=None, horizon=24):
        """
        context_df: 起报点之前的历史（含 target 与历史协变量）
        future_covariates: 预测窗口内的未来协变量（若模型/实验不支持则为 None）
        返回: Forecast(mean, q10, q50, q90)  形状 (horizon,)
        """
        ...
```

- **零样本类**：`needs_training=False`，`predict` 内不依赖额外训练即可出预测。又分两小类：
  - *朴素/统计基线*（Naive、Seasonal-Naive、ETS、Theta）：纯统计方法，无需预训练权重。ETS（指数平滑 / Holt-Winters）与 Theta 是 M3/M4 竞赛公认的强统计基线，逐节点在 `context_df` 上即时拟合并外推，是任何高级模型都该打败的下界。
  - *基础模型*（TimesFM、Chronos-2、Toto）：加载预训练权重直接推理。
- **需训练类**（RandomForest、XGBoost、LightGBM、MLP、LSTM、GRU）：`needs_training=True`，`predict` 内先用 `context_df` 之前的历史 `fit` 再 forecast（**每个起报点都重新训练，从结构上杜绝泄露**），对外表现与零样本一致。当前已接入树模型三件套（RandomForest / LightGBM / XGBoost），复用统一的电价特征工程（多阶 lag、滚动均值方差、差分、小时-峰谷），递归多步预测。

### 3.2 三个基础模型的能力与正确用法

> 当前代码把三个模型都压成了"单变量零样本"，浪费了它们的原生能力。下表是正确用法。

| 模型 | 单变量 | 多变量 | 协变量 | 当前代码问题 | 正确用法 |
|---|---|---|---|---|---|
| TimesFM-2.5 | ✓ | 逐序列 | ✓（covariate 接口） | 仅单变量 | 单变量为主，协变量作为消融项 |
| Chronos-2 | ✓ | **✓ 原生** | **✓ 原生** | 被逐节点循环当单变量用 | 启用 `(n_series, n_variates, history)`，喂协变量 |
| Toto | ✓ | **✓ 原生** | 实验性 | 多节点 `id_mask` 全 0，未接协变量 | 多节点联合建模（空间相关性是其强项） |

### 3.3 自动降级机制

设置"本次实验用协变量"时，框架检查 `model.supports_covariates`：支持就喂，不支持则降级为单变量，并在结果行写入 `covariates_used=False`。这样对比表中能直接看出"哪个模型因能用协变量而获得优势"——这本身就是一条重要消融结论。

---

## 4. 评估协议：多起报点滚动回测

### 4.1 为什么必须多起报点

当前所有脚本只用 `iloc[-PRED_LEN:]` 测最后一天，这是最大的方法论缺陷：单点测试换个日期结论就变，没有统计意义。必须改为**滚动回测（walk-forward / rolling-origin）**。

### 4.2 滚动回测流程

```
给定测试期 [test_start, test_end]：
  起报点 t = test_start, test_start + STRIDE, test_start + 2*STRIDE, ...
  对每个起报点 t：
    context = 数据[t - CONTEXT_LEN : t]        # 仅历史
    actual  = 数据[t : t + HORIZON]            # 真值（仅用于评估）
    对每个模型：
      pred = model.predict(context, future_cov, horizon=HORIZON)
      记录 (起报点 t, 模型, 节点, pred, actual)
  汇总所有起报点的误差 → 求均值与标准差
```

推荐默认参数（可在消融中改动）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| 测试期 | 至少覆盖 4–8 周 | 需横跨工作日/周末、不同负荷水平 |
| 起报步距 STRIDE | 24h | 每天设一个起报点（日前预测场景） |
| 起报点数量 | ≥ 30 个 | 保证统计显著；越多越稳 |
| CONTEXT_LEN | 168h（消融项） | 见 §6 |
| HORIZON | 24h（消融项） | 见 §6 |

> 提示：为了同时覆盖"普通时段"和"极端时段"，可在均匀起报点之外，额外采样若干"已知尖峰日/极端价日"作为压力测试子集，单独报告这部分的 Spike-F1。

### 4.3 多起报点带来的报告方式

每个 (模型, 配置) 不再是一个数字，而是一个**分布**：报告 `MAE 均值 ± 标准差`、`Spike-F1 均值 ± 标准差`。这让"A 比 B 好"变成可信的统计陈述，而非偶然。

---

## 5. 评估指标体系

分三类指标，缺一不可：点误差、概率误差、**尖峰预警**。

### 5.1 点预测指标

| 指标 | 公式要点 | 解读 |
|---|---|---|
| MAE | 平均绝对误差 | 整体偏差，对所有时刻一视同仁 |
| RMSE | 均方根误差 | 放大大误差，对尖峰失误敏感 |
| MASE | MAE / 季节性Naive的MAE | 跨节点/跨市场可比（消除量纲） |
| sMAPE | 对称百分比误差 | 注意电价可能为负/接近零，需谨慎使用 |

### 5.2 概率预测指标

模型都输出分位数，应评估区间质量：

| 指标 | 说明 |
|---|---|
| Pinball Loss（分位数损失） | q10/q50/q90 的加权绝对误差，衡量分位数预测整体质量 |
| 覆盖率 Coverage | 真值落在 [q10, q90] 的比例，理想接近 80% |

### 5.3 ★ Spike-F1：尖峰预警能力（本项目重点指标）

电价预测中，**预测准尖峰**往往比平均误差更有业务价值。把它定义成一个二分类问题。

#### 定义

**第一步：确定尖峰阈值（历史 95 分位）**

```
threshold = 该节点在【训练期/历史窗口】电价的 95 分位数 (P95)
```

关键约束（避免数据泄露）：

- 阈值**必须只用起报点之前的历史数据**计算，绝不能用包含测试期或未来的全量数据算 P95。
- 推荐两种口径，二选一并在报告中说明：
  - **全局固定阈值**：用测试期开始之前的全部历史算一个固定 P95（简单、稳定，推荐起步用）。
  - **滚动阈值**：每个起报点用其之前的滚动历史窗口算 P95（更严谨，贴近真实部署）。
- 阈值**按节点分别计算**（不同节点价格量级差异大）。

**第二步：把"连续电价"转成"是否尖峰"的0/1标签**

```
实际是否尖峰  y_true[t] = 1 if actual[t] >= threshold else 0
预测是否尖峰  y_pred[t] = 1 if pred_mean[t] >= threshold else 0
```

> 可选增强：用预测分位数判断尖峰更合理——例如"若 q90[t] >= threshold 则预测为尖峰"，这能体现概率模型对尾部风险的预警能力。建议同时报告"用 mean 判定"和"用 q90 判定"两套 Spike-F1。

**第三步：在所有起报点 × 所有预测时刻上汇总混淆矩阵，计算 F1**

```
TP = 预测尖峰 且 实际尖峰
FP = 预测尖峰 但 实际不尖峰
FN = 预测不尖峰 但 实际尖峰
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
Spike-F1  = 2 * Precision * Recall / (Precision + Recall)
```

#### 报告要求

- 同时报告 **Precision、Recall、F1** 三者——只看 F1 会掩盖"模型是过度预警还是漏报"。电价场景通常更关心 Recall（少漏报尖峰）。
- 因为尖峰是稀有事件（约5%），F1 是合适的不平衡分类指标；不要用 Accuracy（会被"全预测不尖峰"刷高）。
- 按节点类型分组报告：在 `volatility/spikes` 节点上的 Spike-F1 才是真正考验，`stable` 节点作对照。

#### Spike-F1 参考实现（伪代码）

```python
def compute_spike_f1(records_df, threshold_by_node, use_quantile="mean"):
    # records_df 列: node, pred_mean, pred_q90, actual
    tp = fp = fn = 0
    for node, g in records_df.groupby("node"):
        thr = threshold_by_node[node]          # 仅用历史算出的 P95
        y_true = (g["actual"] >= thr).astype(int)
        pred_signal = g["pred_q90"] if use_quantile == "q90" else g["pred_mean"]
        y_pred = (pred_signal >= thr).astype(int)
        tp += int(((y_pred == 1) & (y_true == 1)).sum())
        fp += int(((y_pred == 1) & (y_true == 0)).sum())
        fn += int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "spike_f1": f1,
            "tp": tp, "fp": fp, "fn": fn}
```

---

## 6. 消融实验矩阵

每个消融"只动一个旋钮"。基准配置（Baseline Config）固定如下，所有消融都从它出发：

```
基准: ERCOT | volatility节点(3个) | 1h | 上下文168 | 步长24 | 零样本 | 无协变量 | ≥30起报点
```

| 编号 | 消融因素 | 变化范围 | 固定项 | 想回答的问题 |
|---|---|---|---|---|
| A | **协变量** | 无 → +负荷 → +负荷+天气 → +负荷+天气+风光 | 其余全固定 | 协变量对各模型的提升幅度？谁最吃协变量？ |
| B | **上下文长度** | 168 → 336 → 720 | 其余全固定 | 长上下文是否更好？边际收益在哪？ |
| C | **单/多变量** | 单变量 → 多节点联合 | 其余全固定 | 多节点空间相关性谁利用得最好？（Toto 预期占优） |
| D | **预测步长** | 24 → 48 → 168 | 其余全固定 | 长预测谁衰减最慢？ |
| E | **零样本 vs 微调** | 零样本 → 微调（用 NYISO 长历史） | 其余全固定 | 微调值不值得？提升多少？ |
| F | **数据频率** | 1h → 15min → 5min | 其余全固定 | 高频是否带来更难/更易？ |

每个消融的产出：一张"指标 vs 旋钮取值"的表 + 折线图，横轴是旋钮，纵轴分别是 MAE、RMSE、Spike-F1，每条线一个模型。

> 工程提示：消融矩阵用**配置驱动**（YAML/JSON 列表），由 `run_experiment(config)` 批量执行，避免再出现硬编码、各脚本不一致的问题。

---

## 7. 从消融到融合模型

消融的最终目的，是回答"每个模型的强项板块是什么"，进而设计融合模型。

### 7.1 识别强项板块

把消融结果整理成一张"能力地图"，例如（示意，以实际结果为准）：

| 场景维度 | 预期强者 | 依据来自消融 |
|---|---|---|
| 有协变量 | Chronos-2 | 消融A |
| 多节点空间相关 | Toto | 消融C |
| 纯单变量短期 | TimesFM | 消融B/D |
| 长预测步长 | 待定 | 消融D |
| 尖峰预警(Spike-F1) | 待定 | 各消融的 Spike-F1 列 |

### 7.2 融合策略（按复杂度递进）

1. **简单加权平均**：按各模型在验证集上的表现加权。最快验证融合是否有效。
2. **场景路由（routing）**：根据当前场景（是否高波动、是否临近已知尖峰）选用最强模型。直接利用 §7.1 的能力地图。
3. **Stacking 集成**：用一个元学习器（如 LightGBM）以各基模型预测为特征，学习最优组合。通常效果最好，但需额外训练集与防泄露切分。
4. **模块级融合（进阶）**：若有精力，可借鉴各模型强项的架构思想（如 Toto 的多变量注意力、Chronos 的协变量编码）做更深的组合。这属于探索性工作。

### 7.3 融合模型的评估

融合模型必须放进**完全相同的滚动回测 + 全套指标（含 Spike-F1）**里，与单模型公平对比。只有在多起报点上稳定优于最佳单模型，才能下"融合有效"的结论。

---

## 8. 执行路线与产出清单

| 阶段 | 内容 | 预计 | 关键产出 | 验证标准 |
|---|---|---|---|---|
| 0 | 数据勘探 | 0.5天 | 数据清单（市场/节点/列/时间范围/能否对齐） | 确认协变量能与电价时间戳对齐 |
| 1 | 统一数据加载器 + 节点清单 | 1天 | `load_slice()`、`configs/nodes.yaml` | 一行代码取出"某节点+协变量"对齐数据 |
| 2 | 统一框架 + 滚动回测 + 指标 | 2天 | `Forecaster` 接口、`run_experiment()`、指标模块（含 Spike-F1） | 所有模型走同一评估路径，跑通基准配置 |
| 3 | 消融矩阵 A–F | 3天 | 结果总表 + 每因素对比图 | 6 个消融各出一张图一张表 |
| 4 | 模块分析与融合 | 按结果定 | 能力地图、融合模型、对比报告 | 融合在多起报点上稳定优于最佳单模型 |

---

## 9. 常见陷阱清单（务必逐条自查）

1. **数据泄露**：Spike 阈值 P95、需训练模型的 fit、归一化统计量——全部只能用起报点之前的数据。
2. **单点测试**：任何结论都要多起报点，报告均值±标准差。
3. **能力被压平**：别让 Chronos-2/Toto 默认跑单变量；要显式启用并标记能力。
4. **不公平对比**：零样本与训练类必须在同一滚动回测、同一上下文可见范围下比。
5. **指标单一**：MAE 看不出尖峰能力；务必并列报告 Spike-F1（Precision/Recall/F1）。
6. **节点不固定**：节点清单要固化进配置，否则消融结果不可比。
7. **量纲不可比**：跨节点/市场比较时用 MASE 而非裸 MAE。
8. **硬编码参数**：所有旋钮走配置，杜绝散落在脚本里的魔法数字。

---

## 附录 A：目录结构建议

```
school/
├── configs/
│   ├── nodes.yaml              # 固化的节点清单
│   └── experiments/           # 各消融实验的配置文件
│       ├── ablation_A_covariates.yaml
│       ├── ablation_B_context.yaml
│       └── ...
├── src/
│   ├── data_processing/
│   │   └── loader.py          # load_slice() 统一数据加载器
│   ├── models/
│   │   ├── base.py            # Forecaster 统一接口
│   │   ├── baselines.py       # RF/XGB/LSTM... (改造为统一接口)
│   │   └── foundation.py      # TimesFM/Chronos/Toto 适配器
│   ├── evaluation/
│   │   ├── backtest.py        # 滚动回测引擎
│   │   ├── metrics.py         # MAE/RMSE/MASE/Pinball/Spike-F1
│   │   └── plotting.py
│   └── forecasting/
│       └── run_experiment.py  # 配置驱动的实验入口
├── data/
│   └── results/               # 按实验编号分目录存结果
└── docs/specs/
    └── experiment_manual.md   # 本手册
```

## 附录 B：实验配置文件示例

```yaml
# configs/experiments/ablation_A_covariates.yaml
name: "ablation_A_covariates"
market: ERCOT
nodes_group: volatility        # 引用 nodes.yaml 中的清单
freq: 1h
context_len: 168
horizon: 24
backtest:
  test_start: "2025-08-01"
  test_end:   "2025-09-30"
  stride_hours: 24             # 每天一个起报点
models: [Naive, SeasonalNaive, ETS, Theta, RandomForest, LightGBM, XGBoost, TimesFM, Chronos2, Toto]
spike:
  quantile: 0.95
  threshold_mode: global       # global / rolling
  signal: [mean, q90]          # 两种判定都报告
# 被消融的旋钮：协变量集合
ablate:
  key: covariates
  values:
    - []
    - [load]
    - [load, temperature]
    - [load, temperature, wind, solar]
```
