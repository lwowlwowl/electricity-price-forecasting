# 电价预测项目

基于时序基础模型（TimesFM-2.5 / Chronos-2 / Toto-1.0 / Toto-2.0）与多种基线，在 ERCOT 实时电价上做**多起报点滚动回测（walk-forward backtest）**的对比实验框架。

整套实验配置驱动：市场、节点、频率、上下文长度、预测步长、起报点、协变量、模型清单全部写在 YAML 里，一条命令跑完「取数 → 选节点 → 构造模型 → 滚动回测 → 落盘结果」。

## 设计要点

- **统一接口**：所有模型实现同一个 `Forecaster` 抽象（`src/models/base.py`），带能力标志（是否需训练、是否支持协变量、是否支持多变量），不支持的能力自动降级。基线和基础模型因此能在同一回测里公平对比。
- **无泄漏滚动回测**：每个起报点只用该点之前的历史（`data.iloc[oi-ctx_len : oi]`）作上下文，需训练的模型每个起报点重新 fit，绝不触碰未来数据。
- **基础模型隔离运行**：TimesFM / Chronos-2 / Toto / Toto2 各装在独立 venv（依赖互相冲突，Toto2 与 Toto 共用 venv），主程序通过 npz 文件 + 子进程 worker 与它们通信，互不污染。
- **丰富指标**：MAE、RMSE、MASE（以 SeasonalNaive 为分母）、sMAPE、Pinball、Coverage（q10–q90），以及电价场景关键的尖峰 F1（Spike-F1，P95 阈值）。
- **三市场窗口**：在三种典型电价模式下做全量对比——W1 稳定期（2025 年 8 月）、W2 负电价期（2025 年 3 月）、W3 极端尖峰期（2026 年 1 月），检验模型在不同市场状态下的鲁棒性。

## 项目结构

```
school/
├── configs/
│   ├── nodes.yaml                      # 各市场代表性节点清单（由脚本生成）
│   ├── parameter_ablation/             # 参数消融实验配置（v1.0）
│   │   ├── baseline.yaml               # W1 基准配置，所有消融的出发点
│   │   ├── baseline_w2_negative.yaml   # W2 负电价基准
│   │   ├── baseline_w3_extreme.yaml    # W3 极端尖峰基准
│   │   ├── ablation_A_covariates*.yaml # 消融A：协变量（×3 窗口）
│   │   ├── ablation_B_context*.yaml    # 消融B：上下文长度（×3 窗口）
│   │   ├── ablation_C_multivariate*.yaml # 消融C：单/多变量（×3 窗口）
│   │   ├── ablation_D_horizon*.yaml    # 消融D：预测步长（×3 窗口）
│   │   ├── ablation_E_finetune.yaml    # 消融E：微调（保留）
│   │   ├── ablation_F_frequency*.yaml  # 消融F：数据频率（×3 窗口）
│   │   └── smoke/                      # 冒烟测试配置
│   └── structural_ablation/            # 结构消融实验配置（v2.0）
│       ├── smoke_toto2.yaml            # 冒烟测试
│       ├── full_toto2.yaml             # Toto-2.0 全消融扫描
│       ├── full_chronos2.yaml          # Chronos-2 全消融扫描
│       ├── full_timesfm.yaml           # TimesFM 全消融扫描
│       └── cross_model_attention.yaml  # 跨模型注意力对比
│
├── src/
│   ├── data_processing/
│   │   ├── loader.py                   # load_slice：从 raw 长表按需切片成宽表（主链路取数）
│   │   └── build_nodes_config.py       # 统计各节点波动/尖峰，生成 nodes.yaml
│   ├── models/
│   │   ├── base.py                     # Forecaster 抽象基类 + Forecast 数据结构
│   │   ├── forecasters.py              # 7 个基线：Naive/SeasonalNaive/ETS/Theta + RF/LightGBM/XGBoost
│   │   ├── foundation.py               # 4 个基础模型的子进程适配器
│   │   └── workers/                    # 各基础模型在自己 venv 里运行的 worker 脚本
│   │       ├── worker_timesfm.py
│   │       ├── worker_chronos2.py
│   │       ├── worker_toto.py
│   │       └── worker_toto2.py
│   ├── evaluation/
│   │   ├── backtest.py                 # 滚动回测引擎（起报点生成、无泄漏预测、指标计算）
│   │   ├── metrics.py                  # MAE/RMSE/MASE/sMAPE/Pinball/Coverage/Spike-F1
│   │   └── plotting.py                 # 单实验汇总图 + 消融旋钮扫描图
│   ├── parameter_ablation/             # 参数消融执行器
│   │   ├── run_experiment.py           # 配置驱动的单实验入口
│   │   └── run_ablation.py             # 参数消融执行器（扫一个旋钮的多个取值）
│   └── structural_ablation/            # 结构消融模块（v2.0）
│       ├── ablations.py                # 14 种结构消融操作实现
│       ├── foundation_ablation.py      # 消融适配器（调度 worker 子进程）
│       ├── run_structural_ablation.py  # 结构消融实验入口
│       └── workers/                    # 各模型的消融 worker
│           ├── worker_toto2_ablation.py
│           ├── worker_chronos2_ablation.py
│           └── worker_timesfm_ablation.py
│
├── analysis/                           # 探索性数据画像脚本与图
├── external/                           # 四个基础模型（各含独立 .venv，需自行克隆+建环境）
│   ├── timesfm/                        #   git clone https://github.com/google-research/timesfm.git
│   ├── chronos-forecasting/            #   git clone https://github.com/amazon-science/chronos-forecasting.git
│   └── toto/                           #   git clone https://github.com/DataDog/toto.git（含 Toto-1.0 与 Toto-2.0）
├── hf_cache/                           # HuggingFace 模型权重缓存（离线可用）
├── data/
│   ├── raw/<市场>/processed/           # 原始长表（loader 直接读取，如 actual_price_hourly.csv）
│   └── results/<实验名>/               # 每个实验的输出（summary.csv / per_origin.csv / 图）
├── docs/                               # 实验手册、概念说明、参考文献
│   ├── specs/
│   │   ├── experiment_manual.md        # v1.0 参数级消融手册
│   │   └── experiment_manual_v2.md     # v2.0 结构级消融与模型融合手册
│   ├── 参数消融实验结果与问答.md         # 参数消融全量结果分析（11 模型 × 3 窗口 × 5 消融）
│   └── concepts/                       # 核心概念说明
└── run_all_ablations.sh                # 一键跑全部消融实验的脚本
```

## 实验矩阵

### 市场窗口

| 窗口 | 别名 | 测试区间 | 特征 |
|------|------|----------|------|
| W1 | `w1_stable` | 2025-08-01 ~ 08-31 | 夏季稳定期，价格波动温和 |
| W2 | `w2_negative` | 2025-03-01 ~ 03-31 | 春季低负荷，出现负电价 |
| W3 | `w3_extreme` | 2026-01-01 ~ 01-31 | 冬季极端尖峰，价格剧烈波动 |

### 参数消融维度（v1.0，已完成）

| 消融 | 旋钮 | 扫描取值 | 回答的问题 |
|------|------|----------|------------|
| A | 协变量 | `[]` → `[load]` → `[load,temp]` → `[load,temp,wind,solar]` | 外部信息是否提升预测？ |
| B | 上下文长度 | 168 / 336 / 720 | 看多远的历史最合适？ |
| C | 单/多变量 | False / True | 多节点联合预测是否更优？ |
| D | 预测步长 | 24 / 48 / 168 | 预测越远衰减多快？ |
| F | 数据频率 | 1h / 15min | 高频数据是否有增益？ |

每组消融均在 W1/W2/W3 三个窗口上完成，共 5 × 3 = 15 组实验，全部 11 个模型参与对比。

### 结构消融（v2.0，下一阶段）

在 v1.0 结论基础上，深入模型内部做结构级消融——逐一移除或替换 Transformer 内部组件（注意力层、位置编码、输出头、Patch 机制等），定位每个基础模型在电价预测上的关键组件，最终设计融合模型。详见 `docs/specs/experiment_manual_v2.md`。

## 快速开始

### 1. 准备基础模型环境（首次）

四个基础模型中，TimesFM / Chronos-2 / Toto 各自克隆到 `external/` 并在各自目录建立 `.venv`（依赖见各自仓库）。Toto-2.0 与 Toto-1.0 共用同一个 venv。模型权重会缓存在 `hf_cache/`。若只跑基线，可跳过这一步。

### 2. （可选）重新生成节点清单

```bash
python src/data_processing/build_nodes_config.py ERCOT
```

按波动性/尖峰频次/平稳度把各市场节点分成 `volatility` / `spikes` / `stable` 三组，固化到 `configs/nodes.yaml`，保证不同实验间节点一致、结果可比。

### 3. 跑一个实验

```bash
# 基准实验（7 基线 + 4 基础模型，W1 稳定期）
python src/parameter_ablation/run_experiment.py configs/parameter_ablation/baseline.yaml

# W2 负电价窗口
python src/parameter_ablation/run_experiment.py configs/parameter_ablation/baseline_w2_negative.yaml

# W3 极端尖峰窗口
python src/parameter_ablation/run_experiment.py configs/parameter_ablation/baseline_w3_extreme.yaml
```

结果写入 `data/results/<实验名>/`，含 `summary.csv`（各模型指标汇总，按 MAE 升序）、`per_origin.csv`（逐起报点明细）、`thresholds.json`（尖峰阈值）。

> 后台运行：基础模型推理较慢，可放后台。建议加 `-u` 让日志实时可见：
> ```bash
> nohup python3 -u src/parameter_ablation/run_experiment.py configs/parameter_ablation/baseline.yaml > run.log 2>&1 &
> ```

### 4. 跑消融实验

消融配置在基准配置基础上加一个 `ablate` 块，指定要扫的旋钮和取值：

```bash
# 单组消融
python src/parameter_ablation/run_ablation.py configs/parameter_ablation/ablation_B_context.yaml

# 一键跑全部消融（所有窗口 × 所有维度）
bash run_all_ablations.sh
```

会对该旋钮的每个取值各跑一次回测，输出合并对比表 `ablation_summary.csv` 和旋钮扫描折线图。

## 配置说明

实验配置（`configs/parameter_ablation/*.yaml`）的主要字段：

| 字段 | 含义 |
|------|------|
| `market` / `nodes_group` | 市场（如 ERCOT）与节点组（`volatility`/`spikes`/`stable`，引用 nodes.yaml）|
| `freq` | 频率（`1h` / `15min` / `5min`）|
| `context_len` | 回看窗口（零样本/统计/基础模型用）|
| `train_context_len` | 需训练模型（树模型）的训练回看窗口，更长，仍无泄漏 |
| `horizon` | 预测步长（如 24＝日前预测）|
| `backtest.*` | 测试区间、起报点步长 `stride_hours`、起报点数量上限 `max_origins` |
| `models` | 参与对比的模型名列表 |
| `covariates` | 协变量列表（消融A 在此增减）|
| `spike` | 尖峰定义（分位阈值 + global/rolling 模式）|

## 可用模型

| 模型 | 类型 | 需训练 | 运行位置 |
|------|------|:---:|------|
| Naive | 随机游走 | ❌ | 进程内 |
| SeasonalNaive | 季节性朴素（MASE 分母）| ❌ | 进程内 |
| ETS | 指数平滑 / Holt-Winters | ❌ | 进程内 |
| Theta | Theta 法（M3 冠军）| ❌ | 进程内 |
| RandomForest | 随机森林 | ✅ | 进程内（每起报点重训）|
| LightGBM | 梯度提升树 | ✅ | 进程内（每起报点重训）|
| XGBoost | 梯度提升树 | ✅ | 进程内（每起报点重训）|
| TimesFM-2.5 | 时序基础模型（Google）| ❌ | 独立 venv 子进程 |
| Chronos-2 | 时序基础模型（Amazon）| ❌ | 独立 venv 子进程 |
| Toto-1.0 | 时序基础模型（Datadog）| ❌ | 独立 venv 子进程 |
| Toto-2.0 | 时序基础模型（Datadog，CPM 架构）| ❌ | 独立 venv 子进程（共用 Toto venv）|

## 注意事项

- **基础模型 venv**：四个基础模型依赖互相冲突，TimesFM / Chronos-2 / Toto 各用一个 `external/<model>/.venv`，Toto-2.0 与 Toto-1.0 共用 toto 的 venv。由 `foundation.py` 通过子进程调用，不要在主环境里直接 import。
- **数据来源**：主链路 `loader.load_slice` 直接读 `data/raw/<市场>/processed/` 下的原始长表并在内存里透视，无需预先生成中间宽表文件。
- **运行目录**：脚本内部用绝对路径定位项目根，从项目根目录运行最稳妥。
- **实验结果**：所有结果按 `data/results/<实验名_窗口后缀>/` 组织，每个目录包含 summary.csv、per_origin.csv、records.csv、thresholds.json 及对比图（summary_compare.png / timeseries_compare.png）。

## 实验进度

- ✅ **v1.0 参数消融**：全部完成。5 类消融 × 3 窗口 = 15 组实验，11 个模型全部跑完，结果文档见 `docs/参数消融实验结果与问答.md`。
- 🔧 **v2.0 结构消融**：代码框架已就位（`src/structural_ablation/`），待模型 venv 就绪后执行。方案见 `docs/specs/experiment_manual_v2.md`。

### 5. 跑结构消融实验

```bash
# 冒烟测试
python src/structural_ablation/run_structural_ablation.py configs/structural_ablation/smoke_toto2.yaml

# 完整实验
python src/structural_ablation/run_structural_ablation.py configs/structural_ablation/full_toto2.yaml
```

## 参考

- TimesFM: https://github.com/google-research/timesfm
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Toto: https://github.com/DataDog/toto
- Toto-2.0 论文: `docs/reference/toto2.pdf`

更详细的方法论见 `docs/specs/experiment_manual.md` 与 `docs/concepts/`。
