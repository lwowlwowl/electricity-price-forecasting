# 电价预测项目

基于时序基础模型（TimesFM-2.5 / Chronos-2 / Toto-1.0）与多种基线，在 ERCOT 实时电价上做**多起报点滚动回测（walk-forward backtest）**的对比实验框架。

整套实验配置驱动：市场、节点、频率、上下文长度、预测步长、起报点、协变量、模型清单全部写在 YAML 里，一条命令跑完「取数 → 选节点 → 构造模型 → 滚动回测 → 落盘结果」。

## 设计要点

- **统一接口**：所有模型实现同一个 `Forecaster` 抽象（`src/models/base.py`），带能力标志（是否需训练、是否支持协变量、是否支持多变量），不支持的能力自动降级。基线和基础模型因此能在同一回测里公平对比。
- **无泄漏滚动回测**：每个起报点只用该点之前的历史（`data.iloc[oi-ctx_len : oi]`）作上下文，需训练的模型每个起报点重新 fit，绝不触碰未来数据。
- **基础模型隔离运行**：TimesFM / Chronos-2 / Toto 各装在独立 venv（依赖互相冲突），主程序通过 npz 文件 + 子进程 worker 与它们通信，互不污染。
- **丰富指标**：MAE、RMSE、MASE（以 SeasonalNaive 为分母）、sMAPE、Pinball、Coverage（q10–q90），以及电价场景关键的尖峰 F1（Spike-F1，P95 阈值）。

## 项目结构

```
school/
├── configs/
│   ├── nodes.yaml                      # 各市场代表性节点清单（由脚本生成）
│   └── experiments/                    # 实验配置（每个 YAML 一个实验）
│       ├── baseline.yaml               # 基准配置，所有消融的出发点
│       ├── foundation_smoke.yaml       # 基础模型冒烟测试
│       └── ablation_B_context.yaml     # 消融B：上下文长度
│
├── src/
│   ├── data_processing/
│   │   ├── loader.py                   # load_slice：从 raw 长表按需切片成宽表（主链路取数）
│   │   └── build_nodes_config.py       # 统计各节点波动/尖峰，生成 nodes.yaml
│   ├── models/
│   │   ├── base.py                     # Forecaster 抽象基类 + Forecast 数据结构
│   │   ├── forecasters.py              # 7 个基线：Naive/SeasonalNaive/ETS/Theta + RF/LightGBM/XGBoost
│   │   ├── foundation.py               # 3 个基础模型的子进程适配器
│   │   └── workers/                    # 各基础模型在自己 venv 里运行的 worker 脚本
│   ├── evaluation/
│   │   ├── backtest.py                 # 滚动回测引擎（起报点生成、无泄漏预测、指标计算）
│   │   ├── metrics.py                  # MAE/RMSE/MASE/sMAPE/Pinball/Coverage/Spike-F1
│   │   └── plotting.py                 # 单实验汇总图 + 消融旋钮扫描图
│   └── forecasting/
│       ├── run_experiment.py           # 配置驱动的单实验入口
│       └── run_ablation.py             # 消融执行器（扫一个旋钮的多个取值）
│
├── analysis/                           # 探索性数据画像脚本与图
├── external/                           # 三个基础模型（各含独立 .venv，需自行克隆+建环境）
│   ├── timesfm/                        #   git clone https://github.com/google-research/timesfm.git
│   ├── chronos-forecasting/            #   git clone https://github.com/amazon-science/chronos-forecasting.git
│   └── toto/                           #   git clone https://github.com/DataDog/toto.git
├── hf_cache/                           # HuggingFace 模型权重缓存（离线可用）
├── data/
│   ├── raw/<市场>/processed/           # 原始长表（loader 直接读取，如 actual_price_hourly.csv）
│   └── results/<实验名>/               # 每个实验的输出（summary.csv / per_origin.csv / 图）
└── docs/                               # 实验手册、概念说明、参考文献
```

## 快速开始

### 1. 准备基础模型环境（首次）

三个基础模型各自克隆到 `external/` 并在各自目录建立 `.venv`（依赖见各自仓库）。模型权重会缓存在 `hf_cache/`。若只跑基线，可跳过这一步。

### 2. （可选）重新生成节点清单

```bash
python src/data_processing/build_nodes_config.py ERCOT
```

按波动性/尖峰频次/平稳度把各市场节点分成 `volatility` / `spikes` / `stable` 三组，固化到 `configs/nodes.yaml`，保证不同实验间节点一致、结果可比。

### 3. 跑一个实验

```bash
# 基准实验（7 基线 + 3 基础模型）
python src/forecasting/run_experiment.py configs/experiments/baseline.yaml

# 只验证基础模型是否跑通（更快）
python src/forecasting/run_experiment.py configs/experiments/foundation_smoke.yaml
```

结果写入 `data/results/<实验名>/`，含 `summary.csv`（各模型指标汇总，按 MAE 升序）、`per_origin.csv`（逐起报点明细）、`thresholds.json`（尖峰阈值）。

> 后台运行：基础模型推理较慢，可放后台。建议加 `-u` 让日志实时可见：
> ```bash
> nohup python3 -u src/forecasting/run_experiment.py configs/experiments/baseline.yaml > run.log 2>&1 &
> ```

### 4. 跑消融实验

消融配置在基准配置基础上加一个 `ablate` 块，指定要扫的旋钮和取值：

```bash
python src/forecasting/run_ablation.py configs/experiments/ablation_B_context.yaml
```

会对该旋钮的每个取值各跑一次回测，输出合并对比表 `ablation_summary.csv` 和旋钮扫描折线图。

## 配置说明

实验配置（`configs/experiments/*.yaml`）的主要字段：

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
| TimesFM-2.5 | 时序基础模型 | ❌ | 独立 venv 子进程 |
| Chronos-2 | 时序基础模型 | ❌ | 独立 venv 子进程 |
| Toto-1.0 | 时序基础模型 | ❌ | 独立 venv 子进程 |

## 注意事项

- **基础模型 venv**：三个基础模型依赖互相冲突，各用一个 `external/<model>/.venv`，由 `foundation.py` 通过子进程调用，不要在主环境里直接 import。
- **数据来源**：主链路 `loader.load_slice` 直接读 `data/raw/<市场>/processed/` 下的原始长表并在内存里透视，无需预先生成中间宽表文件。
- **运行目录**：脚本内部用绝对路径定位项目根，从项目根目录运行最稳妥。

## 参考

- TimesFM: https://github.com/google-research/timesfm
- Chronos: https://github.com/amazon-science/chronos-forecasting
- Toto: https://github.com/DataDog/toto

更详细的方法论见 `docs/specs/experiment_manual.md` 与 `docs/concepts/`。
