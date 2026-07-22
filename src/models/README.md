# src/models — 预测器层

## 文件

| 文件 | 角色 | 新方向中的用途 |
|------|------|---------------|
| `base.py` | `Forecaster` / `Forecast` 抽象基类 | 所有预测器（基线 + 新 DA-TSFM）的统一接口 |
| `forecasters.py` | 7 个统计/树基线（Naive / SeasonalNaive / ETS / Theta / LEAR / RandomForest / LightGBM / XGBoost） + `build_forecaster()` 工厂 | 弱基线对照 |
| `foundation.py` + `workers/` | 4 个外部 TSFM（TimesFM / Chronos / Toto）的子进程适配器 | **外部 TSFM 基线层**，双重用途（见下） |

## foundation.py + workers/ 的新角色

旧范式：用预训练权重做 zero-shot 预测。
新方向（W9 §7）保留这层，双重用途：

1. **零样本参考基线**（§7.2 "不一定下载权重" 的口子）—— 结果表里作参考列，展示"本工作 100M 从零训练 vs TimesFM 200M 零样本"。TSFM 论文标配对照。
2. **从零训练对照的 plumbing 模板**（§7.1 主路径）—— 公平对比需用各模型**训练代码**从零训练（同参数量、同数据）。各外部模型依赖冲突、需独立 venv、子进程调用的隔离工程范式，直接复用 `workers/`，新增 "train-from-scratch" worker 克隆即可。

> 若老师明确"只认从零训练、零样本对照不做"，则本层降级为归档（移回 `src/archive/`）。当前按 §7.2 "不一定" 措辞保留。

## 守卫导入

`forecasters.py:661` 与 `evaluation/backtest.py:43` 对 `foundation` 的引用均为延迟 + try/except 守卫：

```python
try:
    from foundation import FOUNDATION_REGISTRY   # / FoundationForecaster
except ImportError:
    pass   # 或 FoundationForecaster = ()
```

未安装大模型依赖时，统计/树基线与回测框架仍可跑，仅冻结模型选项不可用。

## 依赖

- `base.py` / `forecasters.py`：主环境（pandas / numpy / sklearn / lightgbm / xgboost / statsmodels）。
- `foundation.py` / `workers/`：各 `external/<model>/.venv` 独立环境，子进程调用，**不要在主环境 import**。
