# 电价预测项目

基于 EIA 数据集 和 ENLITEN 数据集的多模型电价预测对比实验。

## 📁 项目结构

```
school/
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   │   ├── EIA/                # EIA 数据
│   │   └── ENLITEN-Grid-Econ-Data/  # ENLITEN 数据集
│   ├── processed/              # 处理后数据
│   │   └── lmp_processed.csv   # 预处理后的电价数据
│   └── results/                # 预测结果
│       ├── forecast_baselines.csv
│       ├── forecast_toto.csv
│       ├── forecast_timesfm.csv
│       ├── forecast_chronos2.csv
│       ├── forecast_comparison.png
│       └── all_models_comparison.csv
│
├── src/                        # 源代码
│   ├── data_processing/        # 数据处理
│   │   └── prepare.py          # 数据预处理脚本
│   ├── models/                 # 模型定义
│   │   └── baselines.py        # Baseline 模型（RF/XGBoost/LSTM等）
│   ├── forecasting/            # 预测脚本
│   │   └── batch.py            # 批量实验（多时间点+多节点选择）
│   └── evaluation/             # 评估工具
│       ├── plotting.py         # 可视化
│       └── tables.py           # 生成对比表格
│
├── experiments/                # 实验脚本
│   ├── run_all.py              # 一键运行所有模型
│   └── run_batch_experiments.py # 批量对比实验（多时间点×节点选择）
│
├── external/                   # 第三方预训练模型（需从以下仓库克隆）
│   ├── chronos-forecasting/    # Chronos-2 模型
│   │   └── `git clone https://github.com/amazon-science/chronos-forecasting.git`
│   ├── timesfm/                # TimesFM 2.5 模型
│   │   └── `git clone https://github.com/google-research/timesfm.git`
│   └── toto/                   # Toto 1.0 模型
│       └── `git clone https://github.com/DataDog/toto.git`
│
├── hf_cache/                   # HuggingFace 缓存（模型权重）
├── docs/                       # 文档资料
│   ├── meetingminutes/         #     会议纪要
│   └── reference/              #     参考资料
└── .gitignore                  # Git 忽略配置
```

## 🚀 快速开始

### 1. 数据预处理

```bash
cd src/data_processing
../../external/toto/.venv/bin/python prepare.py
```

### 2. 运行 Baseline 模型

```bash
cd src/models
../../external/toto/.venv/bin/python baselines.py
```

### 3. 运行预训练模型（可选）

```bash
# TimesFM
cd external/timesfm
.venv/bin/python forecast_timesfm.py

# Chronos-2
cd external/chronos-forecasting
.venv/bin/python forecast_chronos2.py

# Toto
cd external/toto
.venv/bin/python forecast_toto.py
```

### 4. 可视化

```bash
cd src/evaluation
../../external/toto/.venv/bin/python plotting.py
../../external/toto/.venv/bin/python tables.py
```

### 5. 一键运行所有（推荐）

```bash
cd experiments
python run_all.py
```

### 6. 批量对比实验（高级）

运行多个时间点、多种节点选择策略的组合实验：

```bash
cd experiments
python run_batch_experiments.py
```

支持配置：
- **时间节点**: 年终、夏季高峰、节假日等
- **节点选择**: 波动最大、尖峰最多、随机
- **自动汇总**: 生成对比表格

## 📊 实验配置

### 节点选择策略

- **volatility**: 标准差最大的节点（波动最大）
- **spikes**: 尖峰最多的节点（超过均值+2倍标准差的次数）
- **random**: 随机选择

### 时间点配置

在 `src/forecasting/batch.py` 中修改 `EXPERIMENTS` 列表：

```python
EXPERIMENTS = [
    ("年终-波动大", "2020-12-31 00:00", "volatility"),
    ("夏季高峰-尖峰", "2020-07-15 00:00", "spikes"),
    # 添加你自己的...
]
```

## 🧪 可用模型

| 模型 | 类型 | 是否需要训练 |
|------|------|-------------|
| Naive | 统计规则 | ❌ |
| Seasonal-Naive | 统计规则 | ❌ |
| RandomForest | 机器学习 | ✅ |
| XGBoost | 梯度提升 | ✅ |
| LightGBM | 梯度提升 | ✅ |
| MLP | 神经网络 | ✅ |
| LSTM | 深度学习 | ✅ |
| GRU | 深度学习 | ✅ |
| TimesFM-2.5 | 预训练大模型 | ❌ |
| Chronos-2 | 预训练大模型 | ❌ |
| Toto-1.0 | 预训练大模型 | ❌ |

## 📈 结果查看

所有结果保存在 `data/results/` 目录：

- **CSV 文件**: 各模型的预测值和误差指标
- **PNG 图片**: 可视化对比图
- **汇总表格**: `all_models_comparison.csv`

## ⚠️ 注意事项

1. **虚拟环境**: 每个预训练模型有自己的虚拟环境（`.venv`），在 `external/` 下
2. **依赖安装**: Baseline 模型需要安装 xgboost/lightgbm/tensorflow：
   ```bash
   external/toto/.venv/bin/pip install xgboost lightgbm tensorflow
   ```
3. **数据路径**: 所有脚本使用相对路径，确保在正确目录运行

## 📄 文件说明

| 文件 | 功能 |
|------|------|
| `src/data_processing/prepare.py` | 预处理 LMP.xlsx，生成 lmp_processed.csv |
| `src/models/baselines.py` | 运行所有 baseline 模型 |
| `src/forecasting/batch.py` | 批量实验（多时间点+节点选择） |
| `src/evaluation/plotting.py` | 生成对比图（3个深度学习模型） |
| `src/evaluation/tables.py` | 生成对比表格（所有模型） |
| `experiments/run_all.py` | 一键运行完整流程 |

## 🔧 开发计划

- [ ] 添加更多节点选择策略（如：价格最高、变化率最大）
- [ ] 支持外部特征（天气、节假日）
- [ ] 超参数自动调优
- [ ] 交叉验证
- [ ] 模型持久化（保存/加载训练好的模型）

## 📚 参考

- ENLITEN 数据集: [ENLITEN-Grid-Econ-Data](https://github.com/ehatamis/ENLITEN)
- TimesFM: [Google Research](https://github.com/google-research/timesfm)
- Chronos: [Amazon](https://github.com/amazon-science/chronos-forecasting)
- Toto: [Datadog](https://github.com/DataDog/toto)
