# Electricity Price Forecasting

本项目的最终目标是：**提高储能在电力市场中的可执行净收益**，而不只是降低电价预测误差。

## 当前主线

老师要求将日前（DA）与实时（RT）预测拆成独立模型，并先完成DA。

当前独立建模主线已经落地：

1. 固定D-1 10:00起报的24小时DA样本；
2. 季节性、ExtraTrees、GRU收益基线；
3. 独立DA Transformer、零阶决策损失和收益选模入口；
4. 每小时起报、预测4小时且只执行第1步的独立RT Transformer；
5. 两个模型各自拥有独立权重和可编辑draw.io架构图。

ERCOT正常情况下在交付日前一天10:00 Central Prevailing Time开始DAM清算，结果最迟13:30发布。当前DA样本已按D-1日10:00截断输入，并预测D日完整24小时交付日；旧v8相邻滑窗只保留作历史对照。

实施记录从`docs/progress/00_执行索引.md`进入；当前阶段总结和统一比较见文档23、24。
文档07只保留第一轮独立DA/RT流水线的历史验收。`train_formal.py`仍是旧联合模型入口，
新实验应使用当前三个独立模型或联合V2入口。

## 只看这些活跃目录

```text
configs/decision_aware/       当前模型配置
src/decision_aware/           Transformer、损失、策略、数据入口
src/models/                   季节性/树/RNN/Foundation基线适配
src/evaluation/               回测与指标
scripts/decision_aware/       当前审计、训练、比较入口
tests/                        数据与收益正确性测试
docs/PRD_DA_RT独立建模与跨模态融合.md
docs/transformer_3day_learning_plan.md
docs/model_architecture.drawio
docs/training_loop.drawio
docs/da_model_architecture.drawio
docs/rt_model_architecture.drawio
docs/progress/                本轮按顺序整理的实施结论
data/markets/                 唯一正式市场数据入口
external/                     三个原始模型及worker环境
```

根目录的 `transformer_from_scratch.md` 是Transformer入门说明，`TODO-WIN.md` 记录尚未解决的代码问题。
`runtime_cache/` 只保存可重新生成的安装缓存、基础模型worker缓存和PDF预览，已被Git忽略。

## 归档位置

```text
docs/archive/                 历史TODO、实验手册、结果和汇报
scripts/archive/covariates/   旧协变量下载、合并和筛选代码
scripts/archive/decision_aware_legacy/
src/archive/                  旧数据加载与旧实验实现
configs/archive/              旧参数/结构消融配置
```

归档内容默认不运行、不维护。需要追溯旧结论时再查，不要从归档路径启动新实验。

## 数据规则

- 正式训练只读取 `data/markets/<MARKET>/` 下的Parquet；
- 不再恢复顶层 `data/raw/` 或 `data/unified/`；
- 不修改团队提供的Excel或Parquet；
- 当前五条输入流为历史DA价格、历史RT价格、历史负荷、历史风光和本地日历；
- 未来实际负荷、风光、天气不能冒充起报时可用的预报。

数据时间戳与 `hour_local` 一致，当前按小时区间起点解释；Parquet本身没有保存区间起止语义的来源元数据，这一限制必须保留在报告中。

## 常用检查

审计当前DA样本：

```powershell
& external/toto/.venv/Scripts/python.exe -X utf8 scripts/decision_aware/audit_da_sample.py
```

运行数据与收益单元测试：

```powershell
& external/toto/.venv/Scripts/python.exe -X utf8 -m pytest -q `
  tests/test_decision_aware_dataset_da.py `
  tests/test_decision_aware_dataset_v3.py `
  tests/test_decision_aware_policy.py
```

独立模型smoke检查：

```powershell
& external/toto/.venv/Scripts/python.exe -X utf8 scripts/decision_aware/train_da.py --smoke
& external/toto/.venv/Scripts/python.exe -X utf8 scripts/decision_aware/train_rt.py --smoke
```

环境依赖见 `requirements.txt` 和 `requirements-windows-cuda.txt`。

## 相关开源仓库与文献入口

- Chronos-2：[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- TimesFM：[google-research/timesfm](https://github.com/google-research/timesfm)
- Toto：[DataDog/toto](https://github.com/DataDog/toto)
- 本项目已经收集的论文、每篇论文带来的思路以及下一阶段待读清单：
  [`docs/reference/README.md`](docs/reference/README.md)
