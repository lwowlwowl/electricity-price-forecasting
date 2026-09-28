# 09｜CUDA环境验证与XGBoost基线

## CUDA验证结论

2026-09-28在项目根目录`.venv`完成验证：

- Python 3.12.6；PyTorch 2.12.0+cu126；CUDA可用；
- GPU为NVIDIA GeForce RTX 4070 Laptop，显存8188MiB；
- 独立DA模型在GPU上完成前向、Huber损失、反向传播和一次AdamW参数更新；
- 真实数据CUDA smoke跑6个epoch，第6轮`beta=0.0833`，零阶梯度范数21.69；
- 独立RT真实数据CUDA smoke跑2个epoch，滚动执行和连续SOC回测正常结束；
- 该阶段全部项目测试为`34 passed`；后续新增双结算测试后的当前总数见执行索引。

smoke使用32个训练日、16个验证日和16个测试日，只证明链路正确。测试收益不能与完整333日基线比较，也不能作为模型有效性的结论。结果保存在`data/results/da_transformer_cuda_smoke.json`。
RT smoke结果保存在`data/results/rt_transformer_cuda_smoke.json`，同样只验证链路。

## pandas 3兼容修复

新环境安装pandas 3.0.6后，`DatetimeIndex.asi8`使用微秒，而 `Timedelta.value`仍是纳秒。旧代码把两者直接比较，错误地把连续1小时判成缺口，导致DA、RT和旧v3数据集都生成0个窗口。现已在比较前显式转换为纳秒；没有修改 Excel、Parquet、时间合同或样本内容。

## XGBoost完整基线

干净环境还暴露出`requirements.txt`遗漏scikit-learn；XGBoost的24维输出需要 `MultiOutputRegressor`。依赖已补齐，Windows外层并行改为稳定的单进程拟合。

相同DA合同、相同HardTopK和BESS参数下，完整测试集333天结果为：

- MAE：40.10；RMSE：141.38；
- 平均DA-only代理收益：114.79；
- LP Oracle平均收益：192.47；平均regret：77.68；
- 正收益日比例：95.20%；CPU训练约330秒。

XGBoost的预测误差比周期基线差，但平均收益略高于周期基线112.06。这再次说明本项目不能用MAE/RMSE代替收益选模。原始结果在
`data/results/da_xgboost_v1.json`。

## 下一步

环境、测试和强基线已经就绪。完整DA Transformer三个随机种子的正式训练也已
完成，结果见`10_DA正式训练三随机种子.md`。下一项是融合消融。完整双结算未
接通前，DA收益仍只能标为DA-only代理收益。
