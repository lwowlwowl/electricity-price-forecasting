# 旧 `data/raw` 源代码

这里集中保存已经退出主流程的文件：

- `loader.py`：读取 `data/raw/<市场>/processed/*.csv` 的旧加载器；
- `build_nodes_config.py`：从旧长表 CSV 重新生成节点配置；
- `dataset.py`：v1/v2 单实时价格数据集；
- `forecaster.py`：与上述旧数据集配套的预测器封装。
- `train.py`：与 v1/v2 Dataset 配套的旧训练和评估循环。

它们只用于追溯历史实验，不再由 `decision_aware` 包导入，也不保证可以直接运行。
当前训练使用 `src/decision_aware/loader_v2.py` 和 `dataset_v3.py`。
