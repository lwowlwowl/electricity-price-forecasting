# 历史协变量脚本

本目录保存旧的协变量下载、对齐、合并、筛选与 `data/raw` / `model_ready`
生成脚本，仅用于追溯历史实验，默认不再运行或维护。

当前正式训练只读取：

`data/markets/<MARKET>/<MARKET>_统一小时数据_20200101_20260601.parquet`

统一 Parquet 已包含当前模型使用的 DA、RT、负荷、风光、日历与天气字段。
不要用这里的脚本覆盖或修改 `data/markets/` 下的 Excel、Parquet。

本目录保持一层，不再按下载源或 `raw` 状态细分，方便查找。
