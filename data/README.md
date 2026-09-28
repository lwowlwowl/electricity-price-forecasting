# 数据目录约定

当前正式训练只认一个市场数据入口：

```text
data/markets/
├── ERCOT/
├── CAISO/
├── NYISO/
└── PJM/
```

每个市场目录包含同名 Excel 和 Parquet：

- Parquet 是代码读取的正式数据；
- Excel 是便于人工查看的 14 列核心字段版本，不参与训练；
- Parquet 有 31 列，在核心字段之外还包含 17 列实际/预报天气；
- 这些大文件由 `.gitignore` 排除，不会上传 GitHub；
- 不再创建顶层 `data/raw/` 或 `data/unified/`。

其他目录：

- `data/results/`：实验结果；
- `data/checkpoints/`：本地模型权重。

旧协变量代码已移到 `scripts/archive/covariates/`，旧说明已移到
`docs/archive/`。不再保留容易被误认为正式数据入口的 `data/covariates/`。

不要修改团队提供的 Excel 或 Parquet 内容。若将来收到新版数据包，保持相同的市场子目录和文件命名，替换前先备份旧包。

## 2026-09-28 只读检查结果

- CAISO：168,765 行，3 个节点；
- ERCOT：843,825 行，15 个节点；
- NYISO：618,805 行，11 个节点；
- PJM：1,968,925 行，35 个节点；
- 四份 Parquet 的 `(timestamp_utc, node)` 都没有重复；
- DA、RT、负荷、风光、日历和 17 列天气字段都没有缺失值；
- 当前训练节点 `ERCOT/LZ_LCRA` 存在，加载后得到 56,251 个小时点和 13 个当前模型输入/目标字段。
- `loader_v2.py` 已支持四个市场；实际训练哪个市场由配置中的 `market` 和 `node` 决定。
