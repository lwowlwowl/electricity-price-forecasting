# scripts/archive — 旧范式驱动脚本归档

> 状态：**已废弃**。新方向数据管线见 `scripts/covariates/`（活跃保留）。

## 归档内容

| 脚本 | 原职责 |
|------|--------|
| `run_overnight.sh` | ElecFM 多实验串行 |
| `run_fusion_sweep.sh` | ElecFM 参数扫描 |
| `run_lora_quick_test.sh` | LoRA 快速测试 |
| `run_significance_tests.py` | 旧消融显著性检验驱动（可复用的统计检验逻辑已在 `src/evaluation/stat_tests.py`） |
| `plot_fusion_versions.py` | ElecFM 各版本结果绘图 |
| `generate_architecture_report.py` | 旧架构报告生成 |
| `run_all_ablations.sh` | v1.0/v2.0 全消融一键脚本（原位于仓库根） |

## 保留在主线

- `scripts/covariates/` — 协变量数据管线（下载/对齐/合并/model_ready/LLM 新闻特征），**新方向多模态 Encoder 的数据来源**，全部保留。
