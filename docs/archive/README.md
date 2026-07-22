# docs/archive — 旧范式文档归档

> 状态：**已废弃**。新方向文档见 `docs/todo3.md` 与 `docs/team_notes/w8`、`w9`。
> 保留这些文档是因为其中包含已沉淀的实验结论与方法论，写新模型时仍需回查。

## 归档内容

| 文件/目录 | 内容 | 复用价值 |
|-----------|------|----------|
| `fusion/` | ElecFM 融合模型设计/实验/实现文档 | 架构思路（spike head 分叉点选择）可参考 |
| `elecfm/` | 更早的 ElecFM 历史文档 | 仅存档 |
| `参数消融汇报材料.md` + `参数消融实验结果与问答.md` | v1.0 输入配置消融完整分析 | **结论可直接指导新模型输入设计**：喂哪些协变量、context 长度、频率 |
| `结构消融汇报材料.md` + `结构消融实验结果与问答.md` | v2.0 结构消融 + ElecFM 完整研究记录 | 方法论（Wilcoxon、逐层消融）可迁移；FFN 是最关键组件等结论 |
| `融合模型汇报.md` | ElecFM 汇总汇报 | 仅存档 |
| `TODO.md` + `todo2.md` | 旧阶段待办 | 仅存档 |

## 保留在主线的 docs 内容

- `docs/todo3.md` — 当前决策清单（Decision-aware 模型搭建）
- `docs/covariate_conclusions.md` — 协变量结论（多模态输入设计依据）
- `docs/新闻特征初步汇报材料.md` — LLM 新闻特征（Event Encoder 可选输入）
- `docs/model_architecture*.drawio` + `gen_drawio_v4.py` — **新方向架构图**（基于 w9 材料，含 Head_DA/Head_RT、双结算、代理梯度）
- `docs/specs/`、`docs/concepts/`、`docs/reference/`、`docs/team_notes/` — 方法论/参考/团队纪要
