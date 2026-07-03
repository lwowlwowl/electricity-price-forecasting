# ElecFM 历史文档归档

> 按时间顺序排列的设计演进和实验记录  
> 当前活跃文档在 `docs/fusion/`

---

## 📜 版本演进时间线

```
2026-06-25 ────┐
               ├── 参数消融 v1.0 完成
               │   └── docs/specs/experiment_manual.md
               │
2026-06-28 ────┤
               ├── 结构消融 v2.0 完成
               │   └── docs/specs/experiment_manual_v2.md
               │
2026-06-30 ────┤
               ├── ElecFM v1 设计
               │   └── fusion_model_design_v3.md (初版)
               │   └── 决策：开始实现 ElecFM
               │
2026-07-01 ────┤
               ├── ElecFM v1 训练（3节点，quant_head可训）
               │   └── ❌ 失败：Coverage 0.775→0.582
               │
               ├── ElecFM v2 训练（3节点，quant_head冻结）
               │   └── ⚠️ 部分成功：W1 SMAPE 29.19
               │
               └── ElecFM v3 训练（15节点，非LoRA）
                   └── ❌ 严重过拟合
                       └── elecfm_ercot_full_v1_results.md
               │
2026-07-02 ────┤
               ├── LoRA 方案设计
               │   └── elecfm_lora_optimization.md
               │
               ├── LoRA 实现完成
               │   └── IMPLEMENTATION_SUMMARY.md
               │
               └── 文档整理
                   └── 合并为 docs/fusion/*.md

当前：docs/fusion/ (最新)
```

---

## 📁 归档文件说明

### 设计文档

| 文件 | 日期 | 状态 | 说明 |
|------|------|------|------|
| `fusion_model_design_v3.md` | 2026-06-30 | 🔄 已合并 | 原始设计文档，内容已整合到 `docs/fusion/design.md` |
| `elecfm_lora_optimization.md` | 2026-07-02 | 🔄 已合并 | LoRA 方案设计，内容已整合到 `docs/fusion/design.md` |
| `IMPLEMENTATION_SUMMARY.md` | 2026-07-02 | 🔄 已合并 | 实现总结，内容已整合到 `docs/fusion/implementation.md` |

### 实验记录

| 文件 | 日期 | 实验 | 结果 | 关键结论 |
|------|------|------|------|----------|
| `elecfm_ercot_full_v1_results.md` | 2026-07-01 | ElecFM v3（全ERCOT，非LoRA）| ❌ 过拟合 | 79M参数vs125K样本失衡，引出LoRA方案 |

---

## 🔍 何时查看这些文档

| 场景 | 查看 |
|------|------|
| 想了解设计决策的演变过程 | `fusion_model_design_v3.md` |
| 想了解LoRA方案的原始论证 | `elecfm_lora_optimization.md` |
| 想了解第一轮失败的细节 | `elecfm_ercot_full_v1_results.md` |
| 想了解代码实现的具体改动 | `IMPLEMENTATION_SUMMARY.md` |

---

## ✅ 当前推荐

**直接使用**：`docs/fusion/` 下的文档

| 文档 | 用途 |
|------|------|
| `README.md` | 入口导航 |
| `design.md` | 最新设计（合并了v3 + LoRA）|
| `experiments.md` | 完整实验时间线（包含失败记录）|
| `implementation.md` | 实现细节和使用指南 |

---

**维护者**：Claude Code  
**归档时间**：2026-07-02
