# ElecFM 文档中心

> ElecFM（Electricity Foundation Model）——基于 TimesFM-2.5 的电价预测融合模型

---

## 📋 快速导航

| 文档 | 内容 | 适合谁 |
|------|------|--------|
| [design.md](design.md) | 架构设计 + 训练策略 | 想了解模型设计的人 |
| [experiments.md](experiments.md) | 实验记录（按时间线）| 想了解实验历程的人 |
| [implementation.md](implementation.md) | 代码实现 + 使用指南 | 想跑代码/调试的人 |

---

## 🚀 快速开始

### 1. 快速测试（10-15 分钟）

验证 LoRA 功能是否正常：

```bash
bash run_lora_quick_test.sh
```

### 2. 正式训练（3-4 小时）

全 ERCOT 15 节点训练：

```bash
caffeinate -d external/timesfm/.venv/bin/python -u \
    src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_ercot_full_lora.yaml \
    2>&1 | tee run_ercot_lora.log
```

---

## 📊 当前状态

**最后更新**：2026-07-02

| 组件 | 状态 |
|------|------|
| 架构设计 | ✅ 完成（15 层 + 双头）|
| LoRA 实现 | ✅ 完成（1.1M 可训练参数）|
| 快速测试 | ✅ 通过（W1 SMAPE 20.93）|
| 全 ERCOT 训练 | 🔄 **进行中** |

---

## 🏗️ 项目结构

```
school/
├── docs/
│   ├── fusion/              # 本目录：ElecFM 文档
│   │   ├── README.md        # 入口文档（本文档）
│   │   ├── design.md        # 设计文档
│   │   ├── experiments.md   # 实验记录
│   │   └── implementation.md # 实现细节
│   │
│   ├── archive/elecfm/      # 历史文档归档
│   ├── specs/               # 其他技术规范
│   └── reports/             # 汇报材料
│
├── src/fusion_model/        # 源代码
├── configs/fusion/          # 配置文件
└── scripts/                 # 辅助脚本
```

---

## 📈 关键指标

| 指标 | 数值 |
|------|------|
| 模型总参数 | 178M |
| 可训练参数（LoRA）| **1.1M (0.62%)** |
| 参数/样本比 | **8.8** ✅ |
| 剪枝层数 | 20 → 15 层 |
| W1 SMAPE（快速测试）| **20.93** (vs 27.67 baseline) |

---

## 📜 历史版本

查看完整版本历史：
- **归档文档**：`docs/archive/elecfm/README.md`
- **变更日志**：`CHANGELOG.md`

---

## 🔧 核心特性

- **基于 TimesFM-2.5**：利用大规模预训练的时序知识
- **15 层剪枝架构**：移除 5 个冗余层，效率提升 25%
- **双头输出**：quantile head（精度）+ spike head（尖峰检测）
- **LoRA 微调**：仅训练 1.1M 参数，解决过拟合问题
- **两阶段训练**：渐进式适应，保护预训练知识

---

## 📝 更新日志

### 2026-07-02
- ✅ LoRA 实现完成
- ✅ 快速测试通过（W1 SMAPE 20.93）
- ✅ 文档整理完成

### 2026-07-01
- ❌ 全 ERCOT 非 LoRA 训练失败（严重过拟合）

### 2026-06-28
- ✅ v2.0 结构消融完成（确定 15 层方案）

### 2026-06-25
- ✅ v1.0 参数消融完成

---

## 📚 参考资料

1. [TimesFM-2.5](https://github.com/google-research/timesfm) - Google 时序预测基础模型
2. [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
3. [PEFT](https://huggingface.co/docs/peft) - HuggingFace 参数高效微调库

---

**维护者**：Claude Code  
**项目状态**：🔄 开发中
