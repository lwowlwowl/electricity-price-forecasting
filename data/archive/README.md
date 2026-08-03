# data/archive — 旧范式产物归档

> 2026-07-29 归档。这些都是**已归档的旧 ElecFM 融合范式**（冻结零样本 ElecFM 融合 + 消融）
> 的 checkpoint / 结果，新方向（Decision-aware 从零训练 DA-TSFM）不再使用。
> 旧代码本身见 `src/archive/`（说明 `src/archive/README.md`）。

## 内容

```
data/archive/
├── checkpoints/        # 旧 ElecFM 各版本训练权重
│   ├── electfm/  electfm_ercot_full/  electfm_ercot_full_lora/
│   └── electfm_ercot_full_v{5,5b,6,6_allgroups,6_d32,6_d128,7}/ ...
└── results/            # 旧 ElecFM 融合 + 消融实验输出
    ├── fusion/  fusion_all_versions_comparison.png
    └── fusion_electfm_ercot_full_*/  fusion_electfm_lora_*/
```

主线（新方向）产出现在写到 `data/checkpoints/da_tsfm_pilot/`、`data/results/`（留空待新结果）。

> 注：`data/results/` 下还残留 `parameter_ablation/`、`structural_ablation/`（旧 v1/v2 参数与结构消融结果），
> 同属旧范式，可按需继续移入此处。
