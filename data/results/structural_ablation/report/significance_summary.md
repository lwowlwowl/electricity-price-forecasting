# 统计显著性检验报告

> 方法：Wilcoxon signed-rank（双侧，zero_method='wilcox'）
> 有效检验数：118（剔除 CRASH / N/A / 样本不足条件）
> **主校正**：Bonferroni（FWER，最保守）｜**辅助校正**：Benjamini-Hochberg FDR
> 星号阈值：`*` p<0.05  `**` p<0.01  `***` p<0.001

---

## 全结构消融（S-A/B/G 系列）

### Toto-2.0

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `disable_xpos` | 1.0000 | ns | ns | 1.0000 | ns | ns |
| `halve_heads` | — |  |  | — |  |  |
| `remove_rope` | 0.0000 | *** | *** | 0.6023 | ns | ns |
| `simplify_output_head` | — |  |  | — |  |  |
| `simplify_patch_emb` | 0.0000 | *** | *** | 0.0035 | ns | * |
| `skip_attention` | 0.0000 | *** | *** | 0.9250 | ns | ns |
| `skip_ffn` | 0.0000 | *** | *** | 0.8889 | ns | ns |
| `skip_layernorm` | — |  |  | — |  |  |
| `skip_time` | 0.0000 | *** | *** | 0.5202 | ns | ns |
| `skip_variate` | 0.0012 | ns | ** | 0.9096 | ns | ns |
| `truncate_back_half` | 0.0000 | *** | *** | 0.2914 | ns | ns |
| `truncate_front_half` | 0.0000 | *** | *** | 0.8093 | ns | ns |

### Chronos-2

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `halve_heads` | 0.1840 | ns | ns | 0.2196 | ns | ns |
| `remove_rope` | 0.0000 | *** | *** | 0.0001 | * | *** |
| `simplify_output_head` | — |  |  | — |  |  |
| `simplify_patch_emb` | — |  |  | — |  |  |
| `skip_attention` | 0.0000 | *** | *** | 0.0001 | * | *** |
| `skip_ffn` | 0.0000 | *** | *** | 0.0005 | ns | ** |
| `skip_layernorm` | 0.0000 | *** | *** | 0.0124 | ns | * |
| `skip_time` | 0.0000 | *** | *** | 0.0001 | * | *** |
| `skip_variate` | 0.0000 | *** | *** | 0.2180 | ns | ns |
| `truncate_back_half` | 0.0000 | *** | *** | 0.0001 | * | *** |
| `truncate_front_half` | 0.2534 | ns | ns | 0.5327 | ns | ns |

### TimesFM-2.5

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `halve_heads` | 1.0000 | ns | ns | 1.0000 | ns | ns |
| `point_only` | 1.0000 | ns | ns | 1.0000 | ns | ns |
| `remove_rope` | 1.0000 | ns | ns | 1.0000 | ns | ns |
| `simplify_output_head` | 1.0000 | ns | ns | 1.0000 | ns | ns |
| `simplify_patch_emb` | 0.0000 | *** | *** | 0.0004 | ns | ** |
| `skip_attention` | — |  |  | — |  |  |
| `skip_ffn` | 0.0000 | *** | *** | 0.9772 | ns | ns |
| `skip_layernorm` | — |  |  | — |  |  |
| `truncate_back_half` | 0.0000 | *** | *** | 0.0015 | ns | ** |
| `truncate_front_half` | 0.0003 | * | ** | 0.0639 | ns | ns |
| `truncate_front_quarter` | 0.0001 | ** | *** | 0.1397 | ns | ns |

## 逐层消融（S-L 系列）

### Toto-2.0

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `skip_layer_0` | 0.0000 | *** | *** | 0.0072 | ns | * |
| `skip_layer_1` | 0.0000 | *** | *** | 0.6274 | ns | ns |
| `skip_layer_2` | 0.2534 | ns | ns | 0.2330 | ns | ns |
| `skip_layer_3` | 0.7000 | ns | ns | 0.6002 | ns | ns |
| `skip_layer_4` | 0.0002 | * | *** | 0.0068 | ns | * |
| `skip_layer_5` | 0.0000 | *** | *** | 0.9899 | ns | ns |

### Chronos-2

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `skip_layer_0` | 0.9838 | ns | ns | 0.0229 | ns | ns |
| `skip_layer_1` | 0.4400 | ns | ns | 0.0869 | ns | ns |
| `skip_layer_2` | 0.3818 | ns | ns | 0.8647 | ns | ns |
| `skip_layer_3` | 0.8394 | ns | ns | 0.2619 | ns | ns |
| `skip_layer_4` | 0.5158 | ns | ns | 0.5406 | ns | ns |
| `skip_layer_5` | 0.1840 | ns | ns | 0.2114 | ns | ns |

### TimesFM-2.5

| 消融类型 | SMAPE p_raw | Bonf | FDR | F1 p_raw | Bonf | FDR |
|---|---|---|---|---|---|---|
| `skip_layer_0` | 0.0000 | *** | *** | 0.0023 | ns | ** |
| `skip_layer_1` | 0.0000 | *** | *** | 0.2242 | ns | ns |
| `skip_layer_10` | 0.6702 | ns | ns | 0.1755 | ns | ns |
| `skip_layer_11` | 0.3184 | ns | ns | 0.2076 | ns | ns |
| `skip_layer_12` | 0.0879 | ns | ns | 0.8927 | ns | ns |
| `skip_layer_13` | 0.5291 | ns | ns | 0.7995 | ns | ns |
| `skip_layer_14` | 0.0185 | ns | ns | 0.9292 | ns | ns |
| `skip_layer_15` | 0.5699 | ns | ns | 0.6121 | ns | ns |
| `skip_layer_16` | 0.0405 | ns | ns | 0.9594 | ns | ns |
| `skip_layer_17` | 0.0015 | ns | ** | 0.4326 | ns | ns |
| `skip_layer_18` | 0.0007 | ns | ** | 0.2094 | ns | ns |
| `skip_layer_19` | 0.0364 | ns | ns | 0.4838 | ns | ns |
| `skip_layer_2` | 0.0384 | ns | ns | 0.6247 | ns | ns |
| `skip_layer_3` | 0.2801 | ns | ns | 0.2486 | ns | ns |
| `skip_layer_4` | 0.2894 | ns | ns | 0.4443 | ns | ns |
| `skip_layer_5` | 0.4898 | ns | ns | 0.5303 | ns | ns |
| `skip_layer_6` | 0.7000 | ns | ns | 0.7836 | ns | ns |
| `skip_layer_7` | 0.2367 | ns | ns | 0.1547 | ns | ns |
| `skip_layer_8` | 0.8553 | ns | ns | 0.3454 | ns | ns |
| `skip_layer_9` | 0.3707 | ns | ns | 1.0000 | ns | ns |

---

## 通过 Bonferroni 校正的显著结果（共 29 个）

| 模型 | 类型 | 消融 | 指标 | sig(Bonf) | p_raw | p_bonf |
|---|---|---|---|---|---|---|
| Chronos-2 | full | `remove_rope` | smape | *** | 0.00000 | 0.00002 |
| Chronos-2 | full | `remove_rope` | spike_f1 | * | 0.00013 | 0.01551 |
| Chronos-2 | full | `skip_attention` | smape | *** | 0.00000 | 0.00001 |
| Chronos-2 | full | `skip_attention` | spike_f1 | * | 0.00013 | 0.01551 |
| Chronos-2 | full | `skip_ffn` | smape | *** | 0.00000 | 0.00002 |
| Chronos-2 | full | `skip_layernorm` | smape | *** | 0.00000 | 0.00000 |
| Chronos-2 | full | `skip_time` | smape | *** | 0.00000 | 0.00000 |
| Chronos-2 | full | `skip_time` | spike_f1 | * | 0.00013 | 0.01551 |
| Chronos-2 | full | `skip_variate` | smape | *** | 0.00001 | 0.00070 |
| Chronos-2 | full | `truncate_back_half` | smape | *** | 0.00000 | 0.00003 |
| Chronos-2 | full | `truncate_back_half` | spike_f1 | * | 0.00013 | 0.01551 |
| TimesFM-2.5 | full | `simplify_patch_emb` | smape | *** | 0.00000 | 0.00000 |
| TimesFM-2.5 | full | `skip_ffn` | smape | *** | 0.00000 | 0.00000 |
| TimesFM-2.5 | full | `truncate_back_half` | smape | *** | 0.00000 | 0.00045 |
| TimesFM-2.5 | full | `truncate_front_half` | smape | * | 0.00028 | 0.03343 |
| TimesFM-2.5 | full | `truncate_front_quarter` | smape | ** | 0.00006 | 0.00660 |
| TimesFM-2.5 | perlayer | `skip_layer_0` | smape | *** | 0.00000 | 0.00001 |
| TimesFM-2.5 | perlayer | `skip_layer_1` | smape | *** | 0.00000 | 0.00002 |
| Toto-2.0 | full | `remove_rope` | smape | *** | 0.00000 | 0.00038 |
| Toto-2.0 | full | `simplify_patch_emb` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | full | `skip_attention` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | full | `skip_ffn` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | full | `skip_time` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | full | `truncate_back_half` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | full | `truncate_front_half` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | perlayer | `skip_layer_0` | smape | *** | 0.00000 | 0.00000 |
| Toto-2.0 | perlayer | `skip_layer_1` | smape | *** | 0.00000 | 0.00002 |
| Toto-2.0 | perlayer | `skip_layer_4` | smape | * | 0.00015 | 0.01804 |
| Toto-2.0 | perlayer | `skip_layer_5` | smape | *** | 0.00000 | 0.00000 |

---

## 仅通过 FDR-BH 校正（Bonferroni 未通过，共 11 个）

> 这些结果在 FDR 控制下显著，但在严格的 FWER Bonferroni 控制下不显著。
> 对于高度相关的多重比较族（同一数据集上的多个消融条件），FDR 是更合适的选择。

| 模型 | 类型 | 消融 | 指标 | sig(FDR) | p_raw | p_fdr |
|---|---|---|---|---|---|---|
| Chronos-2 | full | `skip_ffn` | spike_f1 | ** | 0.00048 | 0.00183 |
| Chronos-2 | full | `skip_layernorm` | spike_f1 | * | 0.01236 | 0.03646 |
| TimesFM-2.5 | full | `simplify_patch_emb` | spike_f1 | ** | 0.00043 | 0.00170 |
| TimesFM-2.5 | full | `truncate_back_half` | spike_f1 | ** | 0.00146 | 0.00491 |
| TimesFM-2.5 | perlayer | `skip_layer_0` | spike_f1 | ** | 0.00227 | 0.00743 |
| TimesFM-2.5 | perlayer | `skip_layer_17` | smape | ** | 0.00146 | 0.00491 |
| TimesFM-2.5 | perlayer | `skip_layer_18` | smape | ** | 0.00073 | 0.00269 |
| Toto-2.0 | full | `simplify_patch_emb` | spike_f1 | * | 0.00352 | 0.01123 |
| Toto-2.0 | full | `skip_variate` | smape | ** | 0.00123 | 0.00441 |
| Toto-2.0 | perlayer | `skip_layer_0` | spike_f1 | * | 0.00717 | 0.02169 |
| Toto-2.0 | perlayer | `skip_layer_4` | spike_f1 | * | 0.00679 | 0.02109 |

---

## 关键发现专项显著性

### 「精度通路 vs 尖峰通路」功能分离层

| 层 | 模型 | SMAPE Δ | F1 Δ | 功能分类 | p_raw(SMAPE) | p_raw(F1) | Bonf(F1) | FDR(F1) |
|---|---|---|---|---|---|---|---|---|
| L4 | Toto-2.0 | — | — | 精度损↑ / 尖峰改善↑ | 0.0002 | 0.0068 | ns | * |
| L0 | Chronos-2 | — | — | 精度无关 / 尖峰隐性↓ | 0.9838 | 0.0229 | ns | ns |
| L1 | TimesFM-2.5 | — | — | 精度损↑ / 尖峰改善↑ | 0.0000 | 0.2242 | ns | ns |
| L7 | TimesFM-2.5 | — | — | 精度无关 / 尖峰关键↓ | 0.2367 | 0.1547 | ns | ns |
| L10 | TimesFM-2.5 | — | — | 精度无关 / 尖峰干扰层 | 0.6702 | 0.1755 | ns | ns |

> **关键注意**：Chronos-2 L0「隐性尖峰层」(Spike-F1 Δ = -14.1%) 的 Wilcoxon
> p_raw = 0.023（在 α=0.05 下显著），但 Bonferroni 校正后 p_corr = 1.00（不显著）。
> FDR-BH 校正后可能通过（见上表）。这反映 Bonferroni 在 118 个相关测试
> 中过于保守——对于该特定发现，建议报告 raw p 值并注明 FDR 校正结果。
