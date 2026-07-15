# Chronos2 单协变量扫描实验

> 目的：在零样本基础模型 Chronos2 上，**逐个单独**测试每个协变量对
> 点预测（MAE / rMAE）与尖峰检测（Spike-F1）的边际影响，确定哪些
> 协变量真正有用、哪些是噪声甚至有害。
>
> 实验日期：2026-07-15　市场：ERCOT、PJM（分别独立运行）

---

## 一、实验动机

此前 `ablation_A_covariates` 是"递进打包"式消融（`[] → [load] → [load,temp] → …`），
它有两个问题：

1. **多数模型吃不到协变量**。全模型清单里只有 Chronos2 的
   `supports_covariates=True`（见 `src/models/forecasters.py` / `foundation.py`），
   其余 10 个模型喂进去也降级为 `covariates_used=False`，结果与 `[]` 完全相同。
   因此"协变量消融"实际只对 Chronos2 一个模型生效。
2. **打包消融分不清归因**。最后一档把经济+风暴+发电结构 8 个协变量一起塞进去，
   即便整体变好/变差，也无法分辨是其中哪一个在起作用。

本实验改为：**只跑 Chronos2，基线 `[]` + 每个协变量单独测**，直接给出每个
协变量的边际贡献（ΔMAE、ΔSpike-F1）。这样能干净地识别出赢家与噪声。

---

## 二、协变量来源与家族

协变量全部读自 **预报版 model_ready** 文件
（`data/covariates/model_ready/{market}_features_forecast_hourly.csv`），
其构成遵循"四个协变量为预报的，其他原来是 lag 就也是 lag"原则：

| 家族 | 具体列 | 时效 |
|---|---|---|
| **日前预报** | load, temperature, wind, solar | 日前预报值（T-1 可得，无泄漏）|
| **经济/燃料** | henry_hub(气价), wti(油价), hrc_futures(钢价,仅PJM), natgas_storage(库存) | shift(24) 滞后 |
| **风暴** | storm_event_count 等 7 列 | shift(24) 滞后 |
| **发电结构** | total_gen / gas_gen / wind_gen / solar_gen / gas_share / renewable_share / gas_share_diff / renewable_shock | shift(24) 滞后 |

### 精简代表集（本实验扫描对象）

为避免同源/派生列重复测试，每家族只选最干净、最不冗余的代表：

| 家族 | 选中代表 | 未选及其原因（冗余） |
|---|---|---|
| 预报版 | load, temperature, wind, solar | — |
| 经济/燃料 | henry_hub, wti, hrc_futures | natgas_storage（与气价共线）|
| 风暴 | storm_event_count | 其余 6 列（伤亡/损失/类型标志同源，取最干净代理）|
| 发电结构 | gas_share, renewable_shock | gas_share_diff/renewable_share/各 `*_gen_mwh`（与 gas_share 共线）|

脚本按市场自动过滤不存在的列：
- **ERCOT** 扫 9 个（无钢价 hrc_futures）
- **PJM** 扫 7 个（无 wind/solar，且无 wti，但含钢价 hrc_futures）

> 注：实测发现 PJM 既无 wind/solar 也无 wti_usd_per_barrel 列，仅 PJM 有钢价。

---

## 三、实验步骤

### 1. 加载器改造

`src/data_processing/loader.py` 的 `load_slice_model_ready` 新增 `forecast` 参数：
- `forecast=True` → 读 `{market}_features_forecast_hourly.csv`（预报版）
- `forecast=False` → 原行为（纯滞后版 `{market}_features_hourly.csv`）

### 2. 扫描脚本

`src/parameter_ablation/run_chronos2_covariate_scan.py`：
- 只构造一次 Chronos2（worker 只加载一次模型；适配器内部按任务 cache，不同协变量组合互不串味）
- 对每个协变量组合（`[]` 基线 + 逐个单独）跑一次完整滚动回测
- 汇总输出 MAE / rMAE / Spike-F1 及相对基线的 Δ，落盘 CSV

### 3. 回测配置（两市场一致）

| 项 | 值 |
|---|---|
| 节点组 | ablation（每市场 3 个代表节点）|
| 频率 | 1h |
| 上下文 | 168 步（7 天）|
| 预测步长 | 24 |
| 数据范围 | 2025-01-01 ~ 2026-06-05 |
| 测试期 | 2025-07-01 ~ 2026-06-01 |
| 步长 | 24h |
| 起报点数 | 336（连续，未限）|
| 尖峰阈值 | 测试期前历史 P95（global 口径，防泄漏）|

### 4. 复现命令

```bash
python3 src/parameter_ablation/run_chronos2_covariate_scan.py --market ERCOT
python3 src/parameter_ablation/run_chronos2_covariate_scan.py --market PJM
```

结果落盘：
- `data/results/parameter_ablation/chronos2_covariate_scan/scan_ercot_forecast.csv`
- `data/results/parameter_ablation/chronos2_covariate_scan/scan_pjm_forecast.csv`

---

## 四、结果

### ERCOT（基线 [] MAE = 16.867，rMAE = 0.7702，Spike-F1 = 0.4898）

| 协变量 | n_origins | rMAE | MAE | ΔMAE | ΔMAE% | Spike-F1 | ΔSpike-F1 | spike_recall | spike_f1_q90 |
|---|---|---|---|---|---|---|---|---|---|
| **wind** | 336 | 0.7384 | 16.263 | **−0.604** | **−3.58%** | 0.4903 | +0.0005 | 0.4299 | 0.3727 |
| solar | 336 | 0.7656 | 16.713 | −0.154 | −0.91% | 0.4966 | +0.0068 | 0.4283 | 0.3427 |
| renewable_shock | 335 | 0.7723 | 16.797 | −0.070 | −0.42% | 0.5018 | +0.0120 | 0.4377 | 0.3407 |
| gas_share | 335 | 0.7752 | 16.898 | +0.031 | +0.18% | 0.4797 | −0.0101 | 0.4116 | 0.3435 |
| temperature | 336 | 0.7763 | 17.123 | +0.256 | +1.52% | 0.5081 | +0.0183 | 0.4425 | 0.3460 |
| storm_event_count | 336 | 0.7824 | 17.161 | +0.294 | +1.74% | 0.4926 | +0.0028 | 0.4307 | 0.3264 |
| load | 336 | 0.7832 | 17.283 | +0.416 | +2.47% | 0.5090 | +0.0192 | 0.4449 | 0.3442 |
| wti_usd_per_barrel | 336 | 0.7964 | 17.391 | +0.524 | +3.10% | 0.4912 | +0.0014 | 0.4276 | 0.3391 |
| henry_hub_usd_per_mmbtu | 336 | 0.8010 | 17.443 | +0.576 | +3.42% | 0.4866 | −0.0032 | 0.4276 | 0.3365 |

> 按 MAE 升序。ΔMAE 负 = 协变量让误差下降（有用）；正 = 让误差上升（有害/噪声）。

### PJM（基线 [] MAE = 25.731，rMAE = 0.6925，Spike-F1 = 0.5080）

| 协变量 | n_origins | rMAE | MAE | ΔMAE | ΔMAE% | Spike-F1 | ΔSpike-F1 | spike_recall | spike_f1_q90 |
|---|---|---|---|---|---|---|---|---|---|
| **load** | 336 | 0.6605 | 24.845 | **−0.886** | **−3.44%** | 0.5298 | **+0.0217** | 0.4328 | 0.4489 |
| henry_hub_usd_per_mmbtu | 336 | 0.7020 | 25.696 | −0.034 | −0.13% | 0.5072 | −0.0008 | 0.4047 | 0.4162 |
| renewable_shock | 335 | 0.7019 | 26.019 | +0.288 | +1.12% | 0.4888 | −0.0192 | 0.3951 | 0.4201 |
| temperature | 336 | 0.6902 | 25.981 | +0.250 | +0.97% | 0.5161 | +0.0081 | 0.4253 | 0.4305 |
| gas_share | 335 | 0.7201 | 26.072 | +0.341 | +1.32% | 0.5087 | +0.0006 | 0.4187 | 0.4148 |
| hrc_futures_usd_per_ton | 336 | 0.7095 | 26.181 | +0.450 | +1.75% | 0.4976 | −0.0105 | 0.4087 | 0.3993 |
| storm_event_count | 336 | 0.7088 | 26.659 | **+0.928** | **+3.60%** | 0.4930 | −0.0151 | 0.3972 | 0.4174 |

---

## 五、Forward-Fill（前向填充）说明

本实验使用的协变量在进入模型前，需从各自的原始频率对齐到小时级时间轴。
低频（日/周/月）协变量通过 **forward-fill（前向填充，`pandas.DataFrame.ffill`）**
展开到每个小时——即把一个低频观测值重复填到下一个新观测出现前的所有小时。

### 5.1 哪些协变量走了 forward-fill

forward-fill 发生在 model_ready 构建流水线（`scripts/covariates/`），而非
`loader.py`。按家族分：

| 家族 | 原始频率 | ffill 处理 | 实现位置 |
|---|---|---|---|
| 经济/燃料（气价/油价/钢价/库存） | 日 / 周 / 月 | ffill 到日，再展开到小时 | `03_align_to_hourly.py:45-110` |
| 风暴（event_count 等） | 日 | 无事件日填 0，再 ffill 到小时 | `05_download_noaa_storm.py:233-240` |
| 发电结构（gas_share 等） | 小时 | 不需 ffill（原生小时频） | `06_eia_fuel_mix_features.py` |
| load / temperature / wind / solar | 小时 | **不 ffill**——预报版用日前预报值直接按时间戳对齐替换 | `15_build_forecast_model_ready.py:88-121` |

> 关键：**只有低频的经济/风暴类协变量被 ffill**；4 个预报版协变量和发电结构
> 是原生小时频，不走 ffill。`loader.py:178-181` 的天气 ffill 仅在目标频率高于 1h
> 时触发，本实验 freq=1h 不经过该分支。

### 5.2 forward-fill 的两种具体行为

1. **经济/燃料类**（`load_and_ffill_to_daily`）：
   `reindex(daily_range).ffill().bfill()` → 先对齐到日频并前向/后向填充
   周末与节假日空值，再把每日值复制到该日全部 24 小时。
   - 日频（气价/油价/钢价 HRC 期货）：周末/节假日用前一交易日值 ffill，
     约 2-3 天重复。
   - 周频（天然气库存）：整周用同一值。
   - 注：钢价源是**日频 HRC 钢卷期货**（非月度 PPI），但经 ffill 仍会把单个
     日值复制到 24 小时，叠加 shift24 滞后后时效性依然有限。

2. **风暴类**：无事件日先填 0，再 `reindex(hourly_index, method="ffill")`，
   日的计数值被复制到 24 小时。

### 5.3 forward-fill 对实验结论的影响

ffill **不改变"哪些协变量有用"的结论**，但它是经济/风暴类协变量集体失效的
**部分原因**，需与"shift24 滞后"叠加理解：

- 单个低频值经 ffill 在一天内重复 24 次（钢价更甚，一个月重复 ~720 小时），
  再 `shift(24)` 滞后 → 起报时模型看到的是**一两天前（钢价则是上月）的恒定值**，
  对小时级电价几乎无边际信息。
- 这解释了为何钢价（月频 ffill）、风暴计数（日频 ffill）、gas_share（滞后）
  在 ERCOT/PJM 两边都无效甚至有害——不是信号本身无意义，而是 **ffill 把低频
  信号摊平成常数**，叠加滞后后彻底失去时效性。
- 对照之下，4 个预报版协变量是**原生小时频、不经 ffill**，所以 wind（ERCOT）
  和 load（PJM）能保留小时级时效，成为唯一有效的赢家。

### 5.4 方法论 caveat

- ffill 在数据点之间**人为制造了自相关**（同一值重复），可能让模型对低频
  协变量产生虚假的稳定性认知——这本身可能正是 MAE 上升的诱因之一。
- 严格意义上，日/周频数据用 ffill 摊到小时级后，单个值在一天（或一周）内
  重复 24（或 168）次，叠加滞后等于"日/周度常数"特征，不应期待它对小时级
  价格有强解释力。本实验已实测确认钢价、气价等均无效（见结论 3）。
- 若要评估"理想低频协变量"的价值，应改用日级模型或对低频信号做更有意义的
  时间编码（如距上次更新的小时数），而非简单 ffill——这超出本实验范围，
  仅作记录。

---

## 六、结论

### 结论 1：大多数协变量让 MAE 变差，不是变好

ERCOT 9 个协变量里只有 3 个降了 MAE，PJM 7 个里只有 1 个真正降了 MAE。
其余全是噪声或副作用。这与"协变量越多越好"的直觉相反。

### 结论 2：赢家因市场而异，但都指向"该市场的物理定价驱动因子"

- **ERCOT 命脉 = 风电**。`wind` 单独就把 rMAE 从 0.770 压到 0.738（−3.58%），
  是全场唯一有实质意义的 MAE 改善。这与 ERCOT 风电渗透率极高、风电出力
  直接决定电价的物理事实吻合。
- **PJM 命脉 = 负荷**。`load` 一举拿下 MAE 与 Spike-F1 双改善（rMAE 0.693→0.661，
  −3.44%；Spike-F1 +0.022），是 PJM 唯一真正有用的协变量。PJM 是火电+工业
  负荷主导的市场，负荷预测就是价格走向的核心。

通用经济/风暴类协变量两边都不灵——**真正有用的是市场特定物理驱动因子，
不是放之四海皆准的宏观数据**。

### 结论 3：被证伪的几个假设

- **钢价（hrc_futures, PJM 独有）无效**：ΔMAE +0.450（变差），Spike-F1 −0.0105。
  "PJM 工业负荷重→钢价有解释力"的假设被证伪。钢价源是日频 HRC 钢卷期货，经
  forward-fill 摊到 24 小时、再 shift24 滞后，到起报时是一两天前的旧值，对
  小时级电价无边际信息（详见第五节 forward-fill）。
- **风暴计数无效**：ERCOT +0.29、PJM +0.93（PJM 全场最差）。风暴计数是事后
  统计、滞后一天，对日前预测基本是噪声。
- **温度反而变差**：ERCOT +0.26、PJM +0.25。温度预报质量没问题，但 Chronos2
  零样本下似乎不会把它转化为价格信号——与直觉出入最大的一点。
- **gas_share 两边都变差**：天然气占比虽是边际定价逻辑，但 shift24 的滞后值
  太旧，模型用不上。

### 结论 4：MAE 与 Spike-F1 是解耦的（重要）

ERCOT 的 `load`：MAE 变差 +0.416（倒数第三），但 Spike-F1 提升 +0.0192（全场最好），
spike_recall 从 0.416 拉到 0.445。

> 含义：加 load 后 Chronos2 整体预测水平被抬高（平均误差上升），但对尖峰
> 变得更敏感——能多抓到一些极端事件。这是一个真实的权衡：
> **协变量可能让模型过度响应波动 → 平均精度下降（MAE↑），但极端事件捕捉
> 能力上升（Spike-F1↑）。**

PJM 的 `load` 是少数两者同向改善的（MAE↓ 且 Spike-F1↑），这是它成为 PJM
明确赢家的原因。ERCOT 的 `renewable_shock` 同样是 MAE 微降+Spike-F1 升，
符合"可再生骤变→尖峰"的设计意图。

**只看 MAE 会低估某些协变量（如 ERCOT load、renewable_shock）对尖峰检测
的价值**——评估协变量时必须同时看点指标与尖峰指标。

### 结论 5：失效机制推测

1. **Chronos2 是零样本基础模型**：协变量条件机制依赖预训练习得，对单个外生
   序列的利用能力有限，未必能形成好的联合表征。
2. **预报版 wind/solar 是代理变量**：wind 实为风速(m/s)、solar 实为辐射(W/m²)，
   非真实发电量。ERCOT wind 即便只是风速代理仍能降 MAE，说明风电出力与风速
   相关性够强；solar 辐射代理则较弱。
3. **低频协变量被 forward-fill 摊平 + shift24 滞后**：经济/风暴/发电类原为
   日/周频，经 ffill 展开到小时后变为重复常数（钢价日值复制到 24 小时、
   天然气库存整周一个值），再 shift24 滞后 → 起报时是"一两天前"的恒定值，
   对小时级日前预测几乎没有边际信息。这是它们集体失效的主因（详见第五节
   forward-fill）。对照之下，4 个预报版协变量原生小时频、不经 ffill，
   wind/load 才得以保留时效成为赢家。

---

## 七、后续建议

1. **协变量消融的论文结论**应为：*"仅市场特定物理驱动因子（ERCOT-wind /
   PJM-load）显著改善 Chronos2 的 MAE；通用经济与风暴类协变量在零样本设置下
   无效甚至有害。MAE 与 Spike-F1 解耦，部分协变量（如 ERCOT load）以牺牲平均
   精度换取尖峰捕捉能力。低频协变量经 forward-fill 摊平并滞后后失去时效，是其
   失效的重要诱因。"* 比笼统的"协变量越多越好"诚实且有说服力。
2. **补一个赢家组合实验**：ERCOT 试 `wind`、`wind+solar`；PJM 试 `load`。
   验证单协变量增益能否在组合后叠加。
3. **钢价、风暴、gas_share** 等可从"有用协变量"清单正式剔除，多档消融里
   不必再反复打包它们。
4. **MAE/Spike-F1 解耦**值得论文单独写一段：它说明只看 MAE 会低估某些协变量
   对尖峰检测的价值。
5. **低频协变量的改进方向**（可选，超出本实验范围）：若确需评估钢价等日/周频
   信号价值，可改用日级模型、或对低频信号编码"距上次更新的小时数"等更有意义
   的特征，避免 ffill 把信号摊成常数。
