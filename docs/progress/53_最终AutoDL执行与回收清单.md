# 最终AutoDL执行与回收清单

> 准备日期：2026-10-17  
> 目标：只执行已经冻结的最终基线训练和一次性2025评价，不再增加实验路线。

## 1. 当前冻结边界

- 主模型协议：`data/results/allmarket_final_2025_preregistration_v1/protocol.json`；
- 最终基线协议：`data/results/allmarket_final_2025_baseline_preregistration_v1/protocol.json`；
- 盈利分析协议：`data/results/allmarket_final_2025_profit_analysis_preregistration_v1/protocol.json`；
- 盈利分析协议SHA-256：`72a5cff9e460d8073b20aed0cc6b271025a134ac0bcaf131ae7c2642ded09089`；
- 2025打开标记必须不存在；
- 9个四市场Transformer checkpoint已经在本地完成并通过哈希回收；
- 本地允许提前计算192项逐节点XGBoost，但不得读取2025。

上云前必须再次运行：

```powershell
.venv\Scripts\python.exe scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
.venv\Scripts\python.exe scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
.venv\Scripts\python.exe scripts/decision_aware/freeze_allmarket_2025_profit_analysis_protocol.py --verify
Test-Path data/results/allmarket_final_2025_preregistration_v1/LOCKED_2025_OPENED.json
```

最后一行必须为`False`。

## 2. 本地CPU预计算

入口：

```powershell
pwsh -File scripts/decision_aware/run_local_allmarket_final_xgboost_v1.ps1 -Workers 6
```

它按四市场×三任务运行，每个节点内部单线程、节点间6并发；已有合法缓存会复用。完成标准：

逐节点XGBoost三个任务统一使用`reg:absoluteerror`。原因是原生Pseudo-Huber在CAISO/NYISO训练期
诊断中出现数万USD/MWh病态外推；该修复发生在2025打开前。pooled XGBoost仍保留已经过四折验证的
任务专属Pseudo-Huber目标，两类基线不得混淆。

- `data/checkpoints/global_baselines_v1/**/*_final/xgboost_absolute_*_seed0.joblib`共192份；
- `data/results/global_baselines_v1/**/*_final/xgboost_absolute_*_seed0.npz`共192份；
- `data/results/global_baselines_v1/**/*_final/summary.json`共12份；
- `artifact_audit.json`不再缺少任何`global_baselines_v1`路径。

模型文件和预测缓存必须一起上传，否则云端阶段A会重新拟合。

### 2026-10-17本地执行记录

- 主协议核验通过；最终基线协议修订后SHA-256为
  `cdb20963498dd79150feff54da8e07364470ce22bc2f08f4fb78d12b72155686`；
- CAISO三任务已完成：DA MAE 4.78、RT-at-DA MAE 7.17、滚动RT MAE 6.28；
- ERCOT三任务已完成：DA MAE 8.98、RT-at-DA MAE 10.73、滚动RT MAE 9.65；
- NYISO与PJM三任务均已完成；逐节点XGBoost共192项，审计无缺项；
- Python、PowerShell与两个云端Bash入口语法检查通过，项目测试349/349通过；
- `LOCKED_2025_OPENED.json`仍不存在，未读取2025。
- 完整上传包：`data/results/allmarket_final_cloud_inputs_20261017.tar.gz`，989,184,074字节，
  447个归档条目（446个输入文件与1份内置manifest）；SHA-256为
  `2ec157d3c8318c1cc81807173f40df24c1b998c38f0a8650d9c373b72cb62654`。

## 3. 制作上传包

本地CPU任务完成后运行：

```powershell
.venv\Scripts\python.exe scripts/decision_aware/package_allmarket_final_cloud_inputs.py
```

上传包包含：两份协议内全部冻结代码、协议JSON及哈希、9个Transformer checkpoint、192个逐节点
XGBoost checkpoint、192份对应预测缓存和12份summary。脚本会逐文件重新核对协议哈希，并生成归档SHA-256。

## 4. 云端阶段A：最终基线训练

> 2026-10-17 22:13（北京时间）已在RTX 4090实例启动阶段A，后台PID 2075；首项为
> TimesFM-2.5 DA。远端输入归档SHA-256与本地一致，主协议和基线协议均核验通过，
> 启动时`LOCKED_2025_OPENED.json`不存在。

> 22:46发现AutoDL内XGBoost wheel不支持CUDA；三个TimesFM-2.5任务已经完成，pooled XGBoost
> 在设备一致性检查处停止。该问题发生于2025打开前，随后将pooled XGBoost计算后端冻结为CPU，
> 不改变数据、特征、目标函数、树参数或评价协议；修订后基线协议哈希见上文。

> 2026-10-18 01:02阶段A完成：远端201/201项审计通过。结果归档
> `data/results/allmarket_final_baselines_completed_20261017.tar.gz`已下载，SHA-256为
> `94e4ea8aac259c3204a084dfe6b01d7ab7f3b33d1cec9758882488b599ab3935`；内部manifest
> 44/44个文件哈希一致。Windows XGBoost不能直接反序列化Linux生成的pooled joblib，但文件完整，
> 阶段B继续在同一Linux环境读取；该兼容性问题不改变训练结果。

启动AutoDL后，先上传并在远端仓库根目录解压输入包。连接地址和端口以当次AutoDL页面为准，不沿用旧实例。
随后依次运行：

```bash
cd /root/autodl-tmp/electricity-price-forecasting
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
bash -n scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh
PREFLIGHT_ONLY=1 bash scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh
nohup bash scripts/decision_aware/run_cloud_allmarket_final_baselines_v1.sh \
  > data/results/cloud_allmarket_final_baselines_v1/nohup.log 2>&1 &
```

阶段A成功标准：

```bash
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/audit_allmarket_final_baselines.py --require-complete
tail -n 30 data/results/cloud_allmarket_final_baselines_v1/status.tsv
sha256sum -c data/results/allmarket_final_baselines_completed_20261017.tar.gz.sha256
```

必须得到201项完整产物；本阶段仍不得出现2025打开标记。

## 5. 云端阶段B：一次性2025统一评价

阶段A结果先下载到本地并完成包级与内部manifest核验。只有核验通过，才显式授权运行：

```bash
cd /root/autodl-tmp/electricity-price-forecasting
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
/root/autodl-tmp/epf-venv/bin/python scripts/decision_aware/freeze_allmarket_2025_profit_analysis_protocol.py --verify
PREFLIGHT_ONLY=1 bash scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh
nohup bash scripts/decision_aware/run_cloud_allmarket_locked_2025_v2.sh \
  > data/results/cloud_allmarket_locked_2025_v2/nohup.log 2>&1 &
```

若任务因机器中断恢复，只允许保持同一协议哈希并设置`RESUME_LOCKED_RUN=1`。不得更改结构、损失、seed、
阈值或基线后重新打开2025。

阶段B成功标准：

- `final_artifact_audit.json`中`complete=true`；
- 四市场×三任务预测与尾部指标齐全；
- ERCOT 11套完整系统结算齐全；
- Transformer seed 0/1/2及固定集成均完成；
- Oracle、最大回撤、去最高5/10日、7日块Bootstrap和计算成本齐全；
- 最终归档包及逐文件manifest均通过SHA-256。

结果下载并通过上述完整性检查后，必须先运行：

```powershell
.venv\Scripts\python.exe scripts/decision_aware/audit_allmarket_locked_2025_profit_claim.py
```

该审计把固定三seed Transformer集成与pooled all-market XGBoost的ERCOT节点等权日均净收益作为
主比较，重算逐日配对差与7日块Bootstrap区间，并按预注册规则输出“稳定胜出／仅点估计改善／未胜出”。
MAE只能解释价格预测，不能覆盖盈利分类。

## 6. 回收后的唯一工作

结果下载并核验后，重新运行技术报告生成器。它会检测
`data/results/allmarket_locked_2025_summary_v2.json`，自动把“待执行”章节替换为最终2025比较表、
多seed稳定性和实验闭环结论。

之后只允许整理论文、图表、参考文献和文字表述。无论2025胜出、接近或落后，都不再新增结构、seed、
损失、尾部权重或模型；结果较弱时降低论文主张强度，而不是使用2025继续调参。

## 7. 最终执行与回收结果（2026-10-19）

- 阶段B已经完成并回收到本地；最终归档为
  `data/results/allmarket_locked_2025_complete_v2_20261017.tar.gz`，SHA-256为
  `2863eb2c7d937aaf65648c48f5c8376de57ce53c2ab89a8cfbeefea660c02d5a`；
- 归档内327个文件逐项哈希一致；`final_artifact_audit.json`确认129个预测归档、11套ERCOT
  完整结算系统以及4市场×3任务均齐全；
- 中断恢复仅统一容差内的重复真实标签，12项最大差异为`0.000244140625 USD/MWh`，低于
  `0.0003`容差，所有预测数组未变；
- Transformer固定集成在2025四市场×三任务12项MAE中逐项最低，但ERCOT净收益为`28.04`，
  pooled all-market XGBoost为`44.43 USD/节点·日`；配对差`-16.39`，7日块95%区间
  `[-23.19, -10.05]`；
- 盈利主张审计分类为`no_superiority`。最终允许声称预测优势，不允许声称经济收益优势；
- 至此阶段A、阶段B与回收审计全部闭环。详细最终结论见
  `54_2025一次性锁定测试最终结论.md`，不再启动新增实验。
