#!/usr/bin/env bash
# ============================================================================
# 协变量全流程一键复现：下载→清洗→对齐→合并→相关→稳健性→model_ready
# 用法:  bash scripts/covariates/run_all.sh
#        （或 PY=python3.11 bash ... 指定解释器）
# 产物:  data/covariates/model_ready/{market}_features_hourly.csv  (喂模型)
#        data/covariates/analysis/*.csv + analysis/figures/covariates/*.png
# 需网络: 01(FRED+EIA) 02(FRED WTI) 05(NOAA)；02 的 HRC 需手动放 raw
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."          # 切到 repo 根 school/
PY="${PY:-python3}"

step() { echo; echo "==== [$1] $2 ===="; }

step 1/9 "01 下载 P0：Henry Hub 气价 + 天然气库存  [FRED+EIA，需网络]"
$PY scripts/covariates/01_download_p0.py

step 2/9 "02 下载 P1：WTI 原油(FRED) + 清洗 HRC 钢卷期货  [WTI 需网络；HRC 需手动放 steel/raw/hrc_futures_raw.csv]"
$PY scripts/covariates/02_download_p1.py

step 3/9 "03 对齐 P0+P1(含 HRC) 到四市场小时时间轴"
$PY scripts/covariates/03_align_to_hourly.py

step 4/9 "04 P0+P1 相关性分析 + 出图"
$PY scripts/covariates/04_correlation_analysis.py

step 5/9 "05 NOAA 风暴事件 → 市场级日频/小时频  [NOAA，需网络]"
$PY scripts/covariates/05_download_noaa_storm.py

step 6/9 "06 EIA 发电燃料结构 → 供给侧小时频特征  [读本地 data/raw/EIA]"
$PY scripts/covariates/06_eia_fuel_mix_features.py

step 7/9 "07 合并所有协变量(P0+P1+P2) → merged/{market}_all_covariates_hourly.csv"
$PY scripts/covariates/07_merge_all_covariates.py

# ── 08 GDELT 新闻：半成品（1/96 抽样、用 tone 非 LLM），且未并入 model_ready。
#    如需跑新闻：取消下行注释（慢，逐天下载 518 天）。
# $PY scripts/covariates/08_download_gdelt_news.py

step 8/9 "09 稳健性：差分相关 + 滞后峰值表 + 共线性报告"
$PY scripts/covariates/09_robustness_analysis.py

step 9/9 "10 model_ready：市场专属 + 全滞后无泄漏(政策B) → data/covariates/model_ready/"
$PY scripts/covariates/10_build_model_ready.py

echo
echo "✓ 全流程完成。"
echo "  喂模型:  data/covariates/model_ready/{market}_features_hourly.csv"
echo "  分析:    data/covariates/analysis/*.csv"
echo "  图:      analysis/figures/covariates/*.png"
