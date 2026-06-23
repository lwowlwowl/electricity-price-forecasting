#!/bin/bash
# 批量运行所有消融实验（max_origins=30）
# 用法：nohup bash run_all_ablations.sh > ablation_run.log 2>&1 &

cd /Users/wanghaochen/school
export PYTHONUNBUFFERED=1

LOGFILE="/Users/wanghaochen/school/ablation_run.log"

echo "========================================"
echo "开始时间: $(date)"
echo "========================================"

CONFIGS=(
  "configs/parameter_ablation/ablation_A_covariates.yaml"
  "configs/parameter_ablation/ablation_A_covariates_w2_negative.yaml"
  "configs/parameter_ablation/ablation_A_covariates_w3_extreme.yaml"
  "configs/parameter_ablation/ablation_B_context.yaml"
  "configs/parameter_ablation/ablation_B_context_w2_negative.yaml"
  "configs/parameter_ablation/ablation_B_context_w3_extreme.yaml"
  "configs/parameter_ablation/ablation_C_multivariate.yaml"
  "configs/parameter_ablation/ablation_C_multivariate_w2_negative.yaml"
  "configs/parameter_ablation/ablation_C_multivariate_w3_extreme.yaml"
  "configs/parameter_ablation/ablation_D_horizon.yaml"
  "configs/parameter_ablation/ablation_D_horizon_w2_negative.yaml"
  "configs/parameter_ablation/ablation_D_horizon_w3_extreme.yaml"
  "configs/parameter_ablation/ablation_F_frequency.yaml"
  "configs/parameter_ablation/ablation_F_frequency_w2_negative.yaml"
  "configs/parameter_ablation/ablation_F_frequency_w3_extreme.yaml"
)

TOTAL=${#CONFIGS[@]}
DONE=0
FAILED=0

for cfg in "${CONFIGS[@]}"; do
  DONE=$((DONE + 1))
  echo ""
  echo "======================================== [$DONE/$TOTAL]"
  echo "▶ 运行: $cfg"
  echo "  开始: $(date)"
  echo "========================================"

  python3 src/parameter_ablation/run_ablation.py "$cfg"
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 完成: $cfg ($(date))"
  else
    echo "⚠️  退出码=$EXIT_CODE: $cfg ($(date))"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "========================================"
echo "全部完成: $(date)"
echo "成功: $((TOTAL - FAILED))/$TOTAL  失败: $FAILED"
echo "========================================"
