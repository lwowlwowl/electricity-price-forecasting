#!/bin/bash
# run_fusion_sweep.sh
# lambda_spike 扫描：依次跑 0.1 / 0.2 / 0.3 三个配置，结果存到各自独立目录
# 用法：bash run_fusion_sweep.sh
# 后台用法：caffeinate -d bash run_fusion_sweep.sh 2>&1 | tee sweep.log

set -e   # 任何一轮报错立即停止

PYTHON="external/timesfm/.venv/bin/python"
CONFIGS=(
    "configs/fusion/electfm_lsp01.yaml"
    "configs/fusion/electfm_lsp02.yaml"
    "configs/fusion/electfm_lsp03.yaml"
)

echo "=========================================="
echo "ElecFM lambda_spike 扫描"
echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo ""
    echo "────────────────────────────────────────"
    echo "▶ 开始：$name  $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────"

    $PYTHON -u src/fusion_model/run_fusion.py --config "$cfg"

    echo "✅ 完成：$name  $(date '+%H:%M:%S')"
done

echo ""
echo "=========================================="
echo "全部完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录："
ls -d data/results/fusion_electfm_lsp*/  2>/dev/null | head -10
echo "=========================================="
