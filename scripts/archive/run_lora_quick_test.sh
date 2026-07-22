#!/bin/bash
# ElecFM LoRA 快速测试脚本
# 使用方法: bash run_lora_quick_test.sh
# 预计时间: 10-15 分钟

set -e

cd "$(dirname "$0")"

PYTHON="external/timesfm/.venv/bin/python"

# 创建测试配置文件
mkdir -p data/checkpoints/electfm_lora_test
mkdir -p data/results/fusion_electfm_lora_test

cat > configs/fusion/electfm_lora_quick.yaml << 'EOF'
market:       ERCOT
nodes:
  - HB_HOUSTON
freq:         1h
context_len:  168
horizon:      24

pretrained_id: google/timesfm-2.5-200m-pytorch

batch_size:           8
train_stride:         4      # 快速测试，减少样本数

# LoRA 配置
use_lora:             true
lora_r:               4
lora_alpha:           8
lora_dropout:         0.05

# 训练参数
stage1_epochs:        2
stage1_lr:            1.0e-5

stage2_epochs:        2
stage2_lr:            5.0e-7
stage2_lr_min:        5.0e-8

gradient_clip:        1.0
weight_decay:         0.01
early_stop_patience:  5
use_amp:              true

lambda_pinball:       0.8
lambda_spike:         0.2

stride_hours:         24
max_origins:          10
spike_quantile:       0.95
tau_search_range:     [0.05, 0.95]
tau_search_step:      0.05

checkpoint_dir:       data/checkpoints/electfm_lora_test
EOF

echo "========================================"
echo "ElecFM LoRA 快速测试"
echo "========================================"
echo ""
echo "配置:"
echo "  - 节点: HB_HOUSTON (单节点快速测试)"
echo "  - LoRA: r=4, alpha=8"
echo "  - Stage 1: 2 epochs"
echo "  - Stage 2: 2 epochs"
echo ""
echo "开始运行..."
echo ""

$PYTHON -u src/fusion_model/run_fusion.py \
    --config configs/fusion/electfm_lora_quick.yaml \
    2>&1 | tee test_lora.log

echo ""
echo "========================================"
echo "✅ 测试完成!"
echo "========================================"
echo ""
echo "查看结果:"
echo "  日志: tail -50 test_lora.log"
echo "  结果: ls data/results/fusion_electfm_lora_quick/"
