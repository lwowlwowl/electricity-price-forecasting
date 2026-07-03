#!/bin/bash
# 6小时实验串行脚本
# 按优先级排序：allgroups(重要) → V7 SwiGLU(重要) → V6 超参数扫描(补充)

set -e
cd "$(dirname "$0")"
PYTHON="external/timesfm/.venv/bin/python"
RUN="caffeinate -d $PYTHON -u src/fusion_model/run_fusion.py"
mkdir -p logs

echo "========================================"
echo "开始 6小时实验串（$(date)）"
echo "========================================"

# ── 实验 1：V6 15节点全分组（参数/样本比 16，最重要）约 1.5 小时 ──────────────
echo ""
echo "[$(date)] 实验1：V6 allgroups（15节点，参数/样本比 16）"
$RUN --config configs/fusion/electfm_ercot_full_v6_allgroups.yaml \
    2>&1 | tee logs/run_v6_allgroups.log
echo "[$(date)] 实验1 完成"

# ── 实验 2：V7 SwiGLU adapter（Toto 组件，三模型融合叙事完整）约 30 分钟 ──────
echo ""
echo "[$(date)] 实验2：V7 SwiGLU adapter（Toto 组件移植，参数/样本比 6.4）"
$RUN --config configs/fusion/electfm_ercot_full_v7.yaml \
    2>&1 | tee logs/run_v7_swiglu.log
echo "[$(date)] 实验2 完成"

# ── 实验 3：V6 d_attn=32（轻量注意力，敏感性分析）约 30 分钟 ──────────────────
echo ""
echo "[$(date)] 实验3：V6 d_attn=32（敏感性分析）"
$RUN --config configs/fusion/electfm_ercot_full_v6_d32.yaml \
    2>&1 | tee logs/run_v6_d32.log
echo "[$(date)] 实验3 完成"

# ── 实验 4：V6 allgroups + d_attn=32（最保守配置，参数/样本比 10）约 1.5 小时 ──
echo ""
echo "[$(date)] 实验4：V6 allgroups + d_attn=32（参数/样本比 10，最保守）"
cat > /tmp/v6_allgroups_d32.yaml << 'YAML'
market:       ERCOT
nodes: [LZ_LCRA, LZ_WEST, LZ_RAYBN]
freq:         1h
context_len:  168
horizon:      24
pretrained_id: google/timesfm-2.5-200m-pytorch
cross_node_only:      true
v6_allgroups:         true
spike_head_only:      false
use_lora:             false
cross_attn_dim:       32
batch_size:           32
train_stride:         1
stage1_epochs:        30
stage1_lr:            1.0e-3
stage2_epochs:        0
stage2_lr:            0.0
stage2_lr_min:        0.0
gradient_clip:        1.0
weight_decay:         0.01
early_stop_patience:  5
use_amp:              true
lambda_pinball:       0.8
lambda_spike:         0.2
stride_hours:         24
max_origins:          30
spike_quantile:       0.95
tau_search_range:     [0.05, 0.95]
tau_search_step:      0.05
checkpoint_dir:       data/checkpoints/electfm_ercot_full_v6_allgroups_d32
YAML

$RUN --config /tmp/v6_allgroups_d32.yaml \
    2>&1 | tee logs/run_v6_allgroups_d32.log
echo "[$(date)] 实验4 完成"

# ── 实验 5（如果还有时间）：V6 d_attn=128 ────────────────────────────────────
echo ""
echo "[$(date)] 实验5：V6 d_attn=128（更大的注意力，验证是否过拟合）"
$RUN --config configs/fusion/electfm_ercot_full_v6_d128.yaml \
    2>&1 | tee logs/run_v6_d128.log
echo "[$(date)] 实验5 完成"

echo ""
echo "========================================"
echo "全部实验完成（$(date)）"
echo "========================================"
echo ""
echo "结果日志："
echo "  实验1 (V6 allgroups):        logs/run_v6_allgroups.log"
echo "  实验2 (V7 SwiGLU adapter):   logs/run_v7_swiglu.log"
echo "  实验3 (V6 d_attn=32):        logs/run_v6_d32.log"
echo "  实验4 (V6 allgroups+d32):    logs/run_v6_allgroups_d32.log"
echo "  实验5 (V6 d_attn=128):       logs/run_v6_d128.log"
