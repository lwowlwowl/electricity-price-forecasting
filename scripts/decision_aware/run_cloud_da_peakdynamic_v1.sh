#!/usr/bin/env bash
# DA局部峰值幅度后训练v1：源市场训练/选模，ERCOT 2023与冻结2024只做评价。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
cd "$ROOT"

CONFIG="configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_v1.yaml"
BASE_CHECKPOINT="data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt"
CHECKPOINT_DIR="data/checkpoints/universal_lomo_holdout_ercot_da_peakdynamic_v1/da_inner_seed0"
SELECTED_CHECKPOINT="$CHECKPOINT_DIR/selected_peak_dynamic.pt"
SELECTION_SUMMARY="$CHECKPOINT_DIR/peak_dynamic_selection.json"
PREDICTION_DIR="data/results/universal_lomo_ercot_da_peakdynamic_predictions_v1"
INNER_PROFIT_DIR="data/results/universal_lomo_ercot_da_peakdynamic_profit_inner_v1"
OUTER_PROFIT_DIR="data/results/universal_lomo_ercot_da_peakdynamic_profit_outer_v1"
INNER_SETTLEMENT="data/results/universal_lomo_ercot_da_peakdynamic_inner_mpc_v1.json"
OUTER_SETTLEMENT="data/results/universal_lomo_ercot_da_peakdynamic_outer_mpc_v1.json"
SUMMARY_DIR="data/results/universal_lomo_ercot_da_peakdynamic_summary_v1"
RESULT_DIR="data/results/cloud_da_peakdynamic_v1"
ARCHIVE="data/results/cloud_da_peakdynamic_v1_completed_20261020.tar.gz"
mkdir -p "$PREDICTION_DIR" "$INNER_PROFIT_DIR" "$OUTER_PROFIT_DIR" "$SUMMARY_DIR" "$RESULT_DIR"
STATUS_FILE="$RESULT_DIR/status.tsv"
RUN_LOG="$RESULT_DIR/runner.log"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
GPU_MONITOR_PID=""

timestamp() { date --iso-8601=seconds; }
record_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" | tee -a "$STATUS_FILE"; }
start_gpu_monitor() {
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 2 > "$RESULT_DIR/gpu_usage.csv" &
  GPU_MONITOR_PID="$!"
}
stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}
trap 'stop_gpu_monitor; record_status pipeline "failed_line_${LINENO}"' ERR

preflight() {
  test -x "$PYTHON_BIN"
  test -s "$CONFIG"
  test -s "$BASE_CHECKPOINT"
  for task in da rt_at_da rt; do
    test -s "data/results/universal_lomo_${task}_v1/ercot_${task}_v1.npz"
    test -s "data/results/universal_lomo_${task}_outer_frozen_v1/ercot_${task}_v1.npz"
    test -s "data/results/global_f0_f1_stable_v2_predictions/c0_${task}_inner_seed0.npz"
    test -s "data/results/global_outer_prediction_templates/outertemplate_${task}_outer_seed0.npz"
  done
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
import sys
import torch
import yaml
sys.path.insert(0, 'src')
from decision_aware.config import PilotConfig

root = Path.cwd()
base_cfg = yaml.safe_load((root / 'configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml').read_text(encoding='utf-8'))
candidate_path = root / 'configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_v1.yaml'
candidate_cfg = yaml.safe_load(candidate_path.read_text(encoding='utf-8'))
allowed = {
    'global_use_daily_peak_excursion_loss', 'global_daily_peak_excursion_weight',
    'global_daily_peak_excursion_beta', 'global_trainable_scope', 'lr', 'epochs',
    'early_stop_patience', 'global_keep_top_k_checkpoints',
    'global_use_lr_scheduler', 'checkpoint_dir',
}
differences = {key for key in set(base_cfg) | set(candidate_cfg) if base_cfg.get(key) != candidate_cfg.get(key)}
assert differences <= allowed, sorted(differences - allowed)
cfg = PilotConfig.from_yaml(candidate_path)
assert tuple(cfg.global_training_markets) == ('CAISO', 'NYISO', 'PJM')
assert cfg.market == 'ERCOT' and cfg.seed == 0
assert cfg.global_use_entity_embeddings is False
assert cfg.global_normalization_mode == 'context_robust'
assert cfg.da_fusion_mode == 'F1' and cfg.global_training_loss == 'mae'
assert cfg.global_use_daily_peak_excursion_loss is True
assert cfg.global_daily_peak_excursion_weight == 1.0
assert cfg.global_daily_peak_excursion_beta == 1.0
assert cfg.global_use_critical_event_weighting is False
assert cfg.global_trainable_scope == 'decoder_head'
assert cfg.lr == 5e-5 and cfg.epochs == 6
assert cfg.global_keep_top_k_checkpoints == 7
checkpoint = root / 'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '5c3ab8db24d2e6b8cd97745fe207e5ae31171e1359b39400f5f560e08ebb7b0d'
assert torch.cuda.is_available(), 'CUDA unavailable'
print({'gpu': torch.cuda.get_device_name(0), 'training': 'DA decoder/head only', 'epochs': 6, 'test_2025_read': False})
PY
  "$PYTHON_BIN" -m pytest -q -p no:cacheprovider \
    tests/test_local_peak_dynamics.py tests/test_peak_dynamic_pipeline.py
}

train_and_select() {
  local summary="$CHECKPOINT_DIR/summary.json"
  local latest="$CHECKPOINT_DIR/latest.pt"
  if [[ -s "$summary" ]]; then
    record_status train already_complete
  else
    local -a train_args=(--initialize-from "$BASE_CHECKPOINT")
    if [[ -s "$latest" ]]; then train_args=(--resume); fi
    record_status train starting
    "$PYTHON_BIN" scripts/decision_aware/train_global_forecaster.py \
      --config "$CONFIG" --task da --phase inner --seed 0 --device cuda \
      "${train_args[@]}" 2>&1 | tee "$RESULT_DIR/train.log"
    test -s "$summary"
    record_status train complete
  fi
  record_status selection starting
  "$PYTHON_BIN" scripts/decision_aware/select_peak_dynamic_checkpoint.py \
    --checkpoint-dir "$CHECKPOINT_DIR" --mae-guard-fraction 0.02 \
    2>&1 | tee "$RESULT_DIR/selection.log"
  test -s "$SELECTED_CHECKPOINT" && test -s "$SELECTION_SUMMARY"
  record_status selection complete
}

predict_phase() {
  local stage="$1"
  local output="$PREDICTION_DIR/peak_dynamic_da_${stage}_seed0.npz"
  local -a stage_args=()
  if [[ "$stage" == "outer" ]]; then stage_args=(--evaluation-stage outer_frozen); fi
  if [[ -s "$output" && -s "${output%.npz}.json" ]]; then
    record_status "predict_${stage}" already_complete
    return
  fi
  record_status "predict_${stage}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
    --checkpoint "$SELECTED_CHECKPOINT" --task da --holdout-market ERCOT \
    --output "$output" --device cuda --batch-size 64 "${stage_args[@]}" \
    2>&1 | tee "$RESULT_DIR/predict_${stage}.log"
  record_status "predict_${stage}" complete
}

prepare_family() {
  local phase="$1" family="$2" da_source="$3" rtda_source="$4" rt_source="$5"
  local output_root="$INNER_PROFIT_DIR" template_root="data/results/global_f0_f1_stable_v2_predictions"
  if [[ "$phase" == "outer" ]]; then
    output_root="$OUTER_PROFIT_DIR"
    template_root="data/results/global_outer_prediction_templates"
  fi
  "$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
    --phase "$phase" --family "$family" --seed 0 --template-root "$template_root" \
    --output-root "$output_root" --source-da "$da_source" \
    --source-rt-at-da "$rtda_source" --source-rt "$rt_source"
}

settle_phase() {
  local phase="$1" output="$2" prediction_root="$INNER_PROFIT_DIR"
  local suffix="" no_selection=()
  if [[ "$phase" == "outer" ]]; then
    suffix="_outer_frozen"
    prediction_root="$OUTER_PROFIT_DIR"
    no_selection=(--no-selection-allowed)
  fi
  local current_da="data/results/universal_lomo_da${suffix}_v1/ercot_da_v1.npz"
  local current_rtda="data/results/universal_lomo_rt_at_da${suffix}_v1/ercot_rt_at_da_v1.npz"
  local current_rt="data/results/universal_lomo_rt${suffix}_v1/ercot_rt_v1.npz"
  local candidate_da="$PREDICTION_DIR/peak_dynamic_da_${phase}_seed0.npz"
  prepare_family "$phase" current "$current_da" "$current_rtda" "$current_rt"
  prepare_family "$phase" peak_dynamic "$candidate_da" "$current_rtda" "$current_rt"
  local triplets="da_current__rt_at_da_current__rt_current,da_peak_dynamic__rt_at_da_current__rt_current,da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
  record_status "settlement_${phase}" starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
    --config configs/decision_aware/global_ercot_huber_v1.yaml \
    --prediction-dir "$prediction_root" --families current,peak_dynamic \
    --phase "$phase" --triplets "$triplets" "${no_selection[@]}" \
    --include-baselines xgboost_robust \
    --workers 8 --executor process --status "da_peakdynamic_${phase}_v1" \
    --output "$output" 2>&1 | tee "$RESULT_DIR/settlement_${phase}.log"
  record_status "settlement_${phase}" complete
}

summarize_and_package() {
  record_status summary starting
  "$PYTHON_BIN" scripts/decision_aware/summarize_peak_dynamic_da.py \
    --inner-candidate "$PREDICTION_DIR/peak_dynamic_da_inner_seed0.npz" \
    --outer-candidate "$PREDICTION_DIR/peak_dynamic_da_outer_seed0.npz" \
    --inner-settlement "$INNER_SETTLEMENT" --outer-settlement "$OUTER_SETTLEMENT" \
    --selection-summary "$SELECTION_SUMMARY" --output-dir "$SUMMARY_DIR" \
    2>&1 | tee "$RESULT_DIR/summary.log"
  record_status summary complete
  record_status package starting
  local manifest="${ARCHIVE%.tar.gz}.manifest.sha256"
  find "$RESULT_DIR" "$PREDICTION_DIR" "$INNER_PROFIT_DIR" "$OUTER_PROFIT_DIR" \
    "$SUMMARY_DIR" "$CHECKPOINT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$manifest"
  sha256sum "$CONFIG" "$INNER_SETTLEMENT" "$OUTER_SETTLEMENT" >> "$manifest"
  tar -czf "$ARCHIVE" "$CONFIG" "$RESULT_DIR" "$PREDICTION_DIR" \
    "$INNER_PROFIT_DIR" "$OUTER_PROFIT_DIR" "$SUMMARY_DIR" "$CHECKPOINT_DIR" \
    "$INNER_SETTLEMENT" "$OUTER_SETTLEMENT" "$manifest"
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
  record_status package complete
}

{
  record_status preflight starting
  preflight
  record_status preflight complete
  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then exit 0; fi
  start_gpu_monitor
  train_and_select
  predict_phase inner
  predict_phase outer
  stop_gpu_monitor
  settle_phase inner "$INNER_SETTLEMENT"
  settle_phase outer "$OUTER_SETTLEMENT"
  summarize_and_package
  record_status pipeline complete
} 2>&1 | tee -a "$RUN_LOG"
