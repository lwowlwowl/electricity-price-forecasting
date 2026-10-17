#!/usr/bin/env bash
# Prepare an isolated official Transformers/PEFT runtime for TimesFM-2.5 LoRA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="data/results/cloud_timesfm25_lora_setup_v1"
VENV_DIR="external/timesfm25_lora/.venv"
BASE_PYTHON="${BASE_PYTHON:-/root/autodl-tmp/epf-venv/bin/python}"
mkdir -p "$RESULT_DIR" "$(dirname "$VENV_DIR")"

STATUS_FILE="$RESULT_DIR/status.tsv"
record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$STATUS_FILE"
}
trap 'record_status setup "failed_line_${LINENO}"' ERR

record_status setup starting
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

record_status packages starting
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  'transformers==5.8.0' \
  'peft==0.18.0' \
  'accelerate==1.12.0' \
  safetensors huggingface_hub
record_status packages complete

export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$ROOT/hf_cache/hub"
export TRANSFORMERS_CACHE="$ROOT/hf_cache/hub"
export PYTHONUNBUFFERED=1
MODEL_SNAPSHOT="${TIMESFM25_MODEL_PATH:-$ROOT/hf_cache/hub/models--google--timesfm-2.5-200m-transformers/snapshots/5a9806b9b291fad9233b5249d88263f1846304d3}"

record_status checkpoint starting
export TIMESFM25_MODEL_PATH="$MODEL_SNAPSHOT"
test -s "$MODEL_SNAPSHOT/config.json"
test -s "$MODEL_SNAPSHOT/model.safetensors"
"$VENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import accelerate
import peft
import torch
import transformers
from transformers import TimesFm2_5ModelForPrediction

model_id = "google/timesfm-2.5-200m-transformers"
model_source = os.environ["TIMESFM25_MODEL_PATH"]
model = TimesFm2_5ModelForPrediction.from_pretrained(model_source, dtype="auto")
report = {
    "model_id": model_id,
    "model_source": model_source,
    "model_class": type(model).__name__,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "cuda_available": torch.cuda.is_available(),
}
path = Path("data/results/cloud_timesfm25_lora_setup_v1/runtime.json")
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
PY
record_status checkpoint complete

"$VENV_DIR/bin/python" -m pip freeze > "$RESULT_DIR/pip_freeze.txt"
record_status setup complete
