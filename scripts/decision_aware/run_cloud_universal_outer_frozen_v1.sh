#!/usr/bin/env bash
# One frozen 2024 inference/settlement check; never invokes training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/epf-venv/bin/python}"
RESULT_DIR="data/results/cloud_universal_outer_frozen_v1"
mkdir -p "$RESULT_DIR"
exec 9>"$RESULT_DIR/run.lock"
flock -n 9 || { echo 'Frozen check already running'; exit 1; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS_FILE="$RESULT_DIR/status.tsv"
record_status() { printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "$STATUS_FILE"; }
trap 'record_status pipeline "failed_line_${LINENO}"' ERR
record_status preflight starting
"$PYTHON_BIN" - <<'PY' | tee "$RESULT_DIR/preflight.log"
import hashlib
from pathlib import Path
import sys
import torch
import numpy as np
sys.path.insert(0, 'src')
from decision_aware.contracts import FORMAL_EXPERIMENT_CONTRACT
from decision_aware.global_data import load_formal_entity_mappings
from decision_aware.loader_v2 import resolve_market_unified_path
assert torch.cuda.is_available(), 'CUDA unavailable'
checkpoints = {
    'da': ('universal_lomo_holdout_ercot_mae_v1', '5c3ab8db24d2e6b8cd97745fe207e5ae31171e1359b39400f5f560e08ebb7b0d'),
    'rt_at_da': ('universal_lomo_holdout_ercot_rt_at_da_v1', '105bdb0c2b660e2f4a08247f38b691b83969e8d0ffde6f8cb6da6072b3df263e'),
    'rt': ('universal_lomo_holdout_ercot_mae_v1', '18b84d4254cac551406e3c409a4061d83f4f765d060e5cf0416867988b7f029b'),
}
nodes = load_formal_entity_mappings('configs/nodes.yaml').subset(['ERCOT']).node_to_id
inputs = []
for task, (folder, expected_hash) in checkpoints.items():
    path = Path(f'data/checkpoints/{folder}/{task}_inner_seed0/best.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, path
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    FORMAL_EXPERIMENT_CONTRACT.validate_checkpoint(checkpoint)
    assert checkpoint['phase'] == 'inner' and checkpoint['task'] == task
    assert set(checkpoint['global_training_markets']) == {'CAISO', 'NYISO', 'PJM'}
    cfg = checkpoint['config']
    assert cfg['global_use_entity_embeddings'] is False
    assert cfg['global_num_nodes'] == cfg['global_num_markets'] == 0
    assert cfg['global_normalization_mode'] == 'context_robust'
    assert cfg['seed'] == 0
    inputs.append(path)
    template = Path(f'data/results/global_outer_prediction_templates/outertemplate_{task}_outer_seed0.npz')
    with np.load(template, allow_pickle=False) as archive:
        assert np.all(np.char.startswith(archive['dates'].astype(str), '2024-'))
    inputs.append(template)
    model = 'xgboost_pseudohuber' if task == 'rt_at_da' else 'xgboost_absolute'
    for node in nodes:
        path = Path(f'data/results/global_baselines_v1/{task}_outer/{model}_{node}_seed0.npz')
        with np.load(path, allow_pickle=False) as archive:
            assert str(archive['phase'].item()) == 'outer'
            assert str(archive['task'].item()) == task
            assert np.isfinite(archive['prediction']).all()
        inputs.append(path)
inputs += [Path('configs/decision_aware/global_ercot_huber_v1.yaml'), Path('configs/nodes.yaml')]
inputs.append(Path(resolve_market_unified_path('ERCOT')))
inputs += [Path(f'scripts/decision_aware/{name}') for name in (
    'run_cloud_universal_outer_frozen_v1.sh', 'evaluate_universal_holdout.py',
    'prepare_universal_lomo_profit_archives.py', 'evaluate_global_continuous_mpc.py')]
inputs += [Path('src/decision_aware/loader_v2.py'), Path('src/decision_aware/global_data.py')]
manifest = '\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}' for p in inputs)
Path('data/results/cloud_universal_outer_frozen_v1/inputs.sha256').write_text(manifest + '\n', encoding='utf-8')
print({'gpu': torch.cuda.get_device_name(0), 'inputs_verified': len(inputs), 'training': False, 'test_2025': False})
PY
record_status preflight complete
if [[ "${PREFLIGHT_ONLY:-0}" == '1' ]]; then exit 0; fi
# Stop accidental overwrites of an already completed one-time report.
REPORT="data/results/universal_lomo_ercot_outer_frozen_continuous_mpc_v1.json"
test ! -e "$REPORT"
for task in da rt_at_da rt; do
  folder="universal_lomo_holdout_ercot_mae_v1"
  if [[ "$task" == 'rt_at_da' ]]; then folder="universal_lomo_holdout_ercot_rt_at_da_v1"; fi
  output="data/results/universal_lomo_${task}_outer_frozen_v1/ercot_${task}_v1.npz"
  record_status "$task" inference_starting
  "$PYTHON_BIN" scripts/decision_aware/evaluate_universal_holdout.py \
    --checkpoint "data/checkpoints/$folder/${task}_inner_seed0/best.pt" \
    --task "$task" --holdout-market ERCOT --evaluation-stage outer_frozen \
    --device cuda --batch-size 64 --output "$output" 2>&1 | tee "$RESULT_DIR/${task}.log"
  record_status "$task" inference_complete
done
record_status settlement starting
"$PYTHON_BIN" scripts/decision_aware/prepare_universal_lomo_profit_archives.py \
  --phase outer --template-root data/results/global_outer_prediction_templates \
  --output-root data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1 \
  2>&1 | tee "$RESULT_DIR/prepare.log"
"$PYTHON_BIN" scripts/decision_aware/evaluate_global_continuous_mpc.py \
  --config configs/decision_aware/global_ercot_huber_v1.yaml \
  --prediction-dir data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1 \
  --families universal --phase outer --no-selection-allowed --workers 8 --executor process \
  --include-baselines xgboost_robust --status frozen_inner_seed0_2024_stability_not_final_test \
  --output "$REPORT" 2>&1 | tee "$RESULT_DIR/settlement.log"
"$PYTHON_BIN" - <<'PY' | tee "$RESULT_DIR/conclusion.log"
import hashlib
import json
from pathlib import Path
root = Path('data/results')
report_path = root / 'universal_lomo_ercot_outer_frozen_continuous_mpc_v1.json'
r = json.loads(report_path.read_text(encoding='utf-8'))
assert r['selection_allowed'] is False
ours = 'da_universal__rt_at_da_universal__rt_universal'
base = 'da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust'
assert set(r['systems']) == {ours, base}
a, b = r['systems'][ours], r['systems'][base]
assert a['days'] == b['days'] and a['node_count'] == b['node_count'] == 15
paired = r['paired_vs_baselines']['xgboost_robust'][ours]
ci = paired['7day_block_bootstrap_95ci']
summary = {
    'status': 'complete', 'year': 2024, 'retrained': False, 'selection_allowed': False,
    'checkpoint_training_years': [2020, 2021, 2022],
    'checkpoint_training_markets': ['CAISO', 'NYISO', 'PJM'],
    'baseline_training_years': [2020, 2021, 2022, 2023],
    'baseline_saw_target_market': True, 'equal_training_data_comparison': False,
    'days': a['days'], 'nodes': a['node_count'],
    'universal_usd_per_node_day': a['macro_mean_daily_revenue'],
    'xgboost_usd_per_node_day': b['macro_mean_daily_revenue'],
    'paired': paired, 'positive_mean_difference': paired['mean_daily_revenue_difference'] > 0,
    'paired_95ci_entirely_above_zero': ci[0] > 0,
    'total_soc_violations': a['total_soc_violation_count'],
    'locked_2025_evaluated': False,
    'limits': ['Single seed; no claim of seed stability',
               'ERCOT settlement only; other markets have prediction evidence only',
               'LOMO validation-selected checkpoints; not an untouched-market test',
               'Three independent models; no joint-profit retraining in this check'],
}
(root / 'universal_lomo_ercot_outer_frozen_summary_v1.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 2024冻结时间外盈利检查', '',
         '冻结2020–2022训练的seed 0无ID三模型；2023验证选定。未重训，2025未评价。', '',
         '|系统|美元/节点·日|共同交付日|节点数|', '|---|---:|---:|---:|',
         f"|自有通用三模型|{a['macro_mean_daily_revenue']:.2f}|{a['days']}|15|",
         f"|稳健XGBoost|{b['macro_mean_daily_revenue']:.2f}|{b['days']}|15|", '',
         f"配对收益差：{paired['mean_daily_revenue_difference']:.2f}；7日块bootstrap 95%区间：[{ci[0]:.2f}, {ci[1]:.2f}]。", '',
         'XGBoost使用ERCOT 2020–2023训练数据，自有模型使用其他三市场2020–2022数据；同结算合同，但训练数据不等同。', '',
         '本结果只用于冻结稳定性检查，不据此调参或选模型；未完成多seed稳定性及其他市场完整盈利验证。']
(root / 'cloud_universal_outer_frozen_v1/conclusion.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
files = [report_path, root / 'universal_lomo_ercot_outer_frozen_summary_v1.json']
for folder in ['cloud_universal_outer_frozen_v1', 'universal_lomo_ercot_outer_frozen_profit_predictions_v1',
               'universal_lomo_da_outer_frozen_v1', 'universal_lomo_rt_at_da_outer_frozen_v1', 'universal_lomo_rt_outer_frozen_v1']:
    files.extend(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix in {'.npz', '.json', '.md'})
manifest = '\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}' for p in sorted(set(files)))
(root / 'cloud_universal_outer_frozen_v1/artifacts.sha256').write_text(manifest + '\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
sha256sum -c "$RESULT_DIR/inputs.sha256" > "$RESULT_DIR/inputs_verified_after.log"
record_status pipeline complete
