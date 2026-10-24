param(
    [int]$Epochs = 8,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runRoot = Join-Path $repo "data\results\global_robust_ensemble_run_v1"
$settlementRoot = Join-Path $repo "data\results\global_robust_ensemble_settlement_v1"
$profitRoot = Join-Path $repo "data\results\global_implicit_qp_profit_predictions_v1"
New-Item -ItemType Directory -Force -Path $runRoot, $settlementRoot | Out-Null
$status = Join-Path $runRoot "status.tsv"
$log = Join-Path $runRoot "runner.log"
"timestamp`tstage`tstatus`tdetail" | Set-Content -Encoding UTF8 $status

function Write-Status([string]$Stage, [string]$State, [string]$Detail = "") {
    "$(Get-Date -Format o)`t$Stage`t$State`t$Detail" | Add-Content -Encoding UTF8 $status
}

function Invoke-Logged([string]$Stage, [string[]]$Arguments) {
    Write-Status $Stage "started"
    & $python @Arguments 2>&1 | Tee-Object -Append -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Status $Stage "failed" "exit_code=$LASTEXITCODE"
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
    Write-Status $Stage "complete"
}

$trainReportPath = Join-Path $repo "data\results\global_robust_ensemble_train_v1.json"
Invoke-Logged "source_only_train_and_validation" @(
    "scripts/decision_aware/train_global_robust_ensemble_calibrator.py",
    "--epochs", "$Epochs",
    "--newton-steps", "120",
    "--device", "cuda"
)

$trainReport = Get-Content -LiteralPath $trainReportPath -Raw | ConvertFrom-Json
if (-not $trainReport.source_validation_improved) {
    Write-Status "ercot_gate" "blocked" "source_validation_not_all_positive"
    Write-Status "runner" "complete" "stopped_before_ercot"
    exit 0
}
Write-Status "ercot_gate" "passed" "source_validation_all_positive"

Invoke-Logged "apply_frozen_calibrator" @(
    "scripts/decision_aware/apply_robust_ensemble_calibrator.py",
    "--prediction-dir", "data/results/global_implicit_qp_profit_predictions_v1",
    "--checkpoint", "data/checkpoints/global_robust_ensemble_v1/calibrator.pt",
    "--output-family", "implicitqp_robust_ensemble_v1",
    "--phase", "inner"
)

$family = "implicitqp_robust_ensemble_v1"
$candidateTriplet = "da_$family`__rt_at_da_$family`__rt_$family"
$baselineTriplet = "da_xgbpooled__rt_at_da_xgbpooled__rt_xgbpooled"
Invoke-Logged "ercot_exact_settlement" @(
    "scripts/decision_aware/evaluate_global_continuous_mpc.py",
    "--config", "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
    "--prediction-dir", "data/results/global_implicit_qp_profit_predictions_v1",
    "--families", "$family,xgbpooled",
    "--phase", "inner",
    "--triplets", "$candidateTriplet,$baselineTriplet",
    "--workers", "$Workers",
    "--executor", "process",
    "--status", "2023_inner_source_robust_ensemble_development",
    "--output", "data/results/global_robust_ensemble_settlement_v1/$family`_continuous_mpc.json"
)

Invoke-Logged "summarize" @(
    "scripts/decision_aware/summarize_implicit_qp_candidates.py",
    "--input-dir", "data/results/global_robust_ensemble_settlement_v1",
    "--output", "data/results/global_robust_ensemble_summary_v1.json"
)
Write-Status "runner" "complete" "ercot_evaluated"

