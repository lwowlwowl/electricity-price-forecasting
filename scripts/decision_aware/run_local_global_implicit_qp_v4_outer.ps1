param([int]$Workers = 4)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
$checkpointRoot = "data/checkpoints/global_joint_implicit_qp_v1"
$rawRoot = "data/results/global_implicit_qp_outer_raw_v4"
$profitRoot = "data/results/global_implicit_qp_outer_profit_v4"
$runRoot = Join-Path $repo "data\results\global_implicit_qp_outer_run_v4"
New-Item -ItemType Directory -Force -Path $rawRoot, $profitRoot, $runRoot | Out-Null
$status = Join-Path $runRoot "status.tsv"
$log = Join-Path $runRoot "runner.log"
if (-not (Test-Path -LiteralPath $status)) {
    "timestamp`tstage`tfamily`tstatus" | Set-Content -Encoding UTF8 $status
}

function Write-Status([string]$Stage, [string]$Family, [string]$State) {
    "$(Get-Date -Format o)`t$Stage`t$Family`t$State" | Add-Content -Encoding UTF8 $status
}

function Invoke-Logged([string]$Stage, [string]$Family, [string[]]$Arguments) {
    Write-Status $Stage $Family "started"
    & $python @Arguments 2>&1 | Tee-Object -Append -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Status $Stage $Family "failed"
        throw "$Stage failed for $Family with exit code $LASTEXITCODE"
    }
    Write-Status $Stage $Family "complete"
}

$members = @(
    "implicitqp_decoder_balanced_seed0",
    "implicitqp_decoder_balanced_seed1",
    "implicitqp_decoder_balanced_seed2"
)
foreach ($family in $members) {
    foreach ($task in @("da", "rt_at_da", "rt")) {
        $rawOutput = "$rawRoot/$family`_$task`_outer_seed0.npz"
        if (-not (Test-Path -LiteralPath $rawOutput)) {
            Invoke-Logged "export_outer_$task" $family @(
                "scripts/decision_aware/evaluate_universal_holdout.py",
                "--checkpoint", "$checkpointRoot/$family/$task.pt",
                "--task", $task,
                "--holdout-market", "ERCOT",
                "--output", $rawOutput,
                "--evaluation-stage", "outer_frozen",
                "--device", "cuda"
            )
        } else {
            Write-Status "export_outer_$task" $family "reused"
        }
    }
    Invoke-Logged "prepare_outer_profit" $family @(
        "scripts/decision_aware/prepare_universal_lomo_profit_archives.py",
        "--phase", "outer",
        "--template-root", "data/results/global_outer_prediction_templates",
        "--output-root", $profitRoot,
        "--family", $family,
        "--source-da", "$rawRoot/$family`_da_outer_seed0.npz",
        "--source-rt-at-da", "$rawRoot/$family`_rt_at_da_outer_seed0.npz",
        "--source-rt", "$rawRoot/$family`_rt_outer_seed0.npz"
    )
}

Invoke-Logged "apply_outer_calibrator" "implicitqp_robust_ensemble_v1" @(
    "scripts/decision_aware/apply_robust_ensemble_calibrator.py",
    "--prediction-dir", $profitRoot,
    "--checkpoint", "data/checkpoints/global_robust_ensemble_v1/calibrator.pt",
    "--output-family", "implicitqp_robust_ensemble_v1",
    "--phase", "outer"
)

$finalFamily = "implicitqp_robust_seed0_twothirds_v4"
Invoke-Logged "prepare_outer_v4" $finalFamily @(
    "scripts/decision_aware/prepare_fixed_family_ensemble.py",
    "--prediction-dir", $profitRoot,
    "--members", "implicitqp_robust_ensemble_v1,implicitqp_decoder_balanced_seed0",
    "--weights", "1,2",
    "--selection-by-profit",
    "--output-family", $finalFamily,
    "--phase", "outer"
)

$candidateTriplet = "da_$finalFamily`__rt_at_da_$finalFamily`__rt_$finalFamily"
$baselineTriplet = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
Invoke-Logged "outer_exact_settlement" $finalFamily @(
    "scripts/decision_aware/evaluate_global_continuous_mpc.py",
    "--config", "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
    "--prediction-dir", $profitRoot,
    "--families", $finalFamily,
    "--phase", "outer",
    "--include-baselines", "xgboost_robust",
    "--triplets", "$candidateTriplet,$baselineTriplet",
    "--no-selection-allowed",
    "--workers", "$Workers",
    "--executor", "process",
    "--status", "2024_outer_frozen_v4_confirmation",
    "--output", "data/results/global_implicit_qp_v4_outer_continuous_mpc.json"
)
Write-Status "runner" "all" "complete"

