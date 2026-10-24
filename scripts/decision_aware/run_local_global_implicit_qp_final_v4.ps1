param([int]$Workers = 4)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runRoot = Join-Path $repo "data\results\global_implicit_qp_final_v4"
$checkpointRoot = "data/checkpoints/global_joint_implicit_qp_final_v4"
$rawRoot = "data/results/global_implicit_qp_final_v4/raw_2025"
$profitRoot = "data/results/global_implicit_qp_final_v4/profit_2025"
$calibrator = "data/checkpoints/global_robust_ensemble_final_v4/calibrator.pt"
$status = Join-Path $runRoot "status.tsv"
$log = Join-Path $runRoot "runner.log"
New-Item -ItemType Directory -Force -Path $runRoot, $checkpointRoot, $rawRoot, $profitRoot | Out-Null
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

$families = @(
    "implicitqp_final_decoder_balanced_seed0_v4",
    "implicitqp_final_decoder_balanced_seed1_v4",
    "implicitqp_final_decoder_balanced_seed2_v4"
)

foreach ($seed in @(0, 1, 2)) {
    $family = $families[$seed]
    $familyDir = Join-Path $checkpointRoot $family
    $complete = @("da.pt", "rt_at_da.pt", "rt.pt") | ForEach-Object {
        Test-Path -LiteralPath (Join-Path $familyDir $_)
    }
    if ($complete -notcontains $false) {
        Write-Status "train" $family "reused_complete"
        continue
    }
    Invoke-Logged "train" $family @(
        "scripts/decision_aware/train_global_joint_implicit_qp.py",
        "--seed", "$seed",
        "--da-checkpoint", "data/checkpoints/allmarket_final_da_mae_v1/da_final_seed$seed/best.pt",
        "--rt-at-da-checkpoint", "data/checkpoints/allmarket_final_rt_at_da_pseudohuber_v1/rt_at_da_final_seed$seed/best.pt",
        "--rt-checkpoint", "data/checkpoints/allmarket_final_rt_mae_v1/rt_final_seed$seed/best.pt",
        "--family", $family,
        "--output-dir", $checkpointRoot,
        "--result", "$runRoot/$family`_train.json",
        "--epochs", "3",
        "--episode-days", "1",
        "--episodes-per-node", "2",
        "--decision-weight", "0.60",
        "--revenue-scale", "500",
        "--prediction-scale", "25",
        "--lr", "2e-5",
        "--weight-decay", "0.01",
        "--scope", "decoder_head",
        "--quadratic-regularization", "0.25",
        "--barrier-temperature", "0.10",
        "--newton-steps", "60",
        "--source-phase", "final",
        "--train-stage", "final_train",
        "--device", "cuda",
        "--training-seed", "20261023"
    )
}

if (-not (Test-Path -LiteralPath $calibrator)) {
    Invoke-Logged "calibrate" "implicitqp_final_robust_ensemble_v4" @(
        "scripts/decision_aware/train_global_robust_ensemble_calibrator.py",
        "--members", ($families -join ","),
        "--checkpoint-root", $checkpointRoot,
        "--output", $calibrator,
        "--report", "$runRoot/calibrator_train.json",
        "--source-cache", "$runRoot/calibrator_cache.pt",
        "--resume-state", "$runRoot/calibrator_resume.pt",
        "--epochs", "8",
        "--train-per-market", "128",
        "--validation-per-market", "64",
        "--samples-per-market-batch", "4",
        "--lr", "5e-3",
        "--prediction-weight", "0.05",
        "--market-std-weight", "0.50",
        "--worst-market-weight", "0.50",
        "--regularization-weight", "0.01",
        "--quadratic-regularization", "0.25",
        "--barrier-temperature", "0.10",
        "--newton-steps", "60",
        "--source-phase", "final",
        "--train-stage", "final_train",
        "--validation-stage", "final_train",
        "--selection-mode", "fixed_last",
        "--device", "cuda",
        "--seed", "20261024"
    )
} else {
    Write-Status "calibrate" "implicitqp_final_robust_ensemble_v4" "reused_complete"
}

foreach ($family in $families) {
    foreach ($task in @("da", "rt_at_da", "rt")) {
        $raw = "$rawRoot/$family`_$task`_final_seed0.npz"
        if (-not (Test-Path -LiteralPath $raw)) {
            Invoke-Logged "export_2025_$task" $family @(
                "scripts/decision_aware/evaluate_universal_holdout.py",
                "--checkpoint", "$checkpointRoot/$family/$task.pt",
                "--task", $task,
                "--holdout-market", "ERCOT",
                "--evaluation-stage", "locked_2025",
                "--output", $raw,
                "--device", "cuda",
                "--batch-size", "64"
            )
        } else {
            Write-Status "export_2025_$task" $family "reused_complete"
        }
    }
    Invoke-Logged "prepare_profit" $family @(
        "scripts/decision_aware/prepare_universal_lomo_profit_archives.py",
        "--phase", "final",
        "--template-root", "data/results/allmarket_locked_2025_ercot_profit_v1",
        "--output-root", $profitRoot,
        "--family", $family,
        "--source-da", "$rawRoot/$family`_da_final_seed0.npz",
        "--source-rt-at-da", "$rawRoot/$family`_rt_at_da_final_seed0.npz",
        "--source-rt", "$rawRoot/$family`_rt_final_seed0.npz"
    )
}

$robustFamily = "implicitqp_final_robust_ensemble_v4"
Invoke-Logged "apply_calibrator" $robustFamily @(
    "scripts/decision_aware/apply_robust_ensemble_calibrator.py",
    "--prediction-dir", $profitRoot,
    "--checkpoint", $calibrator,
    "--output-family", $robustFamily,
    "--phase", "final"
)

$finalFamily = "implicitqp_final_robust_seed0_twothirds_v4"
Invoke-Logged "prepare_v4" $finalFamily @(
    "scripts/decision_aware/prepare_fixed_family_ensemble.py",
    "--prediction-dir", $profitRoot,
    "--members", "$robustFamily,$($families[0])",
    "--weights", "1,2",
    "--selection-by-profit",
    "--output-family", $finalFamily,
    "--phase", "final"
)

$triplet = "da_$finalFamily`__rt_at_da_$finalFamily`__rt_$finalFamily"
Invoke-Logged "settlement_2025" $finalFamily @(
    "scripts/decision_aware/evaluate_global_continuous_mpc.py",
    "--config", "configs/decision_aware/allmarket_final_da_mae_v1.yaml",
    "--prediction-dir", $profitRoot,
    "--families", $finalFamily,
    "--phase", "final",
    "--triplets", $triplet,
    "--no-selection-allowed",
    "--workers", "$Workers",
    "--executor", "process",
    "--status", "locked_2025_v4_followup_no_post_score_selection",
    "--output", "$runRoot/settlement_2025.json"
)

Invoke-Logged "summarize_2025" $finalFamily @(
    "scripts/decision_aware/summarize_final_decision_aware_v4.py",
    "--candidate-settlement", "$runRoot/settlement_2025.json",
    "--output", "$runRoot/summary_2025.json"
)
Write-Status "runner" "all" "complete"
