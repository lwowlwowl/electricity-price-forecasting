param(
    [int]$Epochs = 3,
    [int]$Workers = 4,
    [switch]$Round2Only
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runRoot = Join-Path $repo "data\results\global_implicit_qp_run_v1"
$rawRoot = Join-Path $repo "data\results\global_implicit_qp_raw_predictions_v1"
$profitRoot = Join-Path $repo "data\results\global_implicit_qp_profit_predictions_v1"
$settlementRoot = Join-Path $repo "data\results\global_implicit_qp_settlement_v1"
$checkpointRoot = Join-Path $repo "data\checkpoints\global_joint_implicit_qp_v1"
New-Item -ItemType Directory -Force -Path $runRoot, $rawRoot, $profitRoot, $settlementRoot, $checkpointRoot | Out-Null
$status = Join-Path $runRoot "status.tsv"
$log = Join-Path $runRoot "runner.log"
if (-not $Round2Only) {
    "timestamp`tstage`tfamily`tstatus" | Set-Content -Encoding UTF8 $status
} elseif (-not (Test-Path -LiteralPath $status)) {
    throw "Round2Only requires the completed round-1 status file: $status"
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

if (-not $Round2Only) {
    Invoke-Logged "prepare_xgboost" "xgbpooled" @(
        "scripts/decision_aware/prepare_universal_lomo_profit_archives.py",
        "--phase", "inner",
        "--output-root", "data/results/global_implicit_qp_profit_predictions_v1",
        "--family", "xgbpooled",
        "--source-da", "data/results/universal_lomo_xgboost_v1/holdout_ercot_da_inner_seed0.npz",
        "--source-rt-at-da", "data/results/universal_lomo_xgboost_v1/holdout_ercot_rt_at_da_inner_seed0.npz",
        "--source-rt", "data/results/universal_lomo_xgboost_v1/holdout_ercot_rt_inner_seed0.npz"
    )
} else {
    foreach ($task in @("da", "rt_at_da", "rt")) {
        $baselineArchive = Join-Path $profitRoot "xgbpooled_$task`_inner_seed0.npz"
        if (-not (Test-Path -LiteralPath $baselineArchive)) {
            throw "Round2Only requires frozen baseline archive: $baselineArchive"
        }
    }
}

$round1 = @(
    @{ Family = "implicitqp_balanced_seed0"; Weight = "0.50"; LearningRate = "2e-4"; Scope = "price_head"; Quadratic = "0.25"; Barrier = "0.25" },
    @{ Family = "implicitqp_decision_seed0"; Weight = "0.80"; LearningRate = "2e-4"; Scope = "price_head"; Quadratic = "0.25"; Barrier = "0.25" },
    @{ Family = "implicitqp_sharp_seed0"; Weight = "0.70"; LearningRate = "1e-4"; Scope = "price_head"; Quadratic = "0.10"; Barrier = "0.05" }
)

function Run-Candidate($candidate) {
    $family = $candidate.Family
    $weight = $candidate.Weight
    Invoke-Logged "train" $family @(
        "scripts/decision_aware/train_global_joint_implicit_qp.py",
        "--seed", "0",
        "--family", $family,
        "--output-dir", "data/checkpoints/global_joint_implicit_qp_v1",
        "--result", "data/results/global_implicit_qp_run_v1/$family`_train.json",
        "--epochs", "$Epochs",
        "--episodes-per-node", "2",
        "--decision-weight", $weight,
        "--lr", $candidate.LearningRate,
        "--scope", $candidate.Scope,
        "--quadratic-regularization", $candidate.Quadratic,
        "--barrier-temperature", $candidate.Barrier,
        "--device", "cuda"
    )

    foreach ($task in @("da", "rt_at_da", "rt")) {
        $checkpoint = "data/checkpoints/global_joint_implicit_qp_v1/$family/$task.pt"
        $output = "data/results/global_implicit_qp_raw_predictions_v1/$family`_$task`_inner_seed0.npz"
        Invoke-Logged "export_$task" $family @(
            "scripts/decision_aware/evaluate_universal_holdout.py",
            "--checkpoint", $checkpoint,
            "--task", $task,
            "--holdout-market", "ERCOT",
            "--output", $output,
            "--evaluation-stage", "inner",
            "--device", "cuda"
        )
    }

    Invoke-Logged "prepare_profit" $family @(
        "scripts/decision_aware/prepare_universal_lomo_profit_archives.py",
        "--phase", "inner",
        "--output-root", "data/results/global_implicit_qp_profit_predictions_v1",
        "--family", $family,
        "--source-da", "data/results/global_implicit_qp_raw_predictions_v1/$family`_da_inner_seed0.npz",
        "--source-rt-at-da", "data/results/global_implicit_qp_raw_predictions_v1/$family`_rt_at_da_inner_seed0.npz",
        "--source-rt", "data/results/global_implicit_qp_raw_predictions_v1/$family`_rt_inner_seed0.npz"
    )

    $candidateTriplet = "da_$family`__rt_at_da_$family`__rt_$family"
    $baselineTriplet = "da_xgbpooled__rt_at_da_xgbpooled__rt_xgbpooled"
    Invoke-Logged "settlement" $family @(
        "scripts/decision_aware/evaluate_global_continuous_mpc.py",
        "--config", "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
        "--prediction-dir", "data/results/global_implicit_qp_profit_predictions_v1",
        "--families", "$family,xgbpooled",
        "--phase", "inner",
        "--triplets", "$candidateTriplet,$baselineTriplet",
        "--workers", "$Workers",
        "--executor", "process",
        "--status", "2023_inner_implicit_qp_candidate_not_final",
        "--output", "data/results/global_implicit_qp_settlement_v1/$family`_continuous_mpc.json"
    )
}

function Write-Summary() {
    Invoke-Logged "summarize" "all" @(
        "scripts/decision_aware/summarize_implicit_qp_candidates.py",
        "--input-dir", "data/results/global_implicit_qp_settlement_v1",
        "--output", "data/results/global_implicit_qp_summary_v1.json"
    )
}

$round2 = @(
    @{ Family = "implicitqp_decoder_balanced_seed0"; Weight = "0.60"; LearningRate = "2e-5"; Scope = "decoder_head"; Quadratic = "0.25"; Barrier = "0.10" },
    @{ Family = "implicitqp_decoder_decision_seed0"; Weight = "0.85"; LearningRate = "1e-5"; Scope = "decoder_head"; Quadratic = "0.10"; Barrier = "0.05" }
)

if ($Round2Only) {
    Write-Status "adaptive_round2" "all" "resumed_by_user"
    foreach ($candidate in $round2) { Run-Candidate $candidate }
    Write-Summary
    Write-Status "adaptive_round2" "all" "complete"
    Write-Status "runner" "all" "complete"
    exit 0
}

foreach ($candidate in $round1) { Run-Candidate $candidate }
Write-Summary

$summary = Get-Content "data/results/global_implicit_qp_summary_v1.json" -Raw | ConvertFrom-Json
$strong = @($summary.rows | Where-Object {
    $_.candidate_macro_mean_daily_revenue -ge 438.9681 -and
    $_.passes_paired_interval -and $_.passes_remove_top5 -and
    $_.improved_nodes -ge 10
})
if ($strong.Count -eq 0) {
    Write-Status "adaptive_round2" "all" "started_no_strong_round1_candidate"
    foreach ($candidate in $round2) { Run-Candidate $candidate }
    Write-Summary
    Write-Status "adaptive_round2" "all" "complete"
} else {
    Write-Status "adaptive_round2" "all" "skipped_strong_round1_candidate"
}
Write-Status "runner" "all" "complete"

