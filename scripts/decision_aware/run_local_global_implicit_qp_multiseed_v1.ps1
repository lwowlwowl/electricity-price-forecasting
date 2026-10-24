param(
    [int]$Epochs = 3,
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runRoot = Join-Path $repo "data\results\global_implicit_qp_multiseed_run_v1"
$rawRoot = Join-Path $repo "data\results\global_implicit_qp_raw_predictions_v1"
$profitRoot = Join-Path $repo "data\results\global_implicit_qp_profit_predictions_v1"
$checkpointRoot = Join-Path $repo "data\checkpoints\global_joint_implicit_qp_v1"
New-Item -ItemType Directory -Force -Path $runRoot, $rawRoot, $profitRoot, $checkpointRoot | Out-Null
$status = Join-Path $runRoot "status.tsv"
$log = Join-Path $runRoot "runner.log"
"timestamp`tstage`tfamily`tstatus" | Set-Content -Encoding UTF8 $status

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

$seed0Family = "implicitqp_decoder_balanced_seed0"
foreach ($task in @("da", "rt_at_da", "rt")) {
    $seed0Archive = Join-Path $profitRoot "$seed0Family`_$task`_inner_seed0.npz"
    if (-not (Test-Path -LiteralPath $seed0Archive)) {
        throw "Missing frozen seed-0 archive: $seed0Archive"
    }
}

foreach ($seed in @(1, 2)) {
    $family = "implicitqp_decoder_balanced_seed$seed"
    Invoke-Logged "train" $family @(
        "scripts/decision_aware/train_global_joint_implicit_qp.py",
        "--seed", "$seed",
        "--family", $family,
        "--output-dir", "data/checkpoints/global_joint_implicit_qp_v1",
        "--result", "data/results/global_implicit_qp_multiseed_run_v1/$family`_train.json",
        "--epochs", "$Epochs",
        "--episodes-per-node", "2",
        "--decision-weight", "0.60",
        "--lr", "2e-5",
        "--scope", "decoder_head",
        "--quadratic-regularization", "0.25",
        "--barrier-temperature", "0.10",
        "--device", "cuda",
        "--training-seed", "20261023"
    )

    foreach ($task in @("da", "rt_at_da", "rt")) {
        Invoke-Logged "export_$task" $family @(
            "scripts/decision_aware/evaluate_universal_holdout.py",
            "--checkpoint", "data/checkpoints/global_joint_implicit_qp_v1/$family/$task.pt",
            "--task", $task,
            "--holdout-market", "ERCOT",
            "--output", "data/results/global_implicit_qp_raw_predictions_v1/$family`_$task`_inner_seed0.npz",
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
}

$seedFamilies = @(
    "implicitqp_decoder_balanced_seed0",
    "implicitqp_decoder_balanced_seed1",
    "implicitqp_decoder_balanced_seed2"
)
$ensembleFamily = "implicitqp_decoder_balanced_ensemble"
Invoke-Logged "prepare_ensemble" $ensembleFamily @(
    "scripts/decision_aware/prepare_fixed_family_ensemble.py",
    "--prediction-dir", "data/results/global_implicit_qp_profit_predictions_v1",
    "--members", ($seedFamilies -join ","),
    "--output-family", $ensembleFamily,
    "--phase", "inner"
)

$families = @($seedFamilies) + @($ensembleFamily, "xgbpooled")
$triplets = @()
foreach ($family in $families) {
    $triplets += "da_$family`__rt_at_da_$family`__rt_$family"
}
Invoke-Logged "settlement" "fixed_multiseed" @(
    "scripts/decision_aware/evaluate_global_continuous_mpc.py",
    "--config", "configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml",
    "--prediction-dir", "data/results/global_implicit_qp_profit_predictions_v1",
    "--families", ($families -join ","),
    "--phase", "inner",
    "--triplets", ($triplets -join ","),
    "--workers", "$Workers",
    "--executor", "process",
    "--status", "2023_inner_fixed_multiseed_validation_not_final",
    "--output", "data/results/global_implicit_qp_multiseed_continuous_mpc_v1.json"
)

Invoke-Logged "summarize" "fixed_multiseed" @(
    "scripts/decision_aware/summarize_implicit_qp_multiseed.py",
    "--input", "data/results/global_implicit_qp_multiseed_continuous_mpc_v1.json",
    "--seed-families", ($seedFamilies -join ","),
    "--ensemble-family", $ensembleFamily,
    "--baseline-family", "xgbpooled",
    "--output-dir", "data/results/global_implicit_qp_multiseed_summary_v1"
)
Write-Status "runner" "all" "complete"
