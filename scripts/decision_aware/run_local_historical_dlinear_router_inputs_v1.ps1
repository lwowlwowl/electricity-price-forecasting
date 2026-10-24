param(
    [string]$PythonBin = ".\.venv\Scripts\python.exe",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$resultRoot = "data/results/historical_dlinear_router_inputs_v1"
$checkpointRoot = "data/checkpoints/historical_dlinear_router_inputs_v1"
$profitRoot = "data/results/historical_transformer_dlinear_profit_v1"
$statusPath = Join-Path $resultRoot "status.tsv"
New-Item -ItemType Directory -Force -Path $resultRoot, $checkpointRoot, $profitRoot | Out-Null

function Record-Status([string]$Name, [string]$State) {
    $stamp = (Get-Date).ToString("o")
    Add-Content -Path $statusPath -Value "$stamp`t$Name`t$State"
}

function Config-For([string]$Task) {
    if ($Task -eq "da") { return "configs/decision_aware/allmarket_final_da_mae_v1.yaml" }
    if ($Task -eq "rt_at_da") { return "configs/decision_aware/allmarket_final_rt_at_da_pseudohuber_v1.yaml" }
    return "configs/decision_aware/allmarket_final_rt_mae_v1.yaml"
}

foreach ($phase in @("inner", "outer")) {
    foreach ($task in @("da", "rt_at_da")) {
        $config = Config-For $task
        $checkpoint = "$checkpointRoot/${task}_${phase}_seed0.pt"
        $prediction = "$resultRoot/ercot_${task}_${phase}.npz"
        if ((Test-Path $checkpoint) -and -not $Force) {
            Record-Status "train_${phase}_${task}" "already_complete"
        } else {
            Record-Status "train_${phase}_${task}" "starting"
            $trainArgs = @(
                "scripts/decision_aware/run_allmarket_historical_dlinear.py",
                "--mode", "train", "--phase", $phase, "--config", $config,
                "--task", $task, "--checkpoint", $checkpoint, "--device", $Device
            )
            if ($Force) { $trainArgs += "--force" }
            & $PythonBin @trainArgs
            if ($LASTEXITCODE -ne 0) { throw "DLinear training failed: $phase/$task" }
            Record-Status "train_${phase}_${task}" "complete"
        }
        if ((Test-Path $prediction) -and -not $Force) {
            Record-Status "evaluate_${phase}_${task}" "already_complete"
        } else {
            Record-Status "evaluate_${phase}_${task}" "starting"
            $evalArgs = @(
                "scripts/decision_aware/run_allmarket_historical_dlinear.py",
                "--mode", "evaluate", "--phase", $phase, "--config", $config,
                "--task", $task, "--checkpoint", $checkpoint, "--market", "ERCOT",
                "--output", $prediction, "--device", $Device
            )
            if ($Force) { $evalArgs += "--force" }
            & $PythonBin @evalArgs
            if ($LASTEXITCODE -ne 0) { throw "DLinear evaluation failed: $phase/$task" }
            Record-Status "evaluate_${phase}_${task}" "complete"
        }
    }

    $sourceRoot = if ($phase -eq "inner") {
        "data/results/universal_lomo_ercot_profit_predictions_v1"
    } else {
        "data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1"
    }
    & $PythonBin scripts/decision_aware/prepare_historical_baseline_profit_archives.py `
        --family dlinear --phase $phase `
        --source-da "$resultRoot/ercot_da_${phase}.npz" `
        --source-rt-at-da "$resultRoot/ercot_rt_at_da_${phase}.npz" `
        --source-rt "$sourceRoot/universal_rt_${phase}_seed0.npz" `
        --output-dir $profitRoot
    if ($LASTEXITCODE -ne 0) { throw "profit archive preparation failed: $phase" }

    foreach ($task in @("da", "rt_at_da", "rt")) {
        Copy-Item -Force "$sourceRoot/universal_${task}_${phase}_seed0.npz" `
            "$profitRoot/universal_${task}_${phase}_seed0.npz"
    }

    $selectionFlag = if ($phase -eq "inner") { "--selection-allowed" } else { "--no-selection-allowed" }
    & $PythonBin scripts/decision_aware/evaluate_global_continuous_mpc.py `
        --config configs/decision_aware/allmarket_final_da_mae_v1.yaml `
        --prediction-dir $profitRoot --families universal,dlinear --phase $phase `
        --triplets "da_universal__rt_at_da_universal__rt_universal,da_universal__rt_at_da_dlinear__rt_universal,da_dlinear__rt_at_da_universal__rt_universal,da_dlinear__rt_at_da_dlinear__rt_universal" `
        $selectionFlag --workers 4 --executor process `
        --status "historical_transformer_dlinear_${phase}" `
        --output "$resultRoot/cross_task_${phase}_mpc_v1.json"
    if ($LASTEXITCODE -ne 0) { throw "cross-task settlement failed: $phase" }
    Record-Status "settlement_${phase}" "complete"
}

Record-Status "pipeline" "complete"
