param(
    [switch]$SkipTraining,
    [int]$Workers = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Checkpoint = "data/checkpoints/economic_spread_factorization_lomo_ercot_v1/best.pt"
$PredictionDir = "data/results/economic_spread_factorization_lomo_ercot_v1"
$Family = "economic_spread_factorization"

Push-Location $RepoRoot
try {
    if (-not $SkipTraining) {
        & $Python scripts/decision_aware/train_economic_spread_factorization.py `
            --scope head `
            --epochs 6 `
            --output $Checkpoint
        if ($LASTEXITCODE -ne 0) { throw "economic spread training failed" }
    }

    & $Python scripts/decision_aware/evaluate_economic_spread_factorization.py `
        --checkpoint $Checkpoint `
        --holdout-market ERCOT `
        --stage inner `
        --output-dir $PredictionDir `
        --rolling-rt-archive "data/results/universal_lomo_ercot_profit_predictions_v1/universal_rt_inner_seed0.npz"
    if ($LASTEXITCODE -ne 0) { throw "inner prediction evaluation failed" }

    & $Python scripts/decision_aware/evaluate_global_continuous_mpc.py `
        --config configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml `
        --prediction-dir $PredictionDir `
        --families $Family `
        --phase inner `
        --workers $Workers `
        --status "2023_ercot_fitted_economic_spread_factorization" `
        --output "data/results/economic_spread_factorization_inner_mpc_v1.json"
    if ($LASTEXITCODE -ne 0) { throw "inner LP/MPC evaluation failed" }

    & $Python scripts/decision_aware/evaluate_economic_spread_factorization.py `
        --checkpoint $Checkpoint `
        --holdout-market ERCOT `
        --stage outer_frozen `
        --output-dir $PredictionDir `
        --rolling-rt-archive "data/results/universal_lomo_ercot_outer_frozen_profit_predictions_v1/universal_rt_outer_seed0.npz"
    if ($LASTEXITCODE -ne 0) { throw "outer prediction evaluation failed" }

    & $Python scripts/decision_aware/evaluate_global_continuous_mpc.py `
        --config configs/decision_aware/universal_lomo_holdout_ercot_mae_v1.yaml `
        --prediction-dir $PredictionDir `
        --families $Family `
        --phase outer `
        --no-selection-allowed `
        --workers $Workers `
        --status "2024_frozen_confirmation_economic_spread_factorization" `
        --output "data/results/economic_spread_factorization_outer_mpc_v1.json"
    if ($LASTEXITCODE -ne 0) { throw "outer LP/MPC evaluation failed" }
}
finally {
    Pop-Location
}

