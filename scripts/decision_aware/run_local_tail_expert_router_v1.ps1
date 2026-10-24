param(
    [string]$PythonBin = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$routerRoot = "data/results/tail_expert_router_v1"
New-Item -ItemType Directory -Force -Path $routerRoot | Out-Null

& $PythonBin scripts/decision_aware/build_tail_expert_router_v1.py --mode build_inner
if ($LASTEXITCODE -ne 0) { throw "inner router candidate construction failed" }

$candidateFamilies = @(
    "universal", "hybrid",
    "router_logistic_50", "router_logistic_60", "router_logistic_70", "router_logistic_80",
    "router_tree_50", "router_tree_60", "router_tree_70", "router_tree_80"
)
$innerTriplets = ($candidateFamilies | ForEach-Object { "da_${_}__rt_at_da_${_}__rt_${_}" }) -join ","
& $PythonBin scripts/decision_aware/evaluate_global_continuous_mpc.py `
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml `
    --prediction-dir $routerRoot --families ($candidateFamilies -join ",") --phase inner `
    --triplets $innerTriplets --selection-allowed --workers 4 --executor process `
    --status tail_expert_router_inner_blocked_oof_selection `
    --output "$routerRoot/inner_router_mpc_v1.json"
if ($LASTEXITCODE -ne 0) { throw "inner router settlement failed" }

& $PythonBin scripts/decision_aware/build_tail_expert_router_v1.py --mode freeze_outer
if ($LASTEXITCODE -ne 0) { throw "outer router freeze failed" }

$outerFamilies = @("universal", "hybrid", "router_final")
$outerTriplets = ($outerFamilies | ForEach-Object { "da_${_}__rt_at_da_${_}__rt_${_}" }) -join ","
& $PythonBin scripts/decision_aware/evaluate_global_continuous_mpc.py `
    --config configs/decision_aware/allmarket_final_da_mae_v1.yaml `
    --prediction-dir $routerRoot --families ($outerFamilies -join ",") --phase outer `
    --triplets $outerTriplets --no-selection-allowed --workers 4 --executor process `
    --status tail_expert_router_outer_frozen_confirmation `
    --output "$routerRoot/outer_router_mpc_v1.json"
if ($LASTEXITCODE -ne 0) { throw "outer router settlement failed" }

Write-Output "tail expert router inner selection and outer confirmation complete"
