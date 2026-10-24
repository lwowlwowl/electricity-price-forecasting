$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
$python = [System.IO.Path]::GetFullPath($python)
$predictionRoot = 'data/results/tail_expert_router_cross_market_profit_inputs_v1'
$resultRoot = 'data/results/tail_expert_router_cross_market_value_v1'
$families = 'transformer,dlinear,static_hybrid,frozen_router'
$triplets = @(
    'da_transformer__rt_at_da_transformer__rt_transformer',
    'da_dlinear__rt_at_da_dlinear__rt_dlinear',
    'da_static_hybrid__rt_at_da_static_hybrid__rt_static_hybrid',
    'da_frozen_router__rt_at_da_frozen_router__rt_frozen_router'
) -join ','

& $python scripts/decision_aware/audit_frozen_router_cross_market_2025.py
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

foreach ($market in @('caiso', 'ercot', 'nyiso', 'pjm')) {
    & $python scripts/decision_aware/evaluate_global_continuous_mpc.py `
        --config configs/decision_aware/allmarket_final_da_mae_v1.yaml `
        --prediction-dir "$predictionRoot/$market" `
        --families $families `
        --phase final `
        --triplets $triplets `
        --no-selection-allowed `
        --workers 4 `
        --executor process `
        --status standardized_cross_market_value_not_realized_profit `
        --output "$resultRoot/$market.json"
    if ($LASTEXITCODE -ne 0) { throw "标准化经济价值评价失败: $market" }
}
