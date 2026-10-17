param(
    [ValidateRange(1, 16)]
    [int]$Workers = 6
)

$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUNBUFFERED = '1'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\decision_aware\run_global_baselines.py'
$resultDir = Join-Path $projectRoot 'data\results\local_allmarket_final_xgboost_v1'
$statusFile = Join-Path $resultDir 'status.tsv'
$auditPath = Join-Path $projectRoot 'data\results\allmarket_final_baselines_v1\artifact_audit.json'
$openedMarker = Join-Path $projectRoot 'data\results\allmarket_final_2025_preregistration_v1\LOCKED_2025_OPENED.json'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

function Write-Status([string]$Stage, [string]$State) {
    $line = "{0}`t{1}`t{2}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'), $Stage, $State
    Add-Content -LiteralPath $statusFile -Value $line -Encoding utf8
    Write-Output $line
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "项目虚拟环境不存在：$pythonExe"
}
if (Test-Path -LiteralPath $openedMarker) {
    throw '2025已经打开，禁止补训最终基线。'
}

Push-Location $projectRoot
try {
    Write-Status 'preflight' 'starting'
    & $pythonExe scripts/decision_aware/freeze_allmarket_2025_protocol.py --verify
    if ($LASTEXITCODE -ne 0) { throw '主模型冻结协议校验失败。' }
    & $pythonExe scripts/decision_aware/freeze_allmarket_2025_baseline_protocol.py --verify
    if ($LASTEXITCODE -ne 0) { throw '最终基线冻结协议校验失败。' }
    Write-Status 'preflight' 'complete'

    $jobs = @(
        @{ Market = 'CAISO'; Config = 'configs\decision_aware\global_caiso_prediction_contract_v1.yaml' },
        @{ Market = 'ERCOT'; Config = $null },
        @{ Market = 'NYISO'; Config = 'configs\decision_aware\global_nyiso_prediction_contract_v1.yaml' },
        @{ Market = 'PJM'; Config = 'configs\decision_aware\global_pjm_prediction_contract_v1.yaml' }
    )
    $tasks = @('da', 'rt_at_da', 'rt')
    foreach ($marketJob in $jobs) {
        foreach ($task in $tasks) {
            $market = $marketJob.Market
            $config = $marketJob.Config
            if ($market -eq 'ERCOT') {
                $config = switch ($task) {
                    'da' { 'configs\decision_aware\global_ercot_taskloss_da_mae_v1.yaml' }
                    'rt_at_da' { 'configs\decision_aware\global_ercot_taskloss_rt_at_da_pseudohuber_v1.yaml' }
                    'rt' { 'configs\decision_aware\global_ercot_taskloss_rt_mae_v1.yaml' }
                }
            }
            $model = 'xgboost_absolute'
            $stage = "local_xgboost_${market}_${task}"
            Write-Status $stage 'starting'
            & $pythonExe $runner `
                --config $config `
                --market $market `
                --task $task `
                --phase final `
                --models $model `
                --seed 0 `
                --workers $Workers
            if ($LASTEXITCODE -ne 0) {
                throw "最终逐节点XGBoost失败：market=$market task=$task exit=$LASTEXITCODE"
            }
            Write-Status $stage 'complete'
        }
    }

    & $pythonExe scripts/decision_aware/audit_allmarket_final_baselines.py
    if ($LASTEXITCODE -ne 0) { throw '最终基线产物审计脚本失败。' }
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    $localMissing = @($audit.missing | Where-Object { $_ -like '*data/checkpoints/global_baselines_v1/*' })
    if ($localMissing.Count -ne 0) {
        throw "逐节点XGBoost仍缺$($localMissing.Count)项。"
    }
    Write-Status 'all_local_xgboost' 'complete'
}
catch {
    Write-Status 'pipeline' 'failed'
    throw
}
finally {
    Pop-Location
}
