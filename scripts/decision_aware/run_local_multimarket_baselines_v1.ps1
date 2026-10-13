param(
    [ValidateSet('seasonal', 'xgboost', 'robust', 'lear', 'all')]
    [string]$Mode = 'seasonal',
    [int]$Workers = 6
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\decision_aware\run_global_baselines.py'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "项目虚拟环境不存在：$pythonExe"
}

$markets = @(
    @{ Name = 'CAISO'; Config = 'configs\decision_aware\global_caiso_prediction_contract_v1.yaml' },
    @{ Name = 'NYISO'; Config = 'configs\decision_aware\global_nyiso_prediction_contract_v1.yaml' },
    @{ Name = 'PJM'; Config = 'configs\decision_aware\global_pjm_prediction_contract_v1.yaml' }
)
$tasks = @('da', 'rt_at_da', 'rt')

foreach ($task in $tasks) {
    foreach ($market in $markets) {
        if ($Mode -eq 'seasonal') {
            $models = 'seasonal'
        }
        elseif ($Mode -eq 'xgboost') {
            $models = 'xgboost,xgboost_absolute,xgboost_pseudohuber'
        }
        elseif ($Mode -eq 'robust') {
            # 只运行ERCOT内部训练期已经冻结的任务专属稳健目标；不在新市场重新挑损失。
            $models = if ($task -eq 'rt_at_da') {
                'xgboost_pseudohuber'
            }
            else {
                'xgboost_absolute'
            }
        }
        elseif ($Mode -eq 'lear') {
            if ($task -eq 'rt') {
                continue
            }
            $models = 'lear'
        }
        else {
            $models = if ($task -eq 'rt') {
                'seasonal,xgboost,xgboost_absolute,xgboost_pseudohuber'
            }
            else {
                'seasonal,lear,xgboost,xgboost_absolute,xgboost_pseudohuber'
            }
        }

        $configPath = Join-Path $projectRoot $market.Config
        Write-Output ("[{0}] START market={1} task={2} models={3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $market.Name, $task, $models)
        & $pythonExe $runner `
            --config $configPath `
            --market $market.Name `
            --task $task `
            --phase inner `
            --models $models `
            --workers $Workers
        if ($LASTEXITCODE -ne 0) {
            throw "基线失败：market=$($market.Name), task=$task, exit=$LASTEXITCODE"
        }
        Write-Output ("[{0}] DONE  market={1} task={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $market.Name, $task)
    }
}

Write-Output ("[{0}] ALL DONE mode={1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Mode)
