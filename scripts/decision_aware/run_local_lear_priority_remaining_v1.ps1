param([int]$Workers = 6)

$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\decision_aware\run_global_baselines.py'

# 先完成节点较少的两个RT-at-DA市场，使六小时窗口内得到更多完整可报告行；
# 最后续跑PJM两个任务。所有已完成节点均由NPZ合同缓存，不重复拟合。
$jobs = @(
    @{ Market = 'CAISO'; Task = 'rt_at_da'; Config = 'configs\decision_aware\global_caiso_prediction_contract_v1.yaml' },
    @{ Market = 'NYISO'; Task = 'rt_at_da'; Config = 'configs\decision_aware\global_nyiso_prediction_contract_v1.yaml' },
    @{ Market = 'PJM'; Task = 'da'; Config = 'configs\decision_aware\global_pjm_prediction_contract_v1.yaml' },
    @{ Market = 'PJM'; Task = 'rt_at_da'; Config = 'configs\decision_aware\global_pjm_prediction_contract_v1.yaml' }
)

foreach ($job in $jobs) {
    $config = Join-Path $projectRoot $job.Config
    Write-Output ("[{0}] START market={1} task={2} models=lear" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $job.Market, $job.Task)
    & $pythonExe $runner `
        --config $config `
        --market $job.Market `
        --task $job.Task `
        --phase inner `
        --models lear `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) {
        throw "LEAR失败：market=$($job.Market), task=$($job.Task), exit=$LASTEXITCODE"
    }
    Write-Output ("[{0}] DONE market={1} task={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $job.Market, $job.Task)
}

Write-Output ("[{0}] ALL PRIORITY LEAR JOBS DONE" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
