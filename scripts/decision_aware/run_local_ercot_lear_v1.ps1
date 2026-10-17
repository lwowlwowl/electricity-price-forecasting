param([int]$Workers = 6)

$ErrorActionPreference = 'Stop'
# 节点已由-Workers并行；禁止每个节点再让BLAS占满全部CPU。
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\decision_aware\run_global_baselines.py'
$config = Join-Path $projectRoot 'configs\decision_aware\global_ercot_huber_v1.yaml'

foreach ($task in @('da', 'rt_at_da')) {
    Write-Output ("[{0}] START market=ERCOT task={1} models=lear" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $task)
    & $pythonExe $runner `
        --config $config `
        --market ERCOT `
        --task $task `
        --phase inner `
        --models lear `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) {
        throw "LEAR失败：market=ERCOT, task=$task, exit=$LASTEXITCODE"
    }
    Write-Output ("[{0}] DONE market=ERCOT task={1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $task)
}

Write-Output ("[{0}] ALL DONE market=ERCOT models=lear" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
