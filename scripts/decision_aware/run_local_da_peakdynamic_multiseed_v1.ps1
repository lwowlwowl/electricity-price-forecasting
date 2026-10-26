$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root 'src'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$ResultDir = Join-Path $Root 'data\results\local_da_peakdynamic_multiseed_v1'
$PredictionDir = 'data/results/universal_lomo_ercot_da_peakdynamic_predictions_v1'
$OuterOriginalDir = 'data/results/universal_lomo_multiseed_outer_v1'
$EnsembleDir = 'data/results/universal_lomo_ercot_da_peakdynamic_ensemble_v1'
$InnerProfitDir = 'data/results/universal_lomo_ercot_da_peakdynamic_multiseed_profit_inner_v1'
$OuterProfitDir = 'data/results/universal_lomo_ercot_da_peakdynamic_multiseed_profit_outer_v1'
$InnerSettlement = 'data/results/universal_lomo_ercot_da_peakdynamic_multiseed_inner_mpc_v1.json'
$OuterSettlement = 'data/results/universal_lomo_ercot_da_peakdynamic_multiseed_outer_mpc_v1.json'
$SummaryDir = 'data/results/universal_lomo_ercot_da_peakdynamic_multiseed_summary_v1'
$AuditDir = 'data/results/universal_lomo_ercot_da_peakdynamic_action_audit_v1'

New-Item -ItemType Directory -Force -Path $ResultDir, $OuterOriginalDir, $EnsembleDir, $InnerProfitDir, $OuterProfitDir, $SummaryDir, $AuditDir | Out-Null
$Status = Join-Path $ResultDir 'status.tsv'
$Log = Join-Path $ResultDir 'runner.log'

function Write-Status([string]$Step, [string]$State) {
    $line = "$(Get-Date -Format o)`t$Step`t$State"
    $line | Tee-Object -FilePath $Status -Append
}

function Run-Python([string[]]$Arguments) {
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) { throw "Python命令失败: $($Arguments -join ' ')" }
}

function Assert-Hash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "缺少冻结文件: $Path" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) { throw "冻结文件哈希变化: $Path ($actual)" }
}

function Original-Path([int]$Seed, [string]$Task, [string]$Phase) {
    if ($Seed -eq 0) {
        $suffix = if ($Phase -eq 'outer') { '_outer_frozen' } else { '' }
        return "data/results/universal_lomo_${Task}${suffix}_v1/ercot_${Task}_v1.npz"
    }
    if ($Phase -eq 'inner') {
        return "data/results/universal_lomo_multiseed_v1/ercot_${Task}_inner_seed${Seed}.npz"
    }
    return "$OuterOriginalDir/ercot_${Task}_outer_seed${Seed}.npz"
}

function Candidate-Path([int]$Seed, [string]$Phase) {
    return "$PredictionDir/peak_dynamic_da_${Phase}_seed${Seed}.npz"
}

function Prepare-Family([string]$Phase, [string]$Family, [string]$Da, [string]$RtAtDa, [string]$Rt) {
    $template = if ($Phase -eq 'outer') { 'data/results/global_outer_prediction_templates' } else { 'data/results/global_f0_f1_stable_v2_predictions' }
    $output = if ($Phase -eq 'outer') { $OuterProfitDir } else { $InnerProfitDir }
    Run-Python @(
        'scripts/decision_aware/prepare_universal_lomo_profit_archives.py',
        '--phase', $Phase, '--family', $Family, '--seed', '0',
        '--template-root', $template, '--output-root', $output,
        '--source-da', $Da, '--source-rt-at-da', $RtAtDa, '--source-rt', $Rt
    )
}

try {
    Write-Status 'preflight' 'starting'
    if (-not (Test-Path -LiteralPath $Python)) { throw "缺少项目Python: $Python" }
    Assert-Hash 'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt' '5c3ab8db24d2e6b8cd97745fe207e5ae31171e1359b39400f5f560e08ebb7b0d'
    Assert-Hash 'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed1/best.pt' 'aaf5e227d92bf87366c40cda344f34e1659263e617ed20a1b2bff7962d800b65'
    Assert-Hash 'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed2/best.pt' 'cdce8237aa19f82f1fad03f3f714af85bcfcdcdd4a75dcf3e6a092033b6bcd5b'
    Assert-Hash 'data/checkpoints/universal_lomo_holdout_ercot_da_peakdynamic_v1/da_inner_seed0/selected_peak_dynamic.pt' '995ff0dac54f89c3c5adfdcaccbe1676fdeae37ce38def92887f920a58fc81a3'
    Run-Python @('-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', 'tmp/pytest_peakdynamic_multiseed', 'tests/test_local_peak_dynamics.py', 'tests/test_peak_dynamic_pipeline.py', 'tests/test_peak_dynamic_multiseed.py')
    & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) { throw 'CUDA GPU不可用' }
    $freezeFiles = @(
        'configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_v1.yaml',
        'configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_seed1_v1.yaml',
        'configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_seed2_v1.yaml',
        'docs/progress/5_尾部事件与幅度建模/3_局部峰值幅度后训练.md',
        'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed0/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed1/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed2/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/rt_at_da_inner_seed1/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1/rt_at_da_inner_seed2/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/rt_inner_seed1/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_mae_v1/rt_inner_seed2/best.pt',
        'data/checkpoints/universal_lomo_holdout_ercot_da_peakdynamic_v1/da_inner_seed0/selected_peak_dynamic.pt'
    )
    $freezeFiles | ForEach-Object {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash.ToLowerInvariant()
        "$hash  $($_ -replace '\\','/')"
    } | Set-Content -Encoding utf8 (Join-Path $ResultDir 'preregistered_inputs.sha256')
    Write-Status 'preflight' 'complete'

    foreach ($seed in 1, 2) {
        $config = "configs/decision_aware/universal_lomo_holdout_ercot_da_peakdynamic_seed${seed}_v1.yaml"
        $base = "data/checkpoints/universal_lomo_holdout_ercot_mae_v1/da_inner_seed${seed}/best.pt"
        $checkpointDir = "data/checkpoints/universal_lomo_holdout_ercot_da_peakdynamic_v1/da_inner_seed${seed}"
        $summary = "$checkpointDir/summary.json"
        if (-not (Test-Path -LiteralPath $summary)) {
            Write-Status "train_seed${seed}" 'starting'
            Run-Python @(
                'scripts/decision_aware/train_global_forecaster.py', '--config', $config,
                '--task', 'da', '--phase', 'inner', '--seed', "$seed", '--device', 'cuda',
                '--initialize-from', $base
            )
            Write-Status "train_seed${seed}" 'complete'
        } else { Write-Status "train_seed${seed}" 'already_complete' }

        Write-Status "select_seed${seed}" 'starting'
        Run-Python @(
            'scripts/decision_aware/select_peak_dynamic_checkpoint.py',
            '--checkpoint-dir', $checkpointDir, '--mae-guard-fraction', '0.02'
        )
        Write-Status "select_seed${seed}" 'complete'

        foreach ($phase in 'inner', 'outer') {
            $candidate = Candidate-Path $seed $phase
            if (-not (Test-Path -LiteralPath $candidate)) {
                $extra = if ($phase -eq 'outer') { @('--evaluation-stage', 'outer_frozen') } else { @() }
                Write-Status "predict_peak_seed${seed}_${phase}" 'starting'
                $command = @(
                    'scripts/decision_aware/evaluate_universal_holdout.py',
                    '--checkpoint', "$checkpointDir/selected_peak_dynamic.pt",
                    '--task', 'da', '--holdout-market', 'ERCOT', '--output', $candidate,
                    '--device', 'cuda', '--batch-size', '64'
                ) + $extra
                Run-Python $command
                Write-Status "predict_peak_seed${seed}_${phase}" 'complete'
            }
        }

        foreach ($task in 'da', 'rt_at_da', 'rt') {
            $output = Original-Path $seed $task 'outer'
            if (-not (Test-Path -LiteralPath $output)) {
                $checkpointRoot = if ($task -eq 'rt_at_da') {
                    'data/checkpoints/universal_lomo_holdout_ercot_rt_at_da_v1'
                } else {
                    'data/checkpoints/universal_lomo_holdout_ercot_mae_v1'
                }
                $checkpoint = "$checkpointRoot/${task}_inner_seed${seed}/best.pt"
                Write-Status "predict_original_seed${seed}_${task}_outer" 'starting'
                Run-Python @(
                    'scripts/decision_aware/evaluate_universal_holdout.py',
                    '--checkpoint', $checkpoint, '--task', $task,
                    '--holdout-market', 'ERCOT', '--output', $output,
                    '--device', 'cuda', '--batch-size', '64',
                    '--evaluation-stage', 'outer_frozen'
                )
                Write-Status "predict_original_seed${seed}_${task}_outer" 'complete'
            }
        }
    }

    Write-Status 'median_ensemble' 'starting'
    foreach ($phase in 'inner', 'outer') {
        foreach ($task in 'da', 'rt_at_da', 'rt') {
            $sources = 0,1,2 | ForEach-Object { Original-Path $_ $task $phase }
            $output = "$EnsembleDir/original_${task}_${phase}_median.npz"
            Run-Python @(
                'scripts/decision_aware/build_prediction_median_ensemble.py',
                '--source', $sources[0], '--source', $sources[1], '--source', $sources[2],
                '--output', $output, '--label', "original_${task}_${phase}_median"
            )
        }
        $peakSources = 0,1,2 | ForEach-Object { Candidate-Path $_ $phase }
        Run-Python @(
            'scripts/decision_aware/build_prediction_median_ensemble.py',
            '--source', $peakSources[0], '--source', $peakSources[1], '--source', $peakSources[2],
            '--output', "$EnsembleDir/peak_da_${phase}_median.npz",
            '--label', "peak_dynamic_da_${phase}_median"
        )
    }
    Write-Status 'median_ensemble' 'complete'

    foreach ($phase in 'inner', 'outer') {
        foreach ($seed in 0, 1, 2) {
            $da = Original-Path $seed 'da' $phase
            $rtda = Original-Path $seed 'rt_at_da' $phase
            $rt = Original-Path $seed 'rt' $phase
            Prepare-Family $phase "original_seed${seed}" $da $rtda $rt
            Prepare-Family $phase "peak_dynamic_seed${seed}" (Candidate-Path $seed $phase) $rtda $rt
        }
        $medianRtda = "$EnsembleDir/original_rt_at_da_${phase}_median.npz"
        $medianRt = "$EnsembleDir/original_rt_${phase}_median.npz"
        Prepare-Family $phase 'original_median' "$EnsembleDir/original_da_${phase}_median.npz" $medianRtda $medianRt
        Prepare-Family $phase 'peak_dynamic_median' "$EnsembleDir/peak_da_${phase}_median.npz" $medianRtda $medianRt
    }

    $families = 'original_seed0,peak_dynamic_seed0,original_seed1,peak_dynamic_seed1,original_seed2,peak_dynamic_seed2,original_median,peak_dynamic_median'
    $triplets = @(
        'da_original_seed0__rt_at_da_original_seed0__rt_original_seed0',
        'da_peak_dynamic_seed0__rt_at_da_original_seed0__rt_original_seed0',
        'da_original_seed1__rt_at_da_original_seed1__rt_original_seed1',
        'da_peak_dynamic_seed1__rt_at_da_original_seed1__rt_original_seed1',
        'da_original_seed2__rt_at_da_original_seed2__rt_original_seed2',
        'da_peak_dynamic_seed2__rt_at_da_original_seed2__rt_original_seed2',
        'da_original_median__rt_at_da_original_median__rt_original_median',
        'da_peak_dynamic_median__rt_at_da_original_median__rt_original_median'
    ) -join ','
    foreach ($phase in 'inner', 'outer') {
        $profitDir = if ($phase -eq 'outer') { $OuterProfitDir } else { $InnerProfitDir }
        $settlement = if ($phase -eq 'outer') { $OuterSettlement } else { $InnerSettlement }
        $selection = if ($phase -eq 'outer') { @('--no-selection-allowed') } else { @() }
        Write-Status "settlement_${phase}" 'starting'
        $command = @(
            'scripts/decision_aware/evaluate_global_continuous_mpc.py',
            '--config', 'configs/decision_aware/global_ercot_huber_v1.yaml',
            '--prediction-dir', $profitDir, '--families', $families,
            '--phase', $phase, '--triplets', $triplets,
            '--include-baselines', 'xgboost_robust', '--workers', '4',
            '--executor', 'process', '--status', "da_peakdynamic_multiseed_${phase}_v1",
            '--output', $settlement
        ) + $selection
        Run-Python $command
        Write-Status "settlement_${phase}" 'complete'
    }

    Write-Status 'summary' 'starting'
    Run-Python @(
        'scripts/decision_aware/summarize_peak_dynamic_multiseed.py',
        '--inner-settlement', $InnerSettlement, '--outer-settlement', $OuterSettlement,
        '--output-dir', $SummaryDir
    )
    Write-Status 'summary' 'complete'

    foreach ($phase in 'inner', 'outer') {
        $profitDir = if ($phase -eq 'outer') { $OuterProfitDir } else { $InnerProfitDir }
        foreach ($token in 'seed0', 'median') {
            Write-Status "audit_${phase}_${token}" 'starting'
            Run-Python @(
                'scripts/decision_aware/audit_peak_dynamic_actions.py',
                '--prediction-dir', $profitDir, '--phase', $phase,
                '--current-family', "original_${token}",
                '--candidate-family', "peak_dynamic_${token}",
                '--workers', '4',
                '--output', "$AuditDir/${phase}_${token}.json"
            )
            Write-Status "audit_${phase}_${token}" 'complete'
        }
    }

    Get-ChildItem -Recurse -File $ResultDir, $PredictionDir, $OuterOriginalDir, $EnsembleDir, $InnerProfitDir, $OuterProfitDir, $SummaryDir, $AuditDir |
        Sort-Object FullName | ForEach-Object {
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            $relative = [IO.Path]::GetRelativePath($Root, $_.FullName).Replace('\','/')
            "$hash  $relative"
        } | Set-Content -Encoding utf8 (Join-Path $ResultDir 'completed_outputs.sha256')
    Write-Status 'pipeline' 'complete'
}
catch {
    Write-Status 'pipeline' "failed: $($_.Exception.Message)"
    throw
}
