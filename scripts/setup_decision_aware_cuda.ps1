$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runtimeRoot = Join-Path $repoRoot "runtime_cache\setup_cache"

# Keep temporary files and caches inside the project instead of the system drive.
$env:TEMP = Join-Path $runtimeRoot "tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $runtimeRoot "pip-cache"
$env:PYTHONPYCACHEPREFIX = Join-Path $runtimeRoot "pycache"
$env:CUDA_CACHE_PATH = Join-Path $runtimeRoot "cuda-cache"
$env:TORCH_HOME = Join-Path $runtimeRoot "torch-home"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

@(
    $env:TEMP,
    $env:PIP_CACHE_DIR,
    $env:PYTHONPYCACHEPREFIX,
    $env:CUDA_CACHE_PATH,
    $env:TORCH_HOME
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

if (-not (Test-Path -LiteralPath $python)) {
    $basePython = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Project .venv not found. Creating it with $basePython ..."
    & $basePython -m venv (Join-Path $repoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw ".venv creation failed (exit code $LASTEXITCODE)" }
}

Write-Host "Project:  $repoRoot"
Write-Host "Python:   $python"
Write-Host "Cache:    $runtimeRoot"
Write-Host ""
Write-Host "[1/2] Installing requirements-windows-cuda.txt (PyTorch downloads are about 2.6 GB)..."
& $python -m pip install `
    -r (Join-Path $repoRoot "requirements-windows-cuda.txt") `
    --progress-bar on
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit code $LASTEXITCODE)" }

Write-Host ""
Write-Host "[2/2] Verifying NVIDIA GPU, CUDA autocast, and GradScaler backward pass..."
$cudaSmoke = @'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check the NVIDIA driver and PyTorch installation.")

device = torch.device("cuda")
model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 1),
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(256, 32, device=device)
y = torch.randn(256, 1, device=device)

optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast(device_type="cuda"):
    loss = torch.nn.functional.mse_loss(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
torch.cuda.synchronize()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GradScaler enabled: {scaler.is_enabled()}")
print(f"Smoke loss: {loss.item():.6f}")
print("CUDA training smoke test passed.")
'@

& $python -c $cudaSmoke
if ($LASTEXITCODE -ne 0) { throw "CUDA training smoke test failed (exit code $LASTEXITCODE)" }

Write-Host ""
Write-Host "Decision-aware CUDA environment installation completed."
