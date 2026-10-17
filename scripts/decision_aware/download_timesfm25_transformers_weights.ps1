param(
    [string]$OutputDirectory = "external/timesfm25_transformers_snapshot",
    [int64]$FileSize = 925187448,
    [int64]$ChunkSize = 16777216,
    [int]$ThrottleLimit = 8
)

$ErrorActionPreference = "Stop"
$url = "https://huggingface.co/google/timesfm-2.5-200m-transformers/resolve/main/model.safetensors"
$expectedSha256 = "b53f6d52114e2ad786890f3c4637ce05f580b7800d6e24401f88b398b76035ef"
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$partRoot = Join-Path $outputRoot "parts"
[System.IO.Directory]::CreateDirectory($partRoot) | Out-Null

$chunks = @()
for ($start = [int64]0; $start -lt $FileSize; $start += $ChunkSize) {
    $end = [Math]::Min($FileSize - 1, $start + $ChunkSize - 1)
    $index = [int]($start / $ChunkSize)
    $chunks += [pscustomobject]@{
        Index = $index
        Start = $start
        End = $end
        Length = $end - $start + 1
        Path = Join-Path $partRoot ("part_{0:D3}.bin" -f $index)
    }
}

$chunks | ForEach-Object -Parallel {
    $chunk = $_
    if ((Test-Path -LiteralPath $chunk.Path) -and
        ((Get-Item -LiteralPath $chunk.Path).Length -eq $chunk.Length)) {
        return
    }
    & curl.exe --silent --show-error --fail --http1.1 --location `
        --retry 8 --retry-all-errors --retry-delay 2 `
        --range "$($chunk.Start)-$($chunk.End)" `
        --output $chunk.Path `
        $using:url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for chunk $($chunk.Index)"
    }
    $actualLength = (Get-Item -LiteralPath $chunk.Path).Length
    if ($actualLength -ne $chunk.Length) {
        throw "chunk $($chunk.Index) length $actualLength != $($chunk.Length)"
    }
} -ThrottleLimit $ThrottleLimit

$missing = @($chunks | Where-Object {
    -not (Test-Path -LiteralPath $_.Path) -or
    (Get-Item -LiteralPath $_.Path).Length -ne $_.Length
})
if ($missing.Count -gt 0) {
    throw "incomplete chunks: $($missing.Index -join ',')"
}

$output = Join-Path $outputRoot "model.safetensors"
$destination = [System.IO.File]::Open(
    $output,
    [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
try {
    foreach ($chunk in $chunks | Sort-Object Index) {
        $source = [System.IO.File]::OpenRead($chunk.Path)
        try {
            $source.CopyTo($destination)
        }
        finally {
            $source.Dispose()
        }
    }
}
finally {
    $destination.Dispose()
}

$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $output).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "model SHA-256 mismatch: $actualSha256 != $expectedSha256"
}
Write-Output "output=$output"
Write-Output "bytes=$((Get-Item -LiteralPath $output).Length)"
Write-Output "sha256=$actualSha256"
