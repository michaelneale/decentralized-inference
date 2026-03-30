param(
    [Parameter(Mandatory = $true)]
    [string]$MeshLlm,
    [Parameter(Mandatory = $true)]
    [string]$BinDir,
    [Parameter(Mandatory = $true)]
    [string]$ModelPath
)

$ErrorActionPreference = "Stop"

$apiPort = 9337
$consolePort = 3131
$maxWaitSeconds = 180
$stdoutLogPath = Join-Path ([System.IO.Path]::GetTempPath()) "mesh-llm-ci.stdout.log"
$stderrLogPath = Join-Path ([System.IO.Path]::GetTempPath()) "mesh-llm-ci.stderr.log"

function Write-ProcessLogs {
    foreach ($path in @($stdoutLogPath, $stderrLogPath)) {
        if (Test-Path $path) {
            Write-Host "--- $path ---"
            Get-Content $path -Tail 80 | Write-Host
        }
    }
}

Write-Host "=== CI Smoke Test ==="
Write-Host "  mesh-llm:  $MeshLlm"
Write-Host "  bin-dir:   $BinDir"
Write-Host "  model:     $ModelPath"
Write-Host "  api port:  $apiPort"
Write-Host "  os:        Windows"

if (-not (Test-Path $MeshLlm)) {
    throw "Missing mesh-llm binary: $MeshLlm"
}

Get-ChildItem -Path $BinDir -Filter "rpc-server*" -ErrorAction SilentlyContinue | Format-Table -AutoSize | Out-String | Write-Host
Get-ChildItem -Path $BinDir -Filter "llama-server*" -ErrorAction SilentlyContinue | Format-Table -AutoSize | Out-String | Write-Host

$process = $null
try {
    $arguments = @(
        "--model", $ModelPath,
        "--no-draft",
        "--bin-dir", $BinDir,
        "--device", "CPU",
        "--port", "$apiPort",
        "--console", "$consolePort"
    )

    Write-Host "Starting mesh-llm..."
    $process = Start-Process `
        -FilePath $MeshLlm `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdoutLogPath `
        -RedirectStandardError $stderrLogPath `
        -PassThru
    Write-Host "  PID: $($process.Id)"

    Write-Host "Waiting for model to load (up to ${maxWaitSeconds}s)..."
    for ($i = 1; $i -le $maxWaitSeconds; $i++) {
        if ($process.HasExited) {
            Write-Host "❌ mesh-llm exited unexpectedly"
            Write-ProcessLogs
            throw "mesh-llm exited before llama_ready"
        }

        try {
            $status = Invoke-RestMethod -Uri "http://localhost:$consolePort/api/status" -Method Get -TimeoutSec 3
            if ($status.llama_ready -eq $true) {
                Write-Host "✅ Model loaded in ${i}s"
                break
            }
        } catch {
        }

        if ($i -eq $maxWaitSeconds) {
            Write-Host "❌ Model failed to load within ${maxWaitSeconds}s"
            Write-ProcessLogs
            throw "Timed out waiting for llama_ready"
        }

        if (($i % 15) -eq 0) {
            Write-Host "  Still waiting... (${i}s)"
        }
        Start-Sleep -Seconds 1
    }

    Write-Host "Testing /v1/chat/completions..."
    $body = @{
        model = "any"
        messages = @(@{
            role = "user"
            content = "Say hello in exactly 3 words."
        })
        max_tokens = 32
        temperature = 0
    } | ConvertTo-Json -Depth 5

    $response = Invoke-RestMethod `
        -Uri "http://localhost:$apiPort/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30

    $content = $response.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($content)) {
        throw "Empty response from inference"
    }
    Write-Host "✅ Inference response: $content"

    Write-Host "Testing /v1/models..."
    $models = Invoke-RestMethod -Uri "http://localhost:$apiPort/v1/models" -Method Get -TimeoutSec 15
    $modelCount = @($models.data).Count
    if ($modelCount -eq 0) {
        throw "No models returned from /v1/models"
    }
    Write-Host "✅ /v1/models returned $modelCount model(s)"
    Write-Host ""
    Write-Host "=== All smoke tests passed ==="
} finally {
    if ($process) {
        Write-Host "Shutting down mesh-llm (PID $($process.Id))..."
        try {
            taskkill /PID $process.Id /T /F | Out-Null
        } catch {
        }
        Start-Sleep -Seconds 2
    }
}
