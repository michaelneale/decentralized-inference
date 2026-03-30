param(
    [Parameter(Mandatory = $true)]
    [string]$Backend,

    [string]$RocmHipSdkFilename = 'AMD-Software-PRO-Edition-25.Q3-WinSvr2022-For-HIP.exe',

    [string]$InstallerCacheDir = ''
)

$ErrorActionPreference = 'Stop'

function Normalize-RecipeArgument {
    param(
        [AllowEmptyString()]
        [string]$Value,
        [string[]]$KnownNames = @()
    )

    if ($null -eq $Value) {
        return $Value
    }

    $normalized = $Value.Trim()
    if (-not $normalized) {
        return ""
    }

    if ($normalized -match '^(?<name>[A-Za-z_][A-Za-z0-9_-]*)=(?<value>.*)$') {
        $matchedName = $Matches.name
        $isKnownName = $KnownNames.Count -eq 0
        foreach ($knownName in $KnownNames) {
            if ($matchedName.Equals($knownName, [System.StringComparison]::OrdinalIgnoreCase)) {
                $isKnownName = $true
                break
            }
        }

        if ($isKnownName) {
            $normalized = $Matches.value
        }
    }

    if ($normalized.Length -ge 2) {
        $first = $normalized[0]
        $last = $normalized[$normalized.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            $normalized = $normalized.Substring(1, $normalized.Length - 2)
        }
    }

    return $normalized.Trim()
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        $argString = if ($Arguments.Count -gt 0) { " " + ($Arguments -join " ") } else { "" }
        throw "Command failed with exit code ${LASTEXITCODE}: $Command$argString"
    }
}

function Add-GitHubPath([string]$PathEntry) {
    if (-not $PathEntry) {
        return
    }

    if ($env:GITHUB_PATH) {
        $PathEntry | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
    } else {
        $env:PATH = "$PathEntry;$env:PATH"
    }
}

function Set-GitHubEnv([string]$Name, [string]$Value) {
    if (-not $Name) {
        return
    }

    if ($env:GITHUB_ENV) {
        "${Name}=${Value}" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    } else {
        Set-Item -Path "env:$Name" -Value $Value
    }
}

function Resolve-RocmInstallRoot {
    $candidates = @()

    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles 'AMD\ROCm')
        $candidates += (Join-Path $env:ProgramFiles 'AMD\HIP')
    }

    foreach ($base in $candidates) {
        if (-not (Test-Path $base)) {
            continue
        }

        $directHipConfig = Join-Path $base 'lib\cmake\hip\hip-config.cmake'
        if (Test-Path $directHipConfig) {
            return $base
        }

        $versionedRoot = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                Test-Path (Join-Path $_.FullName 'lib\cmake\hip\hip-config.cmake')
            } |
            Sort-Object Name -Descending |
            Select-Object -ExpandProperty FullName -First 1

        if ($versionedRoot) {
            return $versionedRoot
        }
    }

    return $null
}

function Test-ExeHeader([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $false
    }

    $header = Get-Content -Path $Path -AsByteStream -TotalCount 2
    return $header.Length -eq 2 -and $header[0] -eq 0x4D -and $header[1] -eq 0x5A
}

$Backend = Normalize-RecipeArgument $Backend @('backend')
$RocmHipSdkFilename = Normalize-RecipeArgument $RocmHipSdkFilename @('rocm_hip_sdk_filename', 'rocmhipsdkfilename')
$InstallerCacheDir = Normalize-RecipeArgument $InstallerCacheDir @('installer_cache_dir', 'installercachedir', 'cache_dir', 'cachedir')

if (-not $InstallerCacheDir) {
    $InstallerCacheDir = Join-Path $env:RUNNER_TEMP 'mesh-sdk-cache'
} elseif (-not [System.IO.Path]::IsPathRooted($InstallerCacheDir)) {
    $InstallerCacheDir = Join-Path (Get-Location) $InstallerCacheDir
}

New-Item -ItemType Directory -Path $InstallerCacheDir -Force | Out-Null

switch ($Backend.ToLowerInvariant()) {
    'cpu' {
    }
    'cuda' {
        $cudaCacheDir = Join-Path $InstallerCacheDir 'cuda'
        New-Item -ItemType Directory -Path $cudaCacheDir -Force | Out-Null

        Invoke-NativeCommand 'choco' @('install', 'cuda', '-y', '--no-progress', '--cache-location', $cudaCacheDir)

        $cudaRoot = $env:CUDA_PATH
        if (-not $cudaRoot -or -not (Test-Path $cudaRoot)) {
            $cudaRoot = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -ExpandProperty FullName -First 1
        }

        if (-not $cudaRoot) {
            throw 'CUDA toolkit install completed, but CUDA_PATH was not found.'
        }

        Add-GitHubPath "$cudaRoot\bin"
        Set-GitHubEnv 'CUDA_PATH' $cudaRoot
    }
    'rocm' {
        $candidateFilenames = @(
            $RocmHipSdkFilename,
            'AMD-Software-PRO-Edition-24.Q4-WinSvr2022-For-HIP.exe',
            'AMD-Software-PRO-Edition-24.Q3-WinSvr2022-For-HIP.exe'
        ) | Where-Object { $_ } | Select-Object -Unique

        $rocmCacheDir = Join-Path $InstallerCacheDir 'rocm'
        New-Item -ItemType Directory -Path $rocmCacheDir -Force | Out-Null

        $installer = $null
        $downloadErrors = @()

        foreach ($filename in $candidateFilenames) {
            $cachedInstaller = Join-Path $rocmCacheDir $filename
            if (Test-ExeHeader $cachedInstaller) {
                Write-Host "Using cached HIP SDK installer $filename"
                $installer = $cachedInstaller
                break
            }

            $url = "https://download.amd.com/developer/eula/rocm-hub/$filename"
            Write-Host "Trying HIP SDK download: $url"

            try {
                Invoke-WebRequest -Uri $url -OutFile $cachedInstaller
                if (Test-ExeHeader $cachedInstaller) {
                    Write-Host "Using HIP SDK installer $filename"
                    $installer = $cachedInstaller
                    break
                }

                $downloadErrors += "Downloaded $filename, but the payload was not a Windows executable."
            } catch {
                $downloadErrors += "Failed to download ${filename}: $($_.Exception.Message)"
            }

            Remove-Item -Path $cachedInstaller -ErrorAction SilentlyContinue
        }

        if (-not $installer -or -not (Test-ExeHeader $installer)) {
            $details = $downloadErrors -join "`n"
            throw "Unable to download a Windows HIP SDK installer.`n$details"
        }

        $installLog = Join-Path $env:RUNNER_TEMP 'hip-sdk-install.log'
        $process = Start-Process $installer -ArgumentList '-install', '-log', $installLog -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "HIP SDK installer exited with code $($process.ExitCode). See $installLog"
        }

        $rocmRoot = Resolve-RocmInstallRoot

        if (-not $rocmRoot) {
            throw "HIP SDK install completed, but ROCM_PATH was not found. See $installLog"
        }

        Add-GitHubPath "$rocmRoot\bin"
        Add-GitHubPath "$rocmRoot\llvm\bin"
        Set-GitHubEnv 'ROCM_PATH' $rocmRoot
        Set-GitHubEnv 'HIP_PATH' $rocmRoot
    }
    'vulkan' {
        $vulkanCacheDir = Join-Path $InstallerCacheDir 'vulkan'
        New-Item -ItemType Directory -Path $vulkanCacheDir -Force | Out-Null

        Invoke-NativeCommand 'choco' @('install', 'vulkan-sdk', '-y', '--no-progress', '--cache-location', $vulkanCacheDir)

        $sdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if (-not $sdk) {
            throw 'Vulkan SDK install completed, but no C:\VulkanSDK directory was found.'
        }

        Add-GitHubPath "$($sdk.FullName)\Bin"
        Add-GitHubPath "$($sdk.FullName)\Bin32"
        Set-GitHubEnv 'VULKAN_SDK' $sdk.FullName
    }
    default {
        throw "Unsupported Windows backend '$Backend'. Use one of: cpu, cuda, rocm, vulkan."
    }
}
