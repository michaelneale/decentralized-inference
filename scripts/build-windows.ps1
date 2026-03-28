param(
    [string]$Backend = "",
    [string]$CudaArch = "",
    [string]$RocmArch = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$llamaDir = Join-Path $repoRoot "llama.cpp"
$buildDir = Join-Path $llamaDir "build"
$meshUiDir = Join-Path $repoRoot "mesh-llm\ui"

function Add-ToPath {
    param([string]$Directory)

    if (-not $Directory -or -not (Test-Path $Directory)) {
        return
    }

    $pathEntries = @($env:PATH -split [System.IO.Path]::PathSeparator)
    if ($pathEntries -contains $Directory) {
        return
    }

    $env:PATH = "$Directory$([System.IO.Path]::PathSeparator)$env:PATH"
}

function Test-CommandSuccess {
    param(
        [string]$Command,
        [string[]]$Arguments = @()
    )

    try {
        $null = & $Command @Arguments 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-CommandPath {
    param([string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function Import-CmdEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    $output = & cmd.exe /s /c "$CommandLine && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize Windows build environment with command: $CommandLine"
    }

    foreach ($line in $output) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            Set-Item -Path "env:$($Matches.name)" -Value $Matches.value
        }
    }
}

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

function Ensure-MsvcToolchain {
    if ((Resolve-CommandPath "cl") -and (Resolve-CommandPath "link") -and (Resolve-CommandPath "lib")) {
        return
    }

    $programFilesX86 = ${env:ProgramFiles(x86)}
    $vswhereCandidates = @()
    if ($programFilesX86) {
        $vswhereCandidates += (Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($env:ProgramFiles) {
        $vswhereCandidates += (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    $vswhereFromPath = Resolve-CommandPath "vswhere"
    if ($vswhereFromPath) {
        $vswhereCandidates += $vswhereFromPath
    }

    $vcvars64 = $null
    $vswhere = $vswhereCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique -First 1
    if ($vswhere) {
        $installationPathOutput = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
        $installationPath = if ($installationPathOutput) { $installationPathOutput.Trim() } else { "" }
        if ($installationPath) {
            $candidate = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $candidate) {
                $vcvars64 = $candidate
            }
        }
    }

    if (-not $vcvars64) {
        $searchRoots = @($programFilesX86, $env:ProgramFiles) | Where-Object { $_ } | Select-Object -Unique
        foreach ($searchRoot in $searchRoots) {
            $candidate = Get-ChildItem -Path $searchRoot -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -like '*Microsoft Visual Studio*VC\Auxiliary\Build\vcvars64.bat' } |
                Select-Object -First 1
            if ($candidate) {
                $vcvars64 = $candidate.FullName
                break
            }
        }
    }

    if (-not (Test-Path $vcvars64)) {
        throw "Visual Studio Build Tools with vcvars64.bat were not found on this Windows runner."
    }

    Import-CmdEnvironment "`"$vcvars64`" >nul"

    if (-not (Resolve-CommandPath "cl")) {
        throw "MSVC toolchain initialization completed, but cl.exe is still not available in PATH."
    }
}

function Resolve-HipPackageRoot {
    $roots = @()
    if ($env:HIP_PATH) {
        $roots += $env:HIP_PATH
    }
    if ($env:ROCM_PATH) {
        $roots += $env:ROCM_PATH
    }

    $roots = $roots | Where-Object { $_ } | Select-Object -Unique

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $directConfig = Join-Path $root "lib\cmake\hip\hip-config.cmake"
        if (Test-Path $directConfig) {
            return [PSCustomObject]@{
                Root   = $root
                HipDir = Split-Path -Parent $directConfig
            }
        }

        $versionedRoot = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                Test-Path (Join-Path $_.FullName "lib\cmake\hip\hip-config.cmake")
            } |
            Sort-Object Name -Descending |
            Select-Object -First 1

        if ($versionedRoot) {
            $configPath = Join-Path $versionedRoot.FullName "lib\cmake\hip\hip-config.cmake"
            return [PSCustomObject]@{
                Root   = $versionedRoot.FullName
                HipDir = Split-Path -Parent $configPath
            }
        }
    }

    return $null
}

function Resolve-RocmRoot {
    $hipPackage = Resolve-HipPackageRoot
    if ($hipPackage) {
        return $hipPackage.Root
    }
    if ($env:ProgramFiles) {
        foreach ($candidate in @(
            (Join-Path $env:ProgramFiles "AMD\ROCm"),
            (Join-Path $env:ProgramFiles "AMD\HIP")
        )) {
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }
    return $null
}

function Resolve-VulkanSdkRoot {
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) {
        return $env:VULKAN_SDK
    }

    if ($env:ProgramFiles) {
        $sdkBase = Join-Path $env:ProgramFiles "VulkanSDK"
        if (Test-Path $sdkBase) {
            $latest = Get-ChildItem -Path $sdkBase -Directory | Sort-Object Name -Descending | Select-Object -First 1
            if ($latest) {
                return $latest.FullName
            }
        }
    }

    return $null
}

function Resolve-Backend {
    param([string]$Requested)

    if ($Requested) {
        $normalized = $Requested.ToLowerInvariant()
        switch ($normalized) {
            "hip" { return "rocm" }
            "rocm" { return "rocm" }
            default { return $normalized }
        }
    }

    if (Test-CommandSuccess "nvidia-smi" @("--query-gpu=name", "--format=csv,noheader")) {
        return "cuda"
    }

    if (Resolve-RocmRoot) {
        return "rocm"
    }

    if ((Resolve-CommandPath "hipInfo") -or (Resolve-CommandPath "hipconfig")) {
        return "rocm"
    }

    if (Test-CommandSuccess "vulkaninfo" @("--summary")) {
        return "vulkan"
    }

    if (Resolve-VulkanSdkRoot) {
        return "vulkan"
    }

    return "cpu"
}

function Ensure-CudaToolchain {
    if (Resolve-CommandPath "nvcc") {
        return
    }

    $candidates = @()
    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH "bin")
    }
    if ($env:ProgramFiles) {
        $toolkitRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $toolkitRoot) {
            $candidates += Get-ChildItem -Path $toolkitRoot -Directory | Sort-Object Name -Descending | ForEach-Object {
                Join-Path $_.FullName "bin"
            }
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "nvcc.exe")) {
            Add-ToPath $candidate
            return
        }
    }

    throw "nvcc not found. Install the CUDA toolkit and ensure nvcc.exe is in PATH."
}

function Ensure-RocmToolchain {
    $rocmRoot = Resolve-RocmRoot
    $hipPackage = Resolve-HipPackageRoot
    if ($rocmRoot) {
        $binDir = Join-Path $rocmRoot "bin"
        $llvmBinDir = Join-Path $rocmRoot "llvm\bin"
        Add-ToPath $binDir
        Add-ToPath $llvmBinDir
        $env:ROCM_PATH = $rocmRoot
        $env:HIP_PATH = $rocmRoot
        $env:CMAKE_PREFIX_PATH = if ($env:CMAKE_PREFIX_PATH) {
            "$rocmRoot;$env:CMAKE_PREFIX_PATH"
        } else {
            $rocmRoot
        }
        if ($hipPackage) {
            $env:hip_DIR = $hipPackage.HipDir
        }
        if (-not $env:HIPCXX) {
            foreach ($candidate in @(
                (Join-Path $llvmBinDir "clang++.exe"),
                (Join-Path $binDir "clang++.exe"),
                (Join-Path $llvmBinDir "clang.exe"),
                (Join-Path $binDir "clang.exe")
            )) {
                if (Test-Path $candidate) {
                    $env:HIPCXX = $candidate
                    break
                }
            }
        }
    }

    $hipConfig = Resolve-CommandPath "hipconfig"
    if ($hipConfig) {
        try {
            $hipCompilerRoot = (& $hipConfig -l).Trim()
            if ($hipCompilerRoot) {
                $clangxx = Join-Path $hipCompilerRoot "clang++.exe"
                $clang = Join-Path $hipCompilerRoot "clang.exe"
                if (Test-Path $clangxx) {
                    $env:HIPCXX = $clangxx
                } elseif (Test-Path $clang) {
                    $env:HIPCXX = $clang
                }
            }
        } catch {
        }

        try {
            $hipRoot = (& $hipConfig -R).Trim()
            if ($hipRoot -and (Test-Path $hipRoot)) {
                $env:HIP_PATH = $hipRoot
                $env:ROCM_PATH = $hipRoot
                if (-not $hipPackage) {
                    $hipPackage = Resolve-HipPackageRoot
                    if ($hipPackage) {
                        $env:hip_DIR = $hipPackage.HipDir
                    }
                }
            }
        } catch {
        }
    }

    if (-not (Resolve-CommandPath "hipconfig") -and -not (Resolve-CommandPath "hipInfo") -and -not $rocmRoot) {
        throw "ROCm/HIP toolchain not found. Install ROCm on Windows and ensure hipconfig or hipInfo is available."
    }

    if (-not $hipPackage) {
        $hipPackage = Resolve-HipPackageRoot
    }
    if (-not $hipPackage) {
        throw "HIP package config not found. Expected hip-config.cmake under the HIP SDK installation."
    }
    $env:hip_DIR = $hipPackage.HipDir
}

function Ensure-VulkanToolchain {
    $sdkRoot = Resolve-VulkanSdkRoot
    if ($sdkRoot) {
        Add-ToPath (Join-Path $sdkRoot "Bin")
        Add-ToPath (Join-Path $sdkRoot "Bin32")
        if (-not $env:VULKAN_SDK) {
            $env:VULKAN_SDK = $sdkRoot
        }
        $env:CMAKE_PREFIX_PATH = if ($env:CMAKE_PREFIX_PATH) {
            "$sdkRoot;$env:CMAKE_PREFIX_PATH"
        } else {
            $sdkRoot
        }
    }

    $hasVulkanHeaders =
        ($env:VULKAN_SDK -and (Test-Path (Join-Path $env:VULKAN_SDK "Include\vulkan\vulkan.h"))) -or
        ($sdkRoot -and (Test-Path (Join-Path $sdkRoot "Include\vulkan\vulkan.h")))
    if (-not $hasVulkanHeaders) {
        throw "Vulkan SDK/development files not found. Install the Vulkan SDK and ensure VULKAN_SDK is configured."
    }

    if (-not (Resolve-CommandPath "glslc")) {
        throw "glslc not found. Install the Vulkan SDK and ensure its Bin directory is in PATH."
    }
}

function Invoke-InRepo {
    param(
        [scriptblock]$Script
    )

    Push-Location $repoRoot
    try {
        & $Script
    } finally {
        Pop-Location
    }
}

$Backend = Normalize-RecipeArgument $Backend @("backend")
$CudaArch = Normalize-RecipeArgument $CudaArch @("cuda_arch", "cudaarch")
$RocmArch = Normalize-RecipeArgument $RocmArch @("rocm_arch", "rocmarch", "amd_arch", "amdarch")

$backendName = Resolve-Backend $Backend
Write-Host "Using Windows backend: $backendName"

Ensure-MsvcToolchain

switch ($backendName) {
    "cuda" {
        Ensure-CudaToolchain
        if ($CudaArch) {
            Write-Host "Using CUDA architectures: $CudaArch"
        } else {
            Write-Host "Using CUDA toolkit at: $(Split-Path -Parent (Resolve-CommandPath 'nvcc'))"
        }
    }
    "rocm" {
        Ensure-RocmToolchain
        if ($RocmArch) {
            Write-Host "Using AMDGPU targets: $RocmArch"
        }
    }
    "vulkan" {
        Ensure-VulkanToolchain
    }
    "cpu" {
        Write-Host "Building Windows backend: CPU only"
    }
    default {
        throw "Unsupported backend '$backendName'. Use one of: cuda, rocm, hip, vulkan, cpu."
    }
}

Invoke-InRepo {
    if (-not (Test-Path $llamaDir)) {
        Write-Host "Cloning michaelneale/llama.cpp (rebase-upstream-master branch)..."
        git clone -b rebase-upstream-master https://github.com/michaelneale/llama.cpp.git $llamaDir
    } else {
        Push-Location $llamaDir
        try {
            $currentBranch = git branch --show-current
            if ($currentBranch -ne "rebase-upstream-master") {
                Write-Host "Switching llama.cpp from '$currentBranch' to rebase-upstream-master..."
                git checkout rebase-upstream-master
            }
            Write-Host "Pulling latest rebase-upstream-master from origin..."
            git pull --ff-only origin rebase-upstream-master
        } finally {
            Pop-Location
        }
    }

    $cmakeArgs = @(
        "-B", $buildDir,
        "-S", $llamaDir,
        "-DGGML_RPC=ON",
        "-DGGML_METAL=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_HIP=OFF",
        "-DGGML_VULKAN=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_OPENSSL=OFF"
    )

    if (Resolve-CommandPath "ninja") {
        $cmakeArgs = @("-G", "Ninja") + $cmakeArgs
    }

    switch ($backendName) {
        "cuda" {
            $cmakeArgs += "-DGGML_CUDA=ON"
            if ($CudaArch) {
                $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
            }
        }
        "rocm" {
            $cmakeArgs += "-DGGML_HIP=ON"
            if ($env:HIPCXX) {
                $cmakeArgs += "-DCMAKE_CXX_COMPILER=$env:HIPCXX"
            }
            if ($env:hip_DIR) {
                $cmakeArgs += "-Dhip_DIR=$env:hip_DIR"
            }
            if ($env:ROCM_PATH) {
                $cmakeArgs += "-DCMAKE_PREFIX_PATH=$env:ROCM_PATH"
            }
            if ($RocmArch) {
                $cmakeArgs += "-DAMDGPU_TARGETS=$RocmArch"
            }
        }
        "vulkan" {
            $cmakeArgs += "-DGGML_VULKAN=ON"
        }
        "cpu" {
        }
    }

    $parallelJobs = [Environment]::ProcessorCount
    & cmake @cmakeArgs
    & cmake --build $buildDir --config Release --parallel $parallelJobs
    Write-Host "Build complete: $buildDir\bin\"

    if (Test-Path $meshUiDir) {
        Write-Host "Building mesh-llm UI..."
        Push-Location $meshUiDir
        try {
            npm ci
            npm run build
        } finally {
            Pop-Location
        }
    }

    Write-Host "Building mesh-llm..."
    cargo build --release --locked -p mesh-llm
    Write-Host "Mesh binary: target\release\mesh-llm.exe"
}
