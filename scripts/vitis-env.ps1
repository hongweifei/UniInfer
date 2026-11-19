# Shared environment discovery for the Vitis build scripts.

function Resolve-VitisSdkRoot {
    param([string]$Requested)

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($Requested) { $candidates.Add($Requested) }
    foreach ($name in @("VITIS_AI_HOME", "RYZEN_AI_HOME", "RYZEN_AI_INSTALLATION_PATH")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { $candidates.Add($value) }
    }
    $base = Join-Path $env:ProgramFiles "RyzenAI"
    if (Test-Path $base) {
        Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { $candidates.Add($_.FullName) }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate "onnxruntime\bin\onnxruntime.dll")) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "未找到 Ryzen AI Software。请安装 SDK，或通过 -SdkRoot 指定安装目录。"
}

function Import-VisualStudioEnvironment {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }

    $programFilesX86 = [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86)
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "找不到 Visual Studio 2022 C++ 工具链，请安装 Visual Studio C++ Build Tools。"
    }
    $vsRoot = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsRoot) {
        throw "找不到已安装的 Visual Studio C++ 工具链。"
    }
    $devCmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
    $cmdLine = '"' + $devCmd + '" -arch=amd64 -host_arch=amd64 >nul && set'
    $lines = & cmd.exe /d /s /c $cmdLine
    foreach ($line in $lines) {
        $separator = $line.IndexOf("=")
        if ($separator -gt 0) {
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        throw "Visual Studio 开发环境初始化失败，找不到 cl。"
    }
}

function Add-VitisPythonPath {
    param([string]$SdkRoot)

    $parent = Split-Path $SdkRoot
    $venv = Get-ChildItem $parent -Directory -Filter "ryzen-ai-*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "Scripts\python.exe" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if ($venv) {
        $env:PATH = "$(Split-Path $venv);$env:PATH"
    }
}
