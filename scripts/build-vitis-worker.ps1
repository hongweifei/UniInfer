param(
    [string]$OrtVersion = "v1.31.0",
    [string]$WorkDir = "",
    [string]$OrtSource = "",
    [string]$OrtInstall = "",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $root ".build\vitis-worker" }
if (-not $OrtSource) { $OrtSource = Join-Path $WorkDir "onnxruntime" }
if (-not $OrtInstall) { $OrtInstall = Join-Path $WorkDir "onnxruntime-install" }

if (-not [Environment]::Is64BitProcess) { throw "需要 64 位 PowerShell" }
if (-not $env:VITIS_AI_HOME -and -not $env:RYZEN_AI_HOME -and -not $env:RYZEN_AI_INSTALLATION_PATH) {
    throw "请先安装 AMD Ryzen AI Software，并设置 VITIS_AI_HOME、RYZEN_AI_HOME 或 RYZEN_AI_INSTALLATION_PATH"
}
foreach ($tool in @("git", "cmake", "python", "cargo")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "缺少工具：$tool" }
}
if (-not (Test-Path (Join-Path $OrtSource "build.bat"))) {
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
    git clone --recursive https://github.com/microsoft/onnxruntime $OrtSource
    Push-Location $OrtSource
    git checkout $OrtVersion
    git submodule sync --recursive
    git submodule update --init --recursive
    Pop-Location
}
New-Item -ItemType Directory -Force -Path $OrtInstall | Out-Null

Push-Location $OrtSource
& .\build.bat --use_vitisai --build_shared_lib --parallel --config Release `
    --skip_tests --cmake_extra_defines "CMAKE_INSTALL_PREFIX=$OrtInstall"
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime Vitis AI 构建失败" }
Pop-Location

Push-Location $root
cargo build -p uniinfer-worker --release --features ep-vitis,onnx-dynamic
if ($LASTEXITCODE -ne 0) { throw "uniinfer-worker 构建失败" }
Pop-Location

$stage = Join-Path $WorkDir "package"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$worker = Join-Path $root "target\release\uniinfer-worker.exe"
if (-not (Test-Path $worker)) { throw "找不到 worker：$worker" }
Copy-Item $worker (Join-Path $stage "uniinfer-worker.exe")

# ONNX Runtime 及 Vitis AI provider DLLs 必须和 worker 同目录。
$ortDll = Get-ChildItem -Path $OrtInstall -Filter "onnxruntime.dll" -Recurse | Select-Object -First 1
if (-not $ortDll) { throw "构建输出中没有 onnxruntime.dll" }
$ortDir = $ortDll.Directory.FullName
Get-ChildItem -Path $OrtDir -Filter "*.dll" | Copy-Item -Destination $stage
Get-ChildItem -Path $OrtInstall -Filter "*.dll" -Recurse | ForEach-Object {
    if (-not (Test-Path (Join-Path $stage $_.Name))) {
        Copy-Item $_.FullName $stage
    }
}

$files = Get-ChildItem -Path $stage -File | Sort-Object Name
$fileToml = ($files | ForEach-Object {
@"
[[files]]
path = "$($_.Name)"
url = ""
size = $($_.Length)
"@
}) -join "`n"

$backendToml = @"
id = "uniinfer-vitis-worker"
name = "UniInfer Vitis AI Worker"
version = "$Version"
description = "AMD Ryzen AI NPU worker with Vitis AI ONNX Runtime"
platform = "windows-x86_64"
worker = "uniinfer-worker.exe"
launcher = "worker"
devices = ["npu", "cpu"]

[[engines]]
id = "uniinfer-vitis-worker"
name = "Vitis AI ONNX Worker"
runtime = "onnx"
description = "ONNX Runtime with Vitis AI execution provider"
vendor = "AMD"
variant = "Vitis AI"
category = "language"
implementations = ["onnx.graph", "onnx.pipeline", "onnx.diffusion", "onnx.layered"]
formats = ["onnx"]
modalities = ["text", "audio", "image", "video"]
devices = ["npu", "cpu"]

$fileToml
"@
Set-Content -Path (Join-Path $stage "backend.toml") -Value $backendToml -Encoding UTF8

$zip = Join-Path $WorkDir "uniinfer-vitis-worker-$Version-windows-x64.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
Write-Host "已生成：$zip"
Write-Host "后端页使用“导入后端包”选择该 zip；发布在线目录时把它上传为 release asset，并生成 ep-catalog.json。"
