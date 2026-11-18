param(
    [string]$SdkRoot = "",
    [string]$WorkDir = "",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $root ".build\vitis-worker" }

if (-not $SdkRoot) {
    $SdkRoot = $env:VITIS_AI_HOME
    if (-not $SdkRoot) { $SdkRoot = $env:RYZEN_AI_HOME }
    if (-not $SdkRoot) { $SdkRoot = $env:RYZEN_AI_INSTALLATION_PATH }
}
if (-not $SdkRoot) {
    throw "请先安装 AMD Ryzen AI Software，并设置 VITIS_AI_HOME、RYZEN_AI_HOME 或 RYZEN_AI_INSTALLATION_PATH"
}
if (-not (Test-Path (Join-Path $SdkRoot "onnxruntime\bin\onnxruntime.dll"))) {
    throw "SDK 中找不到 onnxruntime\bin\onnxruntime.dll：$SdkRoot"
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "请从 Visual Studio Developer PowerShell 运行，普通 PowerShell 没有 cl"
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "缺少工具：cargo"
}

$sdkVenv = Join-Path (Split-Path $SdkRoot) "ryzen-ai-1.9.0\Scripts"
if (Test-Path (Join-Path $sdkVenv "python.exe")) {
    $env:PATH = "$sdkVenv;$env:PATH"
}

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

# Use the SDK's Vitis-enabled ONNX Runtime and provider DLLs.
$ortBin = Join-Path $SdkRoot "onnxruntime\bin"
Get-ChildItem $ortBin -Filter "*.dll" | Copy-Item -Destination $stage -Force
$deploy = Join-Path $SdkRoot "deployment"
foreach ($name in @("vitis-ai-runtime.dll", "onnxruntime-genai.dll")) {
    $src = Join-Path $deploy $name
    if (Test-Path $src) { Copy-Item $src $stage -Force }
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
