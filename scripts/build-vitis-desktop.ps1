param(
    [string]$SdkRoot = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $root ".build\vitis-desktop" }

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

# The SDK ships a Python 3.12 environment; use it instead of an arbitrary
# system Python that may be newer than the ONNX Runtime build supports.
$sdkVenv = Join-Path (Split-Path $SdkRoot) "ryzen-ai-1.9.0\Scripts"
if (Test-Path (Join-Path $sdkVenv "python.exe")) {
    $env:PATH = "$sdkVenv;$env:PATH"
}

Push-Location $root
cargo build --manifest-path apps/desktop/Cargo.toml --release `
    --features ep-vitis,onnx-dynamic
if ($LASTEXITCODE -ne 0) { throw "UniInfer AMD NPU 桌面端构建失败" }
Pop-Location

$stage = Join-Path $WorkDir "UniInfer-Windows-x64-AMD-NPU"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$exe = Join-Path $root "apps\desktop\target\release\uniinfer-desktop.exe"
if (-not (Test-Path $exe)) { throw "找不到桌面端 exe：$exe" }
Copy-Item $exe $stage

# Use the ONNX Runtime build installed by Ryzen AI Software. It is the
# Vitis-enabled build that matches the installed NPU driver and provider.
$ortBin = Join-Path $SdkRoot "onnxruntime\bin"
Get-ChildItem $ortBin -Filter "*.dll" | Copy-Item -Destination $stage -Force
$deploy = Join-Path $SdkRoot "deployment"
foreach ($name in @("vitis-ai-runtime.dll", "onnxruntime-genai.dll")) {
    $src = Join-Path $deploy $name
    if (Test-Path $src) { Copy-Item $src $stage -Force }
}

Write-Host "已生成 AMD NPU 平台目录：$stage"
Write-Host "该目录使用 Ryzen AI SDK 自带的 Vitis-enabled ONNX Runtime，不是 worker 包。"
