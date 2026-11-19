param(
    [string]$SdkRoot = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $root ".build\vitis-desktop" }

. (Join-Path $PSScriptRoot "vitis-env.ps1")
$SdkRoot = Resolve-VitisSdkRoot $SdkRoot
Import-VisualStudioEnvironment
Add-VitisPythonPath $SdkRoot

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "缺少工具：cargo"
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
