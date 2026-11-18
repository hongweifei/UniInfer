param(
    [string]$OrtVersion = "v1.31.0",
    [string]$WorkDir = "",
    [string]$OrtSource = "",
    [string]$OrtInstall = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) { $WorkDir = Join-Path $root ".build\vitis-desktop" }
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
# Windows target already adds DirectML; onnx-dynamic avoids linking the
# prebuilt ORT and loads the Vitis-enabled DLL from the application directory.
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

# Vitis AI ORT and provider DLLs must live next to the app executable.
$ortDll = Get-ChildItem -Path $OrtInstall -Filter "onnxruntime.dll" -Recurse | Select-Object -First 1
if (-not $ortDll) { throw "构建输出中没有 onnxruntime.dll" }
$ortDir = $ortDll.Directory.FullName
Get-ChildItem -Path $ortDir -Filter "*.dll" | Copy-Item -Destination $stage
Get-ChildItem -Path $OrtInstall -Filter "*.dll" -Recurse | ForEach-Object {
    if (-not (Test-Path (Join-Path $stage $_.Name))) {
        Copy-Item $_.FullName $stage
    }
}

Write-Host "已生成 AMD NPU 平台目录：$stage"
Write-Host "启动前确保 Vitis AI runtime DLL 在 PATH；该目录是内置 Vitis EP 的桌面端，不是 worker 包。"
