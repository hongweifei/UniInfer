# 分发与 EP 策略

平台安装包已经内置该平台的主用 EP；这些 EP 不再作为独立 worker 重复分发。worker 只用于安装包没有内置的额外 EP。

## 平台包

| 安装包 | 内置 EP |
| :--- | :--- |
| Windows Universal | CPU + DirectML |
| Windows NVIDIA | CPU + DirectML + CUDA + TensorRT + NVRTX |
| Windows Intel | CPU + DirectML + OpenVINO |
| Windows AMD GPU | CPU + DirectML |
| Windows AMD Ryzen AI NPU | CPU + DirectML + Vitis AI |
| macOS | CPU + CoreML |
| Linux NVIDIA | CPU + CUDA + TensorRT + NVRTX |
| Linux AMD | CPU + ROCm + MIGraphX |
| Linux Intel | CPU + OpenVINO + oneDNN |
| Linux ARM | CPU + ACL + XNNPACK |
| Android | CPU + NNAPI + XNNPACK |

平台包用户不需要再下载同一 EP 的 worker。

## 额外 worker

worker 只分发平台包没有内置的 EP，例如：

| Worker | 目标 |
| :--- | :--- |
| WebGPU | Windows/Linux 可选 GPU 路径 |
| XNNPACK | x86/ARM CPU 补充 |
| QNN | Qualcomm NPU |
| ROCm | 未内置 ROCm 的平台 |
| OpenVINO | Universal 包按需安装 |
| Vitis AI | 非 AMD NPU 平台按需安装 |

同一 EP 已经由平台包内置时，不应再提供重复 worker。

## Worker 版本

worker 独立于主程序版本，但必须声明：

```text
worker protocol
app compatibility range
ORT API
ORT/EP version
platform/arch
```

主程序只要求 worker 协议和模型契约兼容，不要求主程序与 worker 使用相同版本。

## 安装

- 平台包：直接安装，包含主用 EP
- 额外 EP：后端页在线下载 `ep-catalog.json` 中的 worker，或“导入后端包”
- worker 包自带 `onnxruntime.dll` 和 provider DLL

worker 包结构：

```text
backend.toml
uniinfer-worker.exe
onnxruntime.dll
provider DLLs
```
