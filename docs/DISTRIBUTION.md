# 分发与 EP 矩阵

ONNX Runtime 的 Execution Provider 在编译期链接，不能在一个通用安装包里运行时切换。因此按目标平台和硬件提供独立安装包；每个 Windows 包都包含 DirectML 作为基础回退。

## Windows

| 安装包 | ONNX EP | CUDA runtime |
| :--- | :--- | :--- |
| Universal | CPU + DirectML | 不需要 |
| Intel | CPU + DirectML + OpenVINO | 不需要 |
| AMD GPU | CPU + DirectML | 不需要 |
| AMD Ryzen AI NPU | CPU + DirectML + Vitis AI | 不需要 |
| NVIDIA CUDA 12 | CPU + DirectML + CUDA 12 + TensorRT | 用户安装匹配版本 |
| NVIDIA CUDA 13 | CPU + DirectML + CUDA 13 + TensorRT | 用户安装匹配版本 |

CUDA/cuDNN/TensorRT 不随安装包分发，由用户从 NVIDIA 官方安装；安装包只携带匹配的 ONNX Runtime CUDA EP。

CUDA 12/13 分开构建，因为 ONNX Runtime 构建和驱动要求不同。启动时如果 CUDA provider 初始化失败，仍可回退 DirectML。

## 其他平台

| 平台 | ONNX EP |
| :--- | :--- |
| macOS | CPU + CoreML |
| Linux NVIDIA | CPU + CUDA + TensorRT |
| Linux AMD | CPU + ROCm / MIGraphX |
| Linux Intel | CPU + OpenVINO / oneDNN |
| Android | CPU + NNAPI |
| iOS | CPU + CoreML |

## 构建约束

- `ort` 的 `download-binaries` 只提供固定的预编译 feature 组合；例如 Windows 有 `cuda13,tensorrt,nvrtx,directml`，但没有 `openvino,directml` 或 `vitis,directml`。
- 超出预编译组合的包需要自定义 ONNX Runtime 构建，并设置 `ORT_LIB_LOCATION`。
- Windows 包必须在目标依赖上启用 `ep-directml`；CLI 和桌面端已经这样做。
- AMD Ryzen AI 使用 Vitis AI EP；普通 AMD GPU 在 Windows 使用 DirectML，在 Linux 使用 ROCm。
