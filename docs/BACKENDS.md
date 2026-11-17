# 推理引擎

UniInfer 把内置引擎与下载后端统一管理：每个内置引擎（ONNX Runtime、Candle、llama.cpp、whisper.cpp 等）单独展示 runtime、实现、格式、模态与设备；外置 Sidecar 后端逐项声明自己的引擎。后端没有全局激活状态，安装后由模型 manifest 的 `[backend]` 选择使用哪个后端和变体。

## 目录

桌面端「后端」页面支持：

- 自动从 `llama.cpp` 和 `stable-diffusion.cpp` 官方 release 发现全部 Windows x64 变体，包括 CPU、Vulkan、CUDA、ROCm、SYCL、OpenVINO 等。
- 按官方项目、版本和加速变体下载并校验，安装到 `<配置文件目录>/backends/<id>/<version>/`。
- 卸载引擎；安装后后端即存在于列表中，供模型配置选择。
- 同一项目的全部变体按项目分组展示；每个引擎只显示一张卡片，变体是独立下载项，CUDA 等带版本号的变体会分别保留。
- 后端下载与在线模型下载共用「下载任务」队列，支持排队、进度、取消、失败重试和清除记录。

## 引擎清单

官方引擎安装时生成内部 `backend.toml` 启动描述；普通用户不需要编辑或导入它。

```toml
id = "llama-vulkan"
name = "llama.cpp Vulkan"
version = "0.1.0"
description = "GGUF text inference on Vulkan"
platform = "windows-x86_64"
launcher = "llama-server"
worker = "llama-server.exe"
devices = ["gpu"]

[[engines]]
id = "llama-vulkan"
name = "llama.cpp Vulkan"
runtime = "llama"
description = "llama.cpp GGUF inference on Vulkan"
vendor = "llama.cpp"
variant = "Vulkan"
category = "language"
homepage = "https://github.com/ggml-org/llama.cpp"
implementations = ["llama.text"]
formats = ["gguf"]
modalities = ["text"]
devices = ["gpu"]

[[files]]
path = "llama-b11836-bin-win-vulkan-x64.zip"
url = "https://github.com/ggml-org/llama.cpp/releases/download/b11836/llama-b11836-bin-win-vulkan-x64.zip"
size = 123456789
sha256 = "..."
```

- `worker` 必须是相对路径，并且位于 `files` 中；`launcher = "llama-server"` 表示使用官方 `llama-server` 适配器，`worker` 指向解压后的官方可执行文件。
- `.zip` 引擎归档会自动解压，`worker` 使用解压后的相对路径。
- 安装时校验文件大小与 SHA-256；缺少 `sha256` 的文件不会被拒绝，但生产目录应始终提供。
- `devices` 用于放置预算，可选 `cpu`、`gpu`、`npu`。
- `engines` 必须至少包含一项，逐项写清推理引擎 id、名称、runtime、vendor、variant、category、格式、模态与设备；界面会按官方项目和加速变体分组展示。
- 官方 `llama-server` 适配器使用官方 HTTP `/completion` 流式协议。

## 官方引擎

- `llama.cpp`：下载官方 release 的 `llama-server` 压缩包，设置 `launcher = "llama-server"`。UniInfer 启动官方 server，通过 `/completion` 流式协议适配。
- `stable-diffusion.cpp`：下载官方 `sd-server` 压缩包（含 ROCm 变体），设置 `launcher = "stable-diffusion-server"`。UniInfer 启动官方 server，通过 A1111兼容的 `/sdapi/v1/txt2img` 生成图像；模型 manifest 需要提供完整 checkpoint 文件。
- `ONNX Runtime`：官方 DLL 仍需 `ort/load-dynamic` 适配，当前由内置 ONNX Runtime 提供。
- `whisper.cpp`：官方 server/CLI 需要 ASR 适配器，当前由内置 whisper.cpp 提供。
- `Candle`：上游不提供官方二进制，只能作为内置 Rust 引擎。

内部 `backend.toml` 只是安装后的启动描述，用户不需要编辑或手动导入。

## 模型选择后端

模型 manifest 通过 `[backend]` 指定使用哪个后端族和变体：

```toml
[backend]
id = "llama.cpp"
variant = "auto"
```

- `id` 是后端族，对应桌面上按项目分组的 `name`，例如 `llama.cpp`、`stable-diffusion.cpp`；`builtin` 表示只使用内置引擎。
- `variant` 是已安装变体，例如 `CPU`、`Vulkan`、`CUDA 12.4`、`ROCm 10.0`、`SYCL`；省略或 `auto` 时用设备后端名称与变体名称做通用匹配，优先匹配的加速变体，没有匹配时回退 CPU。
- 没有 `[backend]` 时保持默认路由：内置引擎优先，模型按 `runtime` 和 `implementation` 匹配。
- 模型运行配置界面会读取已安装后端，按项目生成后端选项，并把设备推荐变体显示在“自动”选项上。
- 变体支持不是白名单：桌面端同时用设备后端名称、GPU 厂商/型号和变体名称做通用匹配；CUDA 在检测到 AMD 时禁用，ROCm 在检测到 NVIDIA 时禁用，其他加速变体只要有 GPU/NPU 就允许下载。

## 运行时语义

- 所有已安装后端变体都会注册为进程隔离传输引擎；模型 manifest 决定具体使用哪一个，不存在全局“当前运行后端”。
- 内置引擎仍在主进程处理；未选择外置后端的模型继续使用内置路由。
- 后端 worker 崩溃或取消时只影响对应模型进程。
- 新安装的后端在下一次读取 `/v1/backends` 时注册到运行时，不需要切换激活状态；删除后端会移除文件，正在加载的会话需要卸载或重启后释放。
