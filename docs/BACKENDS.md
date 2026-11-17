# 推理引擎

UniInfer 把内置引擎与下载后端统一管理：每个内置引擎（ONNX Runtime、Candle、llama.cpp、whisper.cpp 等）单独展示 runtime、实现、格式、模态与设备；外置 Sidecar 后端逐项声明自己的引擎。两类后端共享同一套列表、激活状态和重启语义。

## 目录

桌面端「后端」页面支持：

- 自动从 `llama.cpp` 官方 release 发现 Windows CPU、Vulkan、CUDA 引擎。
- 按官方项目、版本和加速变体下载并校验，安装到 `<配置文件目录>/backends/<id>/<version>/`。
- 激活、卸载引擎；激活后重启应用生效。
- 选择内置引擎组作为回退。

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
- `ONNX Runtime`：官方 DLL 仍需 `ort/load-dynamic` 适配，当前由内置 ONNX Runtime 提供。
- `whisper.cpp`：官方 server/CLI 需要 ASR 适配器，当前由内置 whisper.cpp 提供。
- `Candle`：上游不提供官方二进制，只能作为内置 Rust 引擎。

内部 `backend.toml` 只是安装后的启动描述，用户不需要编辑或手动导入。

## 运行时语义

- 激活的后端注册为进程隔离传输引擎，优先处理其声明的格式/模态。
- 图像、视频等内置引擎仍由主进程处理。
- 后端 worker 崩溃或取消时只影响对应模型进程。
- 后端激活/删除后需要重启主程序才能替换运行时；API `/v1/backends` 会返回 `restart_required`。
