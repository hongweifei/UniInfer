# 推理后端包

UniInfer 把内置引擎与下载后端统一管理：每个内置引擎（ONNX Runtime、Candle、llama.cpp、whisper.cpp 等）单独展示 runtime、实现、格式、模态与设备；外置 Sidecar 后端逐项声明自己的引擎。两类后端共享同一套列表、激活状态和重启语义。

## 目录

桌面端「后端」页面支持：

- 填写远程目录 JSON URL，读取可下载后端。
- 下载并校验后端包，安装到 `<配置文件目录>/backends/<id>/<version>/`。
- 导入包含 `backend.toml` 的本地后端目录。
- 激活、删除后端；激活后重启应用生效。
- 选择内置引擎组作为回退。

## 后端清单

`backend.toml`：

```toml
id = "llama-vulkan"
name = "llama.cpp Vulkan"
version = "0.1.0"
description = "GGUF text inference on Vulkan"
platform = "windows-x86_64"
worker = "bin/uniinfer-worker.exe"
devices = ["gpu"]

[[engines]]
id = "llama-vulkan"
name = "llama.cpp Vulkan"
runtime = "llama"
description = "llama.cpp GGUF inference on Vulkan"
implementations = ["llama.text"]
formats = ["gguf"]
modalities = ["text"]
devices = ["gpu"]

[[files]]
path = "bin/uniinfer-worker.exe"
url = "https://example.com/llama-vulkan/uniinfer-worker.exe"
size = 123456789
sha256 = "..."
```

- `worker` 必须是相对路径，并且位于 `files` 中。
- 安装时校验文件大小与 SHA-256；缺少 `sha256` 的文件不会被拒绝，但生产目录应始终提供。
- `devices` 用于放置预算，可选 `cpu`、`gpu`、`npu`。
- `engines` 必须至少包含一项，逐项写清推理引擎 id、名称、runtime、格式、模态与设备；界面会按引擎展示，而不是笼统显示一个包。
- 当前 sidecar worker 实现现有 MessagePack 张量协议；加载、取消、流式扩展沿用同一 worker 生命周期。

## 远程目录

```json
{
  "backends": [
    {
      "id": "llama-vulkan",
      "name": "llama.cpp Vulkan",
      "version": "0.1.0",
      "description": "GGUF on Vulkan",
      "platform": "windows-x86_64",
      "worker": "bin/uniinfer-worker.exe",
      "devices": ["gpu"],
      "engines": [
        {
          "id": "llama-vulkan",
          "name": "llama.cpp Vulkan",
          "runtime": "llama",
          "description": "llama.cpp GGUF inference on Vulkan",
          "implementations": ["llama.text"],
          "formats": ["gguf"],
          "modalities": ["text"],
          "devices": ["gpu"]
        }
      ],
      "files": [
        {
          "path": "bin/uniinfer-worker.exe",
          "url": "https://example.com/uniinfer-worker.exe",
          "size": 123456789,
          "sha256": "..."
        }
      ]
    }
  ]
}
```

## 构建后端包

```powershell
cargo build --release -p uniinfer-worker --features llama-vulkan
```

将 `uniinfer-worker.exe` 及其依赖 DLL 放入包目录，生成 `backend.toml` 和 SHA-256，上传到任意静态文件服务即可。CUDA 包使用 `--features llama-cuda,ep-cuda`，DirectML 包使用 `--features ep-directml`，CoreML 包使用 `--features ep-coreml,candle-metal`。

## 运行时语义

- 激活的后端注册为进程隔离传输引擎，优先处理其声明的格式/模态。
- 图像、视频等内置引擎仍由主进程处理。
- 后端 worker 崩溃或取消时只影响对应模型进程。
- 后端激活/删除后需要重启主程序才能替换运行时；API `/v1/backends` 会返回 `restart_required`。
