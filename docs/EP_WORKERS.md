# 独立 EP Worker 包

自定义 EP 不进入主安装包时，使用独立 worker 进程。主程序只负责启动 worker；ONNX Runtime 和 EP provider 由 worker 包提供。

## 包结构

```text
uniinfer-worker.exe
onnxruntime.dll
<EP provider DLLs>
backend.toml
```

`backend.toml` 使用：

```toml
launcher = "worker"
worker = "uniinfer-worker.exe"
```

该包按普通 sidecar 后端安装，模型通过 `[backend]` 选择。

## XNNPACK 示例

1. 构建带 XNNPACK 的共享 ONNX Runtime：

```bash
./build.sh --use_xnnpack --build_shared_lib --config Release
```

2. 构建 worker：

```bash
cargo build -p uniinfer-worker --release --features ep-xnnpack,onnx-dynamic
```

3. 将 worker、ONNX Runtime DLL、XNNPACK/provider DLL 和 `backend.toml` 打包为 zip。

4. 主程序安装该 sidecar 后，在模型配置中选择该后端变体。

主程序不需要启用 XNNPACK feature；worker 内独立完成 ORT 和 EP 初始化。

## 边界

- 当前 worker 协议支持文本和音频模型；图像 `ImageSlot` 需要单独扩展。
- worker 与主程序通过 MessagePack tensor 帧通信，图像/大张量传输需要另行设计。
- 每个 worker 包必须匹配平台、架构、ORT API 版本和 provider 版本。

## 安装

后端页提供“导入后端包”，选择预先构建好的 zip 即可安装，不需要 Rust/CMake 环境。

本地导入要求 zip 根目录包含 `backend.toml`，并且清单中的 `worker` 文件存在。导入后会走普通 sidecar 注册流程，模型配置可以直接选择。

在线附加 EP 需要发布者把 worker 包托管到可下载地址，并提供后端目录索引；当前官方目录只包含 llama/sd/whisper/ONNX Runtime 的官方 release。
