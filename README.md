<h1 align="center">UniInfer</h1>

<p align="center">
  <b>一套 Rust 运行时，统一 ONNX、GGUF、Safetensors 的实时与批量推理</b><br/>
  实时双工四通道 · 图像与视频生成 · CPU/GPU/NPU 混合推理 · 桌面与移动同源
</p>

<p align="center">
  <a href="#项目状态"><img src="https://img.shields.io/badge/status-M0-blue.svg" alt="Status"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
  <a href="#许可证"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License"></a>
  <a href="docs/DESIGN.md"><img src="https://img.shields.io/badge/docs-design%20%2B%20roadmap-informational.svg" alt="Docs"></a>
  <a href="#贡献"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
</p>

---

## 项目状态

**M0 已交付，M1 已接通。** Trait、Manifest、Registry、ONNX CPU 引擎、字节级文本后端与 CLI 可运行；REST/SSE、会话取消、错误信封、`/v1/models`、`/metrics`、文本基线测量与 EP 编译矩阵均已接通。未接入的能力仍属目标契约。

## 目录

- [设计目标](#设计目标)
- [核心特性](#核心特性)
- [架构](#架构)
- [支持矩阵](#支持矩阵)
- [快速开始](#快速开始)
- [模型清单](#模型清单)
- [API 一览](#api-一览)
- [项目结构](#项目结构)
- [依赖策略](#依赖策略)
- [性能与基线](#性能与基线)
- [路线图](#路线图)
- [文档](#文档)
- [贡献](#贡献)
- [安全](#安全)
- [致谢](#致谢)
- [许可证](#许可证)

## 设计目标

**一个路由层覆盖三种格式。** 模型格式在加载时检测，`EngineRegistry` 按「格式 × 模态」解析引擎；调用方通过模型清单指定可选引擎。

**实时与批量两条通道正交。** 实时通道承载渐进预览帧、逐帧处理与流式语音；批量通道承载端到端图像与视频生成。同一格式可以在两条通道上分别路由。

**四路输入、四路输出。** `text`、`audio`、`image`、`video` 在事件流中地位对等，编排式与原生全双工后端共享同一套帧循环语义。

**一套 API，桌面与移动同源。** `axum` 提供 OpenAI 兼容 REST 与 WebSocket；Tauri 2 复用同一 Web UI 交付桌面与移动应用。

**CPU、GPU、NPU 统一放置。** `DeviceInventory` 探测三类设备与可用 EP，`PlacementPlan` 按内存、带宽与代价模型分配算子、层与阶段；混合推理覆盖 EP 分区、层卸载、阶段分置与跨引擎组合。

## 核心特性

- **三格式统一接入**：ONNX 走 `ort` + `onnx-genai`；GGUF 与 Safetensors 走 `CandleEngine`，同一 `InferenceEngine` 契约。
- **实时双工**：`/v1/realtime` 支持文本、音频、图像、视频四通道；打断级联在 3 帧内生效，取消语义跨层传播。
- **生成能力**：图像覆盖 SD 1.5/2.1、SDXL、SD 3/3.5、FLUX.1-dev、PixArt-Σ、Stable Cascade，支持 img2img、inpaint、LoRA/DoRA 堆叠与 AnimateDiff；视频覆盖 LTX-Video 2B/13B 与 AnimateDiff。
- **显式能力声明**：格式、量化、模态、设备类别、EP 的支持状态同时来自模型清单与 `capabilities()`；降级路径附带稳定 `fallback_reason`。
- **加载期预检**：量化、EP、显存与放置可行性四项在 `load_session` 前校验，错误在请求入口返回，推理循环保持干净。
- **取消与租约**：`CancellationToken` + RAII `SessionLease`，客户端断开、任务取消、模型驱逐共用一条释放路径，覆盖 CPU/GPU/NPU 资源。
- **异构放置**：CPU、GPU、NPU 统一探测；GPU 覆盖 CUDA、TensorRT、Metal、DirectML、ROCm、OpenVINO，NPU 覆盖 QNN、NNAPI、CoreML（ANE）、OpenVINO、Vitis AI、CANN。
- **混合推理**：算子级 EP 回退、层与 KV 卸载、阶段分置、跨引擎组合四层可选，回退与迁移均进入 `PlacementPlan` 与指标。
- **纯 Rust 推理路径**：服务端、引擎、桌面外壳与移动端共享同一代码库。
- **可观测**：请求、加载、prefill、decode 分层 span，延迟直方图与会话计量随每个里程碑扩展。

## 架构

```mermaid
flowchart TB
    subgraph API["API · axum"]
        A1["OpenAI 兼容 REST"]
        A2["/v1/realtime · WebSocket"]
    end

    subgraph Router["Router / Session"]
        R1["mode 路由：realtime / batch"]
        R2["四通道事件编排"]
        R3["租约 · 抢占 · 取消"]
    end

    subgraph Traits["模态 Backend"]
        T1["TextGenerationBackend"]
        T2["RealtimeVoiceBackend"]
        T3["ImageGenerationBackend"]
        T4["VideoGenerationBackend"]
    end

    subgraph Registry["EngineRegistry"]
        G1["格式 + 模态 → 引擎"]
        G2["量化 / EP / 显存预检"]
    end

    subgraph Placement["Placement"]
        P1["DeviceInventory · CPU/GPU/NPU"]
        P2["PlacementPlan · T1–T4"]
    end

    subgraph Engines["InferenceEngine"]
        E1["OnnxEngine · ort + onnx-genai"]
        E2["CandleEngine · GGUF / Safetensors"]
        E3["可选引擎 · frink / mistral.rs"]
    end

    subgraph GUI["GUI · Tauri 2"]
        U1["桌面"]
        U2["移动端"]
    end

    API --> Router --> Traits --> Registry --> Placement --> Engines
    GUI --> API
```

引擎契约保持最小，模态语义由 Backend 层承载：

```rust
pub trait InferenceEngine: Send + Sync {
    fn engine_name(&self) -> &'static str;
    fn supported_formats(&self) -> &[ModelFormat];
    fn supported_modalities(&self, format: ModelFormat) -> &[Modality];
    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization];
    fn supported_devices(&self) -> &[DeviceClass];
    fn capabilities(&self) -> EngineCapabilities;
    fn load_session(&self, spec: EngineLoadSpec) -> Result<EngineSession>;
}
```

`EngineSession` 提供 `run`、`run_streaming`、`cancel`；`EngineLoadSpec` 携带 `PlacementPlan`，帧循环、背压与打断级联在 Router 层实现，引擎按计划执行与取消。

## 支持矩阵

| 格式 | 模态 | 默认引擎 | 可选引擎 | 通道 |
| :--- | :--- | :--- | :--- | :--- |
| ONNX | 文本 / 嵌入 / 双工 / 图像 / 逐帧视频 | `OnnxEngine` | — | realtime + batch |
| GGUF | 文本（Qwen3 量化）/ 视觉语言 / 双工 | `CandleEngine` | `frink` | realtime + batch |
| GGUF | 图像生成 | `CandleEngine`（plakat） | — | batch |
| GGUF | 视频生成 | `CandleEngine`（candle-video / plakat） | — | batch |
| Safetensors | 文本 / 视觉语言 / 双工 | `CandleEngine` | `mistral.rs` | realtime + batch |
| Safetensors | 图像生成 | `CandleEngine`（plakat） | — | batch |
| Safetensors | 视频生成 | `CandleEngine`（candle-video / plakat） | — | batch |
| ONNX | 实时预览帧 | `OnnxEngine` | — | realtime |
| ONNX | 实时逐帧视频 | `OnnxEngine` + RAVE | — | realtime |

**量化**：GGUF 覆盖 Q2_K–Q8_0、Q4_0、Q8_0；FP8 在加载期返回 `UnsupportedQuantization` 并给出替代建议；IQ/MXFP4 由 `frink` 可选引擎覆盖。Safetensors 以 FP16/BF16 mmap 加载，FP8 的 ISQ 路径在 M11 评估。

**设备类别与 EP**：

| 设备类别 | 平台 | `OnnxEngine` EP | `CandleEngine` 后端 | 放置角色 |
| :--- | :--- | :--- | :--- | :--- |
| CPU | 全平台 | CPU EP | CPU | 回退执行、层卸载、小模型常驻 |
| GPU | Windows / Linux NVIDIA | CUDA、TensorRT | CUDA | prefill/decode 主力、扩散与视频 |
| GPU | macOS / iOS | CoreML（GPU 委派） | Metal | 实时双工与生成主力 |
| GPU | Windows AMD / Intel | DirectML | CPU / CUDA | AMD Radeon 与 Intel Arc 的 Windows 路径 |
| GPU | Android | NNAPI（GPU 委派） | — | 轻量视觉/语音前端 |
| GPU | Linux AMD | ROCm | — | 经 `OnnxEngine` 覆盖 |
| GPU | Linux Intel | OpenVINO | — | 经 `OnnxEngine` 覆盖 |
| NPU | Android | QNN、NNAPI（NPU/DSP 委派） | 经 `OnnxEngine` 路由 | ASR/TTS、常驻小模型 |
| NPU | Windows Copilot+ | QNN、OpenVINO | 经 `OnnxEngine` 路由 | 语音前端、低功耗推理 |
| NPU | AMD Ryzen AI | Vitis AI | 经 `OnnxEngine` 路由 | XDNA / XDNA2，本地语音与视觉编码 |
| NPU | Apple Silicon / iOS | CoreML（ANE 委派） | 经 `OnnxEngine` 路由 | 语音与视觉编码器 |
| NPU | 服务器 Ascend | CANN | 经 `OnnxEngine` 路由 | 高并发语音/视觉前端 |

**混合推理**：

| 层级 | 机制 | 承载 | 典型场景 |
| :--- | :--- | :--- | :--- |
| T1 算子级 | ORT 图分区，未覆盖算子回退 CPU | `OnnxEngine` | NPU 算子覆盖不完整的模型 |
| T2 层与缓存级 | transformer 层、KV cache 按预算分置 | `CandleEngine` | 权重超过单卡显存 |
| T3 阶段级 | 编码器、骨干、解码器、VAE 分置 | `plakat`、`candle-video`、编排式双工 | ASR 在 NPU、LLM 在 GPU、TTS 在 CPU |
| T4 引擎级 | Router 按模态或阶段分配引擎 | `OnnxEngine` + `CandleEngine` | NPU 语音前端 + GPU 文本生成 |

放置策略为 `auto`、`pinned`、`strict`；`auto` 默认按 **NPU → GPU → CPU** 选择，目标设备不可用或引擎不支持时回退，并把原因写入 `PlacementPlan.fallback_reason`。`PlacementPlan` 记录阶段→设备、卸载层数、KV cache 位置与回退清单，经 `/v1/models` 与会话 span 可查。NPU 与 AMD GPU 路径由 `OnnxEngine` 的 EP 承载，`CandleEngine` 负责 GGUF/Safetensors 的 CPU、CUDA、Metal。放置顺序对三种格式一致：NPU → GPU → CPU；Candle 暂未提供 NPU 后端，因此 GGUF/Safetensors 会回退 GPU/CPU 并记录 `engine_no_npu`。Vitis AI EP 需要 AMD 提供的 ONNX Runtime 构建（Ryzen AI SDK）；标准预编译包在链接期缺 Vitis 符号，`ep-vitis` 的 CI 只做 feature 编译检查。

## 快速开始

### 前置条件

- Rust 最新 stable 工具链
- M0 自带 `models/tiny-onnx` 固定件；GGUF、Safetensors 与 GPU/NPU 路径随里程碑接入
- CPU 为 M0 的默认设备；EP 由编译期 feature 选择，混合推理由放置计划编排

### 模型获取（ModelScope）

模型默认从 ModelScope 下载，`models/pull.sh` 封装了 `modelscope download`：

```bash
models/pull.sh Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b
```

脚本优先使用已安装的 `modelscope`，否则通过 `uvx --from modelscope modelscope` 运行；`local_dir` 会解析为绝对路径。固定件 `models/tiny-onnx` 由本地生成脚本产出，不依赖模型仓库。

Qwen3-0.6B 的固定获取脚本：

```bash
models/fetch-qwen3.sh                 # GGUF Q4_K_M
models/fetch-qwen3-safetensors.sh     # Safetensors BF16

cargo run -p uniinfer-cli -- run \
  --manifest models/qwen3-0.6b/manifest.toml \
  --prompt "你好" --max-tokens 32
```

GGUF 权重来自 ModelScope `unsloth/Qwen3-0.6B-GGUF`，Safetensors 与 tokenizer 来自 `Qwen/Qwen3-0.6B`。Qwen3.5 的 `qwen3_5` 架构尚未被 Candle 0.11 覆盖，接入需要新增引擎实现。

### 可选：llama.cpp 内置引擎（GGUF）

`uniinfer-engine-llama` 把 llama.cpp 作为内置依赖编译，GGUF 路由优先于 Candle。默认不启用，避免给纯 ONNX/Candle 构建引入 C++ 工具链：

```bash
cargo run -p uniinfer-cli --features llama -- run \
  --manifest models/qwen3-0.6b/manifest.toml \
  --prompt "你好" --max-tokens 32
```

构建前置条件：CMake、C++ 编译器（Windows 用 MSVC）和 `libclang`（bindgen 需要）。Windows 示例：

```powershell
$env:PATH = "C:\path\to\cmake\bin;$env:PATH"
$env:LIBCLANG_PATH = "C:\path\to\llvm\bin"
cargo build -p uniinfer-cli --features llama
```

AMD GPU 使用 Vulkan 后端。安装 Vulkan SDK 后启用 `llama-vulkan`；NVIDIA 用 `llama-cuda`：

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.363.0"
cargo run -p uniinfer-cli --features llama-vulkan -- run \
  --manifest models/qwen3-0.6b/manifest.toml \
  --prompt "你好" --max-tokens 32
```

`RUST_LOG=info` 可看到 llama.cpp 的设备与层卸载日志，例如 `using device Vulkan0 (AMD Radeon ...)`、`offloaded 29/29 layers to GPU`。未启用 GPU feature 时引擎只报告 CPU，放置计划自然回退。

### M0：CLI 与 ONNX CPU 路径

设备探测：

```bash
cargo run -p uniinfer-cli -- devices
```

一次性生成（固定件把输入字节推进一位，`ab` 生成 `cde`）：

```bash
cargo run -p uniinfer-cli -- run \
  --manifest models/tiny-onnx/manifest.toml \
  --prompt ab \
  --max-tokens 3
```

一键冒烟：

```bash
bash scripts/e2e/m0_smoke.sh
```

### 构建

```bash
cargo build --release
```

按目标平台选择 EP feature（随 M1/M6 接入）：

```bash
cargo build --release -p uniinfer-cli --features ep-cuda      # NVIDIA GPU
cargo build --release -p uniinfer-cli --features ep-coreml    # Apple GPU / ANE
cargo build --release -p uniinfer-cli --features ep-directml  # Windows GPU
cargo build --release -p uniinfer-cli --features ep-qnn       # Qualcomm NPU
cargo build --release -p uniinfer-cli --features ep-openvino  # Intel GPU / NPU
cargo build --release -p uniinfer-cli --features ep-rocm       # Linux AMD GPU
cargo build --release -p uniinfer-cli --features ep-vitis      # AMD Ryzen AI NPU（需 SDK 的 ORT 构建）
cargo build --release -p uniinfer-cli --features candle-cuda   # Candle CUDA 后端
cargo build --release -p uniinfer-cli --features candle-metal  # Candle Metal 后端
```

### 启动服务

```bash
cargo run -p uniinfer-cli --release -- serve \
  --manifest models/tiny-onnx/manifest.toml \
  --manifest models/tiny-onnx/manifest-2.toml \
  --addr 127.0.0.1:8080
```

GGUF 一次性推理（M4 目标路径）：

```bash
cargo run -p uniinfer-cli --release -- run \
  --manifest models/qwen2.5-1.5b-instruct-q4_k_m.toml \
  --prompt "用一句话解释张量并行" \
  --max-tokens 64
```

### 调用

流式聊天：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "用一句话解释张量并行"}],
    "stream": true
  }'
```

图像生成：

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "flux.1-dev-q5_k_m",
    "prompt": "雨夜的霓虹街道，电影感，35mm",
    "size": "1024x1024",
    "steps": 20,
    "seed": 42
  }'
```

视频生成走异步任务：

```bash
curl http://127.0.0.1:8080/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "ltx-video-2b-distilled", "prompt": "海浪拍打礁石，日出", "duration": 2}'
# → {"id": "job_7f3c", "status": "queued"}

curl http://127.0.0.1:8080/v1/videos/generations/job_7f3c
```

### 桌面端

```bash
cargo run -p uniinfer-desktop
```

应用在进程内启动 loopback API（随机端口 + 一次性 bearer token），窗口通过 Tauri command 获取地址与 token。默认加载 `models/tiny-onnx/manifest.toml`；`UNIINFER_MANIFEST` 指定单个清单，`UNIINFER_MANIFESTS`（分号分隔）加载多个模型并在标题栏切换。使用 WebView2，Windows 11 已内置。

## 模型清单

清单描述模型的格式、量化、能力与资源需求，加载前完成路由与预检：

```toml
schema_version = 1
id = "qwen2.5-1.5b-instruct-q4_k_m"
format = "gguf"
quantization = "q4_k_m"
modalities = ["text"]
streaming_capable = true
context_len = 32768
tokenizer = "qwen2"
min_vram = "2GiB"
[placement]
policy = "auto"
prefer = ["npu", "gpu", "cpu"]
allow_cpu_fallback = true
offload_layers = "auto"
kv_cache = "auto"

[[files]]
path = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
sha256 = "9f8e6c1d0a4b7e2f5c8d1a3b6e9f0c2d4a7b1e8f3c6d9a2b5e0f4c7d1a8b3e6f"  # 由 fetch 脚本写入
```

字段说明见 [ROADMAP · M0](docs/ROADMAP.md)；清单校验失败返回 `ManifestInvalid`，不支持的量化为 `UnsupportedQuantization`，放置无法满足时返回 `PlacementUnsatisfiable`。`auto` 允许按预算回退并记录 `fallback_reason`，`pinned` 固定阶段设备，`strict` 关闭算子回退。

## API 一览

| 端点 | 方法 | 用途 |
| :--- | :--- | :--- |
| `/v1/chat/completions` | POST | 文本生成，`stream: true` 走 SSE |
| `/v1/audio/transcriptions` | POST | 语音转写 |
| `/v1/audio/speech` | POST | 语音合成 |
| `/v1/images/generations` | POST | 文生图 |
| `/v1/images/edits` | POST | img2img / inpaint / outpaint |
| `/v1/videos/generations` | POST | 视频生成任务 |
| `/v1/videos/generations/{id}` | GET | 任务查询 |
| `/v1/realtime` | WS | 实时会话，text + audio VAD |
| `/v1/models` | GET | 已加载模型、能力矩阵与放置计划 |
| `/v1/devices` | GET | 运行时设备清单与各模型放置/回退原因 |
| `/v1/sessions/{id}/cancel` | POST | 取消生成 |

实时会话以 JSON 文本帧承载控制事件，以 20ms PCM16 二进制帧承载音频：

```json
{"type": "session.start", "model": "moshi-duplex-8bit", "channels": ["text", "audio"]}
{"type": "input_audio_buffer.append"}
{"type": "response.create"}
```

当前实现 text 与 audio 通道：客户端发送 `session.start`、`input.text`、`response.create`、`response.cancel`，音频以 20ms / 16kHz / PCM16（640 字节）二进制帧上传。服务端返回 `session.started`、`input.text.accepted`、`input.speech.started`、`input.speech.stopped`、`output.text.delta`、`response.cancelled`、`response.done`。能量 VAD 检测到语音时会取消当前响应（打断）。ASR/TTS 尚未接入，`image`、`video` 通道返回 `unsupported_channel`。

## 项目结构

```
# 当前 M0 已落地前 5 个；其余随路线图加入
crates/
  uniinfer-core          # 枚举、Manifest schema、EngineError、Trait
  uniinfer-devices       # DeviceInventory、代价模型、PlacementPlan
  uniinfer-registry      # 格式+模态路由、预检、SessionLease、显存预算
  uniinfer-service       # Runtime、ModelSlot、生成入口与装配
  uniinfer-engine-onnx   # ort + onnx-genai；EP feature 门控
  uniinfer-engine-candle # candle-core/transformers；GGUF / Safetensors
  uniinfer-backends      # 四个模态 Backend，引擎无关
  uniinfer-router        # mode 路由、帧编排、打断级联、异步任务
  uniinfer-api           # axum 路由、OpenAI 兼容层、WebSocket
  uniinfer-cli           # 冒烟、清单校验、bench 驱动
  uniinfer-bench         # 场景矩阵与基线对比
apps/desktop             # Tauri 2，独立 workspace
models/                  # manifest + fetch 脚本 + sha256
scripts/e2e/             # 里程碑出口脚本
```

依赖方向由 `cargo metadata` 在 CI 断言：`uniinfer-devices` 位于 core 与 registry 之间，`uniinfer-backends` 的依赖闭包限定在 core 与 registry，`engine-*` 之间互不依赖。

## 依赖策略

依赖跟踪上游最新稳定版；`ort` 这类仅发布 RC 的项目使用最新 RC。当前基线版本见 [DESIGN §12](docs/DESIGN.md)。`Cargo.lock` 随仓库提交，升级 PR 重跑契约测试、数值容差与基线对比，主版本升级附 ADR。

## 性能与基线

M1 起按「场景 × 模型 × 引擎 × 设备类别 × EP」矩阵测量。`uniinfer-bench` 输出 JSON，`compare` 以 +10% 报警、+25% 失败判定回归：

```bash
cargo run -p uniinfer-bench -- run \
  --manifest models/tiny-onnx/manifest.toml \
  --prompt ab --max-tokens 3 --iterations 5 \
  --save bench/baseline.json

cargo run -p uniinfer-bench -- compare \
  --baseline bench/baseline.json --current bench/current.json
```

当前指标：TTFT、总耗时、tok/s、生成字节数、峰值 RSS。实时因子、首音频/首帧延迟、峰值显存与跨设备迁移字节随 M3–M8 加入。

## 路线图

| 阶段 | 主题 | 里程碑 |
| :--- | :--- | :--- |
| 控制面 | 契约 → REST/SSE → 桌面端 → 双工语音 | M0–M3 |
| 引擎 | GGUF 文本 → Safetensors/双工 | M4–M5 |
| 设备 | NPU EP、PlacementPlan、T1–T4 混合推理 | M6 |
| 生成 | 图像 → 视频 | M7–M8 |
| 合并 | 预览帧与逐帧视频 | M9 |
| 扩展 | 移动端、可选引擎与规模化 | M10–M11 |

完整出口门禁、决策门与风险登记见 [ROADMAP](docs/ROADMAP.md)。当前进度：**M0、M1 已交付；M2 桌面骨架可用；M3 文本与 PCM VAD 通道已接通；GGUF 文本已通过 Candle 接入 Qwen3-0.6B Q4_K_M**。

## 文档

| 文档 | 内容 |
| :--- | :--- |
| [DESIGN.md](docs/DESIGN.md) | 分层架构、引擎路由、格式-模态矩阵、EP 策略 |
| [ROADMAP.md](docs/ROADMAP.md) | 里程碑、出口门禁、决策门、风险登记 |
| `docs/decisions/` | 架构决策记录，随对应里程碑落地 |
| `docs/ENGINE_MATRIX.md` | 引擎一致性表，随 M5 交付 |

## 贡献

1. 先在 Issue 中对齐范围与接口；路由、Trait、事件协议属于契约变更，需附设计说明。
2. 提交前运行：

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```

3. 新增引擎需通过 `tests/engine_contract.rs` 的全量契约断言，并在一致性表登记支持矩阵。
4. 行为变更与文档修订同批合入；决策记录写入 `docs/decisions/`。

## 安全

模型权重携带各自的许可证与使用条款，分发与商用前请核对上游模型卡片。安全问题请通过仓库的 Security Advisory 私下报告，附复现步骤与影响范围。

## 致谢

UniInfer 建立在以下项目之上：

- [Candle](https://github.com/huggingface/candle) — Rust 张量与模型框架
- [ort](https://github.com/pykeio/ort) — ONNX Runtime 的 Rust 绑定
- [onnx-genai](https://github.com/justinchuby/onnx-genai) — ONNX 生成式推理
- [plakat](https://github.com/vulogov/plakat) — Candle 图像生成
- [candle-video](https://github.com/FerrisMind/candle-video) — LTX-Video 的 Rust 实现
- [frink](https://crates.io/crates/frink-server) — GGUF 推理引擎
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) — Safetensors 推理与 ISQ
- [RAVE](https://github.com/acids-ircam/RAVE) — 实时音频/逐帧模型
- [Tauri](https://tauri.app) — 桌面与移动外壳

## 许可证

本项目采用 **MIT OR Apache-2.0** 双许可，任选其一：

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

### 贡献许可

你提交的贡献默认按 MIT OR Apache-2.0 双许可授权，适用 [Apache-2.0 §5](LICENSE-APACHE) 的条款；你以书面形式明确声明的许可安排从其声明。
