# UniInfer 设计

## 1. 目标

UniInfer 是一个 Rust 推理运行时，统一 ONNX、GGUF、Safetensors 与 GGML 模型，覆盖文本、语音、视觉、图像与视频能力。运行时向本地应用提供 OpenAI 兼容 API，并把模型加载、设备放置、取消、超时与指标收敛到同一层。

## 2. 核心抽象

### Capability

模型通过 capability 描述可执行请求：

```text
operation
required_inputs
optional_inputs
outputs
mode
```

示例：

```toml
[[capabilities]]
name = "speech_to_speech"
operation = "generate"
required_inputs = ["audio"]
optional_inputs = ["text"]
outputs = ["audio"]
mode = "batch"
```

API 形成 `CapabilityRequest`，按 operation、输入集合、输出集合与 mode 匹配模型。

### Manifest

manifest 声明 runtime、implementation、format、quantization、placement、capabilities、按能力分组的配置（text / vision / audio / image / pipeline）与 files。

### Engine

`InferenceEngine` 提供 runtime、implementations、format、modality、quantization、device 与 `load_session`。`EngineRegistry` 按 runtime、implementation、format、量化、输入模态与 placement 选引擎。

### EngineSession

`EngineSession` 是 named-tensor 执行契约：

```rust
fn inputs(&self) -> &[TensorSpec];
fn outputs(&self) -> &[TensorSpec];
fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
fn run_with_deadline(...);
fn run_streaming(...);
fn native_decoder(&self) -> Option<&dyn NativeTextDecoder>;
fn cancel(&self);
```

`NativeTextDecoder` 承载 llama.cpp 这类内部完成 tokenize 与 decode 的引擎。

### ModelExecutor

`ModelExecutor` 统一 slot 输出：

```text
ModelRequest::Text / Audio / Image
ModelEvent::TextDelta / Progress
ModelOutput::Text / Audio / Image / Video
```

`ModelSlot`、`AudioSlot`、`ImageSlot` 实现同一契约。

## 3. ONNX

ONNX 是默认执行层。

### onnx.graph

单图模型直接映射 named tensors。文本、GenAI KV cache、fixture 与图像 fixture 使用这一路径。

### onnx.pipeline

多图模型由 manifest 声明：

```text
graph step      -> 执行一个 ONNX 图
merge step      -> 把图像特征写进 embedding
adapter step    -> Whisper/CTC 等有状态解码器
auto inputs     -> attention_mask / position_ids / empty KV
```

预处理与 token layout 通过 profile 选择：

```text
image_preprocessor = "smolvlm"
token_layout = "smolvlm"
```

引擎执行步骤与绑定，模型语义留在 manifest 与 adapter。

## 4. 其他引擎

| 引擎 | 实现 | 能力 |
| :--- | :--- | :--- |
| `OnnxEngine` | `onnx.graph` / `onnx.pipeline` | 文本、视觉、ASR、图像 fixture |
| `CandleEngine` | text / vision family registry | GGUF/Safetensors 文本、Safetensors 视觉 |
| `LlamaEngine` | llama.cpp | GGUF 文本 |
| `LlamaMtmdEngine` | llama.cpp mtmd | GGUF 视觉 |
| `WhisperEngine` | whisper.cpp | GGML 转写 |

## 5. 图像生成

`ImageSlot` 接收 `ModelRequest::Image`，返回 `ModelOutput::Image`。API 层负责 capability 选择、参数校验与 `b64_json` 响应。

```text
prompt / negative_prompt
size / steps / guidance / seed
  -> ImageSlot
  -> ImageGenerationBackend
  -> PNG / WebP
```

ONNX 主路径使用 `onnx.diffusion`：

```text
CLIP tokenizer
  -> text_encoder/model.onnx
  -> EulerDiscrete scheduler
  -> unet/model.onnx loop
  -> vae_decoder/model.onnx
  -> RGB8
```

`[diffusion]` 声明组件文件、张量名、调度器参数、latent 通道、VAE scale 与默认步数。首个模型为 `Heliosoph/sd-turbo-onnx`：512×512、1–4 步、guidance 1.0、EulerDiscrete。

- SD1.5 / SDXL：同一实现，增加双文本编码器与 UNet 条件输入。
- FLUX：增加 T5/CLIP 双编码器与 guidance embedding。
- img2img、inpaint、outpaint：`/v1/images/edits`，复用 VAE encoder 与 UNet。
- 进度通过 `ModelEvent::Progress { step, total }`。
- 固定 seed 复现；OOM 映射 `InsufficientVram`。

`plakat` 作为 GGUF/Safetensors 后端，在 ONNX 路径稳定后接入，承载 SDXL/FLUX GGUF 与 AnimateDiff。

## 6. 视频生成

`VideoSlot` 接收 `ModelRequest::Video`，返回 `ModelOutput::Video`。

```text
text -> video
image + text -> video
video + text -> video
```

- `/v1/videos/generations` 返回 `202 + job_id`，查询与取消走独立端点。
- `candle-video` LTX-Video 2B distilled 先行：2–5s、512–768px、MP4、VAE tiling。
- `plakat` AnimateDiff 提供第二路径。
- 文本编码器结果可缓存；任务状态落盘；取消终止 worker 并释放显存。
- 13B 在 24GB 机型通过，或形成预检拒绝结论。

## 7. 实时多通道

`/v1/realtime` 使用 WebSocket：

```text
文本帧：session.start / input_audio_buffer.append / response.create / cancel
二进制帧：20ms PCM16 mono
输出：output.text / output.audio / output.image / output.video
```

当前实现覆盖文本输入、PCM 音频输入、能量 VAD、ASR → 文本与打断。

后续扩展：

- `output.audio`：TTS executor，文本 → `ModelOutput::Audio`。
- `output.image`：扩散预览帧，`ModelEvent::Progress` 携带 step。
- `output.video`：decode → preprocess → inference → encode，背压到采集端。
- 丢弃策略：丢旧帧，生成侧持续前进。
- 打断级联：取消 TTS → 清空播放缓冲 → 取消 LLM → 发 `output.audio.clear` 与 `output.text.cancelled`。

## 8. 设备与放置

`DeviceInventory` 探测 CPU、GPU、NPU、执行提供器、内存与带宽。`PlacementPlan` 记录阶段设备、层卸载、KV cache 位置、迁移字节与 `fallback_reason`。

| 层级 | 机制 | 承载 |
| :--- | :--- | :--- |
| T1 算子级 | ORT 图分区，未覆盖算子回退 CPU | `OnnxEngine` |
| T2 层与缓存级 | transformer 层、KV cache 分置 | `CandleEngine` |
| T3 阶段级 | 编码器、骨干、解码器、VAE 分置 | `plakat`、`candle-video`、编排式双工 |
| T4 引擎级 | 模态或阶段分配给不同引擎 | `OnnxEngine` + `CandleEngine` |

放置策略：

```toml
[placement]
policy = "auto"            # auto | pinned | strict
allow_cpu_fallback = true
stages.asr = "npu"
stages.llm = "gpu"
stages.tts = "cpu"
offload_layers = "auto"
kv_cache = "auto"
```

`auto` 按 NPU → GPU → CPU 选择；`pinned` 固定阶段；`strict` 关闭算子回退。

## 9. 桌面与移动

Tauri 2 桌面端在进程内启动 loopback API，复用同一配置、模型目录与接口。移动端沿用同一 manifest 与 capability 契约，音频 I/O 通过 PCM 回调与原生播放器接入。

## 10. 测试与基线

- `cargo test --workspace` 覆盖核心契约、路由、API 与 fixture 模型。
- 真实模型测试位于各 crate 的 `tests/`，fixture 缺失时自动跳过。
- `cargo clippy --workspace --all-targets -- -D warnings` 与 `cargo fmt --check` 作为提交门禁。
- `uniinfer-bench` 记录 TTFT、tok/s、峰值 RSS、ASR 时长与按模型指标。
- 图像按固定 seed 回归；视频记录时长、分辨率与峰值显存。

## 11. 依赖策略

依赖以 `Cargo.lock` 为准。升级随独立 PR 合入，并重跑契约测试、数值容差与基线。

- ONNX Runtime 由 `ort` 提供，固定版本。
- Candle 固定版本，文本/视觉 family 通过 adapter 注册。
- `llama-cpp-2` 内置编译，feature 门控。
- `onnx.diffusion` 作为图像生成主路径；`plakat` 承载 GGUF/Safetensors 图像后端；`candle-video` 承载视频后端。
- `frink`、`mistral.rs` 作为可选引擎评估。
