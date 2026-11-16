# UniInfer 路线图

本路线图按垂直切片组织：每个里程碑交付一条可运行、可验收的完整路径。当前能力作为基线，后续里程碑覆盖图像、编辑、视频、实时多通道、设备矩阵、移动端与可选引擎。

## 1. 当前基线

- capability 路由与 `model = "auto"` 自动选择。
- ONNX `onnx.graph` 与 `onnx.pipeline`：文本、视觉、Whisper/CTC。
- Candle GGUF/Safetensors 文本与 Safetensors 视觉。
- llama.cpp GGUF 文本与 mtmd 视觉。
- whisper.cpp GGML 转写。
- OpenAI 兼容聊天、多图聊天、语音转写、图像生成管线。
- `ModelExecutor` 与 `ModelOutput::{Text,Audio,Image,Video}`。
- `ImageSlot`、PNG、`b64_json`、`tiny-image-onnx` fixture。
- Tauri 2 桌面端。

## 2. 组织原则

1. **垂直切片。** 每个里程碑给出一条可演示路径、测试脚本与量化门禁。
2. **能力先行。** 新功能先定义 capability 与 `ModelExecutor` 请求/输出，再接模型。
3. **manifest 驱动。** 新模型优先只写 manifest；模型差异进入 adapter 与 profile。
4. **ONNX 优先。** ONNX 用通用 graph 与 pipeline 承载文本、视觉、语音与生成。
5. **设备中性。** CPU、GPU、NPU 由 `DeviceInventory` 探测，`PlacementPlan` 记录阶段、层、KV 与回退原因。
6. **风险前置。** 外部依赖先做时间盒探针，结论写入文档并决定范围。
7. **契约优先。** 跨引擎验收能力、取消、流式、错误码与放置；数值一致在同设备、同驱动内验收。

## 3. 能力面

| 能力 | 输入 | 输出 | 状态 |
| :--- | :--- | :--- | :--- |
| 文本聊天 | text (+image) | text | 已实现，支持多图 |
| 语音转写 | audio | text | 已实现 |
| 图像生成 | text | image | 管线已实现，真实扩散后端下一阶段 |
| 图像编辑 | image (+text) | image | 规划 |
| 视频生成 | text / image (+text) | video | 规划 |
| 视频编辑 | video (+text) | video | 规划 |
| 语音合成 | text | audio | 规划 |
| 实时多通道 | text/audio/image/video | text/audio/image/video | 实时文本与 ASR 闭环已实现，音频/图像/视频输出规划 |

## 4. 里程碑总览

| 里程碑 | 交付 | 前置 | 出口摘要 |
| :--- | :--- | :--- | :--- |
| M6 | ONNX 扩散：`onnx.diffusion` + SD-Turbo ONNX；SD1.5/SDXL/FLUX 随后 | 当前基线 | 固定 seed 可复现；CPU/GPU 延迟与显存入库 |
| M7 | 图像编辑与视频生成：edits、LTX/AnimateDiff、异步 job | M6 | 图像编辑可用；2s 视频端到端；取消释放显存 |
| M8 | 实时多通道：TTS、`output.audio/image/video`、预览帧 | M6、M7 | 首音频/首帧延迟入库；过载有界 |
| M9 | 设备矩阵与混合推理：NPU EP、T1–T4、代价模型 | 当前基线 | 设备探测正确；NPU 数值达标；层卸载可用 |
| M10 | 移动端：Tauri iOS/Android、PCM I/O、NPU 放置 | M8、M9 | 三台真机完成语音对话 |
| M11 | 可选引擎与规模化：frink、mistral.rs、多模型驻留 | M7、M9、M10 | 同模型双引擎同协议；换入换出可证 |

## 5. 里程碑详述

### M6 · ONNX 扩散

**范围**：新增 `onnx.diffusion` 实现，用 ONNX Runtime 串联 CLIP 文本编码器、UNet、VAE 解码器与 Euler 调度器，`/v1/images/generations` 使用真实扩散模型。

**首个模型**：`Heliosoph/sd-turbo-onnx`，FP32，512×512，1–4 步，guidance 1.0。该导出使用标准 diffusers ONNX 布局：

```text
text_encoder/model.onnx   input_ids -> last_hidden_state
unet/model.onnx           sample + timestep + encoder_hidden_states -> out_sample
vae_decoder/model.onnx    latent_sample -> sample
tokenizer/vocab.json + merges.txt
scheduler/scheduler_config.json
```

**交付**

- `DiffusionConfig`：组件路径、张量名、调度器参数、latent 通道、VAE scale、默认步数与 guidance。
- `OnnxDiffusionSession`：CLIP tokenize、EulerDiscrete 调度、UNet 循环、VAE 解码、f32 image 输出。
- 固定 seed 的初始 latent，可复现。
- `ImageSlot` 传入 `width/height/steps/guidance/seed` 与 negative prompt。
- 每一步 `ModelEvent::Progress`；取消检查在步边界执行，并终止当前 ORT run。
- OOM 映射 `InsufficientVram`，驱逐后重试一次。
- CPU 与 CUDA 的延迟、峰值内存/显存写入基线。

**出口**

- 512×512、1 步与 4 步各生成有效 PNG；固定 seed 在同设备、同驱动逐字节复现。
- 1 步 CPU 延迟与峰值内存入库；CUDA 可用时 4 步延迟入库。
- 取消在第 k 步返回 `Cancelled`，显存回落到加载前 ±5%。
- 2 个并发请求按 permit 排队。
- `onnx.diffusion` 与 `onnx.graph`、`onnx.pipeline` 的契约测试同时通过。

**后续路径**

- SD1.5 / SDXL ONNX：同一 `onnx.diffusion`，增加 dual text encoder 与 UNet 条件输入。
- FLUX ONNX：增加 T5/CLIP 双编码器、guidance embedding 与 transformer blocks。
- `plakat` 作为 GGUF/Safetensors 图像后端，在 ONNX 路径稳定后接入，用于 SDXL/FLUX GGUF。

**范围边界**：实时预览帧、多 ControlNet、视频。

### M7 · 图像编辑与视频生成

**范围**：`/v1/images/edits`、`/v1/videos/generations`。

**交付**

- img2img、inpaint、outpaint 请求与 mask。
- 视频异步任务：`202 + job_id`、查询、取消、完成事件。
- `candle-video` LTX-Video 2B distilled：2–5s、512–768px、MP4、VAE tiling。
- `plakat` AnimateDiff 路径。
- 文本 + 视频 → 视频编辑能力。
- 任务状态落盘，取消终止 worker 并释放显存。

**出口**

- 2s/512px 视频端到端产出，耗时与峰值显存入库。
- 13B 在 24GB 机型通过，或形成 24GB 预检拒绝结论。
- 取消后 2s 内显存回落；重复提交产生独立 job。
- 视频输出 MP4。

**范围边界**：实时逐帧、端到端实时生成、音频轨。

### M8 · 实时多通道

**范围**：`output.audio`、`output.image` 预览、`output.video` 逐帧、TTS。

**前置探针**

- ONNX 扩散管线是否提供逐步回调与步间取消。支持则原生预览；否则 batch + 进度合成预览帧，并记录 `fallback_reason = "preview_from_batch"`。
- RAVE 类逐帧 ONNX 模型的可得性与每帧预算。支持则接入逐帧管道；否则实时视频采用逐帧滤镜管线。

**交付**

- TTS executor：文本 → `ModelOutput::Audio`，`/v1/audio/speech`。
- `output.audio` 流式 PCM。
- `output.image` 预览帧，`ModelEvent::Progress` 携带 step。
- `output.video` 逐帧 decode → preprocess → inference → encode。
- 丢弃策略：丢旧帧，生成侧持续前进；背压到采集端。
- 预览帧单调序号，客户端据此判定丢帧。

**出口**

- 首音频延迟、首帧延迟 p50/p95 入库。
- 2× 帧率注入下内存有界、延迟有界、丢帧计数可观测。
- 打断 ≤ 3 帧生效；30 分钟浸泡内存稳定。

**范围边界**：端到端实时视频生成。

### M9 · 设备矩阵与混合推理

**范围**：`DeviceInventory`、`PlacementPlan`、EP 矩阵与 T1–T4。

**交付**

- ONNX EP：CUDA、TensorRT、CoreML、DirectML、ROCm、OpenVINO、QNN、NNAPI、Vitis AI、CANN。
- Candle CUDA/Metal。
- `PlacementPolicy`：auto / pinned / strict。
- T1 算子回退、T2 层与 KV 卸载、T3 阶段分置、T4 引擎级组合。
- 代价模型：compute + transfer，prefill 与 decode 分开评分。
- `/v1/models` 与 span 输出 `PlacementPlan`、`fallback_reason`。

**出口**

- Windows NVIDIA、Windows AMD、Windows Copilot+、macOS、Android 正确报告 CPU/GPU/NPU 与 EP。
- NPU 与 CPU 输出在容差内；回退计数可观测。
- 超过单卡显存的模型通过层卸载运行。
- ASR 在 NPU、LLM 在 GPU、TTS 在 CPU 的编排路径延迟达标。
- `strict` 对未覆盖算子返回 `PlacementUnsatisfiable`。
- 取消与驱逐释放设备资源。

**范围边界**：训练、跨节点分布式、AOT 编译。

### M10 · 移动端

**范围**：Tauri 2 iOS/Android、PCM 音频 I/O、按机型放置。

**交付**

- Tauri 2 移动目标；麦克风 PCM 回调；原生音频播放；loopback API；权限声明。
- 编排式实时会话作为基线。
- `PlacementPlan` 将 ASR/TTS 放 NPU，LLM 放 GPU/CPU。
- 设备矩阵：低端 Android、旗舰 Android、iPhone。

**出口**

- 三台真机各完成 5 轮语音对话；首音频延迟 p95 入库。
- 旗舰机型将阶段放到 NPU；回退原因可查。
- 前后台切换、来电中断后会话恢复。
- 连续 15 分钟运行稳定；安装包与峰值内存入库。

**范围边界**：移动端原生双工、移动端图像/视频生成。

### M11 · 可选引擎与规模化

**交付**

- `frink` GGUF 可选引擎：契约测试、IQ/MXFP4、并发吞吐。
- `mistral.rs` Safetensors 可选引擎评估：ISQ、paged attention、TTFT/tok/s。
- Registry 多模型 LRU、显存预算、预热池、租约公平性、排队上限。
- 原生双工在移动端 NPU/GPU 的逐机型验证。
- 安装器、模型下载管理、可选遥测。

**出口**

- 同一 GGUF 模型在 Candle 与 frink 间切换，接口与事件流一致。
- 显存预算下 3 个模型驻留，换入换出可复现。
- 并发压测输出 p95 与错误率；超限请求快速失败。
- 至少一台旗舰机型跑通原生双工。

**范围边界**：训练、多节点分布式推理。

## 6. 横切轨道

| 轨道 | 内容 | 门禁 |
| :--- | :--- | :--- |
| Manifest 与供应链 | capability、sha256、fetch 脚本、许可证 | 清单解析与校验通过 |
| 构建与 EP 矩阵 | CPU 默认；EP feature 编译检查 | PR 编译全部组合；硬件 runner 跑对应用例 |
| 契约测试 | 能力、取消、流式、错误码、放置 | 新引擎通过契约测试 |
| 可观测性 | 分层 span、延迟直方图、队列/会话计量 | 每条新增路径带 span |
| 性能预算 | `bench/baseline.json` + `bench compare` | 回归 +10% 报警、+25% 阻塞 |
| 文档 | DESIGN、ARCHITECTURE、ROADMAP、ADR | 行为变更与文档同批合入 |

## 7. 决策门

| 门 | 时点 | 问题 | 分支 |
| :--- | :--- | :--- | :--- |
| G1 | M8 前 | ONNX 扩散管线提供逐步回调与步间取消？ | 是：原生预览帧；否：batch + 进度合成预览帧 |
| G2 | M8 前 | 存在满足帧预算的 RAVE 类 ONNX 模型？ | 是：逐帧管道；否：逐帧滤镜管线 |
| G3 | M11 | `mistral.rs` 的 TTFT/tok/s/显存支持采纳？ | 是：Safetensors 可选引擎；否：保留 Candle |
| G4 | M7 | LTX 13B 在 24GB 可用？ | 是：声明支持；否：预检拒绝并写入矩阵 |
| G5 | M11 | `frink` 在契约、吞吐、量化覆盖领先？ | 是：GGUF 默认可选；否：保留可选 |
| G6 | M9 | 目标 NPU EP 在目标设备可用且数值达标？ | 是：进入默认放置；否：回退 GPU/CPU 并记录原因 |

## 8. 风险登记

| 风险 | 触发信号 | 应对 |
| :--- | :--- | :--- |
| ONNX 扩散图张量名不一致 | 加载后输入/输出名与预期不符 | manifest 显式记录每个组件的输入/输出名；加载时校验 |
| 调度器数值偏差 | 同 seed 输出与 diffusers 不一致 | 以 diffusers EulerDiscrete 为参考，固定 sigmas/timesteps 单测 |
| SD-Turbo 许可限制 | 商业使用门槛 | 模型目录随附许可说明；不默认分发权重 |
| plakat API 漂移 | 升级后编译失败 | 锁定版本 + adapter 隔离；ONNX 主路径不受影响 |
| 流式扩散回调缺失 | G1 结论为否 | batch + 进度合成预览帧 |
| RAVE 模型缺失 | G2 结论为否 | 实时视频采用逐帧滤镜 |
| FLUX 显存超标 | 峰值超过 16GB | 预检拒绝；NF4/更高量化作为可选 |
| LTX 13B 资源超标 | OOM 或超时 | 2B distilled 为基线；13B 写机型矩阵 |
| 打断语义侵入引擎 | 帧逻辑进入引擎 crate | 架构测试报警；`InterruptPolicy` 留在 Router |
| EP 组合膨胀 | CI 超时或链接错误 | 白名单组合；PR 编译全部，硬件 runner 跑子集 |
| NPU 算子覆盖不足 | 回退比例超阈值 | T1 回退计数 + T3 阶段分置 |
| 跨设备迁移开销 | p95 超基线 | 代价模型纳入 transfer；热点阶段 pinned |
| 驱动碎片化 | 同型号行为不一致 | EP/驱动版本写入清单与基线；启动探测 |
| 取消泄漏显存 | 浸泡 RSS/显存漂移 | `CancellationToken` + 租约；浸泡门禁 |
| 移动端音频延迟 | 首音频 p95 超预算 | 编排式基线先行；按机型矩阵逐台承诺 |

## 9. 完成定义

里程碑完成需要：

1. 代码合入 `main`，默认构建与全 feature 构建通过。
2. 出口场景有脚本或集成测试，在 CI 或硬件 runner 运行。
3. 新增契约覆盖能力、取消、流式与错误路径测试。
4. 基线与预算写入 `bench/baseline.json`，含设备与驱动信息。
5. 用户文档、`fallback_reason` 与支持矩阵同批更新。
6. 每条新增路径带指标与开关。
