#!/usr/bin/env python3
"""Generate deterministic ONNX components for the diffusion smoke test."""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent


def tensor(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def save(name, nodes, inputs, outputs, inits):
    graph = helper.make_graph(nodes, name, inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = OUT / f"{name}.onnx"
    onnx.save(model, path)
    return path


# Text encoder: project each token id to an 8-wide vector.
text_input = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 77])
text_output = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 77, 8])
text_nodes = [
    helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["ids_f", "axis"], ["ids_3d"]),
    helper.make_node("Expand", ["ids_3d", "shape"], ["last_hidden_state"]),
]
save("text_encoder", text_nodes, [text_input], [text_output], [
    tensor("axis", np.array([2], dtype=np.int64)),
    tensor("shape", np.array([1, 77, 8], dtype=np.int64)),
])

# UNet: keep the latent shape, add the timestep to each element.
unet_sample = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [1, 4, 8, 8])
unet_time = helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])
unet_encoder = helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, [1, 77, 8])
unet_output = helper.make_tensor_value_info("out_sample", TensorProto.FLOAT, [1, 4, 8, 8])
save("unet", [helper.make_node("Add", ["sample", "timestep"], ["out_sample"])],
     [unet_sample, unet_time, unet_encoder], [unet_output], [])

# VAE: collapse channels, resize the latent spatially and expand to RGB.
vae_input = helper.make_tensor_value_info("latent_sample", TensorProto.FLOAT, [1, 4, 8, 8])
vae_output = helper.make_tensor_value_info("sample", TensorProto.FLOAT, [1, 3, 64, 64])
vae_nodes = [
    helper.make_node("ReduceMean", ["latent_sample"], ["mean"], axes=[1], keepdims=1),
    helper.make_node("Resize", ["mean", "", "scales"], ["scaled"], mode="nearest"),
    helper.make_node("Expand", ["scaled", "rgb_shape"], ["sample"]),
]
save("vae_decoder", vae_nodes, [vae_input], [vae_output], [
    tensor("scales", np.array([1.0, 1.0, 8.0, 8.0], dtype=np.float32)),
    tensor("rgb_shape", np.array([1, 3, 64, 64], dtype=np.int64)),
])

rows = []
for path in sorted(OUT.glob("*.onnx")):
    rows.append(f'[[files]]\npath = "{path.name}"\nsha256 = "{hashlib.sha256(path.read_bytes()).hexdigest()}"')
(OUT / "sha256.txt").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(OUT.glob("*.onnx"))) + "\n")
manifest = f'''id = "tiny-diffusion-onnx"
runtime = "onnx"
implementation = "onnx.diffusion"
format = "onnx"
quantization = "fp32"
streaming_capable = false
context_len = 77
tokenizer = "bytes"
min_vram = "0GiB"

[placement]
policy = "auto"
prefer = ["npu", "gpu", "cpu"]
allow_cpu_fallback = true

[text]
vocab_size = 256

[image]
width = 64
height = 64

[image.diffusion]
max_length = 77
pad_token = 0
latent_channels = 4
vae_scale = 1.0
steps = 1
guidance = 1.0
text_encoder = {{ path = "text_encoder.onnx", output = "last_hidden_state" }}
unet = {{ path = "unet.onnx", output = "out_sample" }}
vae_decoder = {{ path = "vae_decoder.onnx", output = "sample" }}

{chr(10).join(rows)}

[[capabilities]]
name = "text_to_image"
operation = "generate"
required_inputs = ["text"]
outputs = ["image"]
mode = "batch"
'''
(OUT / "manifest.toml").write_text(manifest)
