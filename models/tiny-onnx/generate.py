#!/usr/bin/env python3
"""Generate the tiny ONNX byte model used by M0 smoke tests.

The graph maps each input id to the one-hot of `(id + 1) mod 256`, so greedy
decoding advances one byte per step. It exercises the ONNX engine, tensor
plumbing and cancellation; it is not a language model.

The ONNX protobuf is emitted directly with the standard library, so fixture
generation needs no Python packages.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
VOCAB = 256
INT64 = 7
FLOAT = 1
UINT8 = 2


def varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def key(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def vint(field: int, value: int) -> bytes:
    return key(field, 0) + varint(value)


def string(field: int, value: str) -> bytes:
    raw = value.encode()
    return key(field, 2) + varint(len(raw)) + raw


def message(field: int, payload: bytes) -> bytes:
    return key(field, 2) + varint(len(payload)) + payload


def floats(field: int, values: list[float]) -> bytes:
    return b"".join(key(field, 5) + struct.pack("<f", value) for value in values)


def tensor(
    name: str,
    dtype: int,
    dims: list[int],
    int_values: list[int] | None = None,
    float_values: list[float] | None = None,
) -> bytes:
    body = b"".join(vint(1, dim) for dim in dims)
    body += vint(2, dtype)
    body += b"".join(vint(7, value) for value in (int_values or []))
    body += floats(4, float_values or [])
    body += string(8, name)
    return body


def dimension(value: int | None = None, param: str | None = None) -> bytes:
    if param is not None:
        return string(1, param)
    return vint(1, value or 0)


def value_info(name: str, dtype: int, dims: list[tuple[str, int | str]]) -> bytes:
    shape = b"".join(
        message(1, dimension(param=value) if kind == "param" else dimension(value=value))
        for kind, value in dims
    )
    tensor_type = vint(1, dtype) + message(2, shape)
    return string(1, name) + message(2, message(1, tensor_type))


def node(op_type: str, inputs: list[str], outputs: list[str], **attrs: int) -> bytes:
    body = b"".join(string(1, name) for name in inputs)
    body += b"".join(string(2, name) for name in outputs)
    body += string(4, op_type)
    for name, value in attrs.items():
        attr = string(1, name) + vint(3, value) + vint(20, 2)
        body += message(5, attr)
    return body


def build_model() -> bytes:
    nodes = [
        node("Add", ["input_ids", "one"], ["shifted"]),
        node("Mod", ["shifted", "depth"], ["wrapped"]),
        node("OneHot", ["wrapped", "depth", "values"], ["logits"], axis=-1),
    ]
    initializers = [
        tensor("one", INT64, [], int_values=[1]),
        tensor("depth", INT64, [], int_values=[VOCAB]),
        tensor("values", FLOAT, [2], float_values=[0.0, 1.0]),
    ]
    graph = b"".join(message(1, value) for value in nodes)
    graph += string(2, "tiny-byte-lm")
    graph += b"".join(message(5, value) for value in initializers)
    graph += message(
        11,
        value_info("input_ids", INT64, [("value", 1), ("param", "seq")]),
    )
    graph += message(
        12,
        value_info(
            "logits",
            FLOAT,
            [("value", 1), ("param", "seq"), ("value", VOCAB)],
        ),
    )

    opset = vint(2, 13)
    model = vint(1, 8)
    model += string(2, "uniinfer")
    model += message(7, graph)
    model += message(8, opset)
    return model


def build_vision_model() -> bytes:
    """Tiny graph with a uint8 image input, used for API vision plumbing."""
    nodes = [
        node("Size", ["image"], ["image_size"]),
        node("Mul", ["image_size", "zero"], ["zero_offset"]),
        node("Add", ["input_ids", "zero_offset"], ["offset_ids"]),
        node("Add", ["offset_ids", "one"], ["shifted"]),
        node("Mod", ["shifted", "depth"], ["wrapped"]),
        node("OneHot", ["wrapped", "depth", "values"], ["logits"], axis=-1),
    ]
    initializers = [
        tensor("zero", INT64, [], int_values=[0]),
        tensor("one", INT64, [], int_values=[1]),
        tensor("depth", INT64, [], int_values=[VOCAB]),
        tensor("values", FLOAT, [2], float_values=[0.0, 1.0]),
    ]
    graph = b"".join(message(1, value) for value in nodes)
    graph += string(2, "tiny-vision-lm")
    graph += b"".join(message(5, value) for value in initializers)
    graph += message(
        11,
        value_info("input_ids", INT64, [("value", 1), ("param", "seq")]),
    )
    graph += message(11, value_info("image", UINT8, [("param", "n")]))
    graph += message(
        12,
        value_info(
            "logits",
            FLOAT,
            [("value", 1), ("param", "seq"), ("value", VOCAB)],
        ),
    )

    opset = vint(2, 13)
    model = vint(1, 8)
    model += string(2, "uniinfer")
    model += message(7, graph)
    model += message(8, opset)
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "model.onnx"
    model_path.write_bytes(build_model())

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = f"""schema_version = 1
id = "tiny-byte-onnx"
format = "onnx"
quantization = "fp32"
modalities = ["text"]
streaming_capable = false
context_len = 64
tokenizer = "bytes"
min_vram = "0GiB"

[placement]
policy = "auto"
prefer = ["npu", "gpu", "cpu"]
allow_cpu_fallback = true

[text]
vocab_size = {VOCAB}

[[files]]
path = "model.onnx"
sha256 = "{digest}"
"""
    (OUT_DIR / "manifest.toml").write_text(manifest, encoding="utf-8", newline="\n")
    (OUT_DIR / "sha256.txt").write_text(
        f"{digest}  model.onnx\n", encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {model_path.name} "
        f"({model_path.stat().st_size} bytes, sha256 {digest[:12]}...)"
    )

    vision_path = OUT_DIR / "vision.onnx"
    vision_path.write_bytes(build_vision_model())
    vision_digest = hashlib.sha256(vision_path.read_bytes()).hexdigest()
    vision_manifest = f"""schema_version = 1
id = "tiny-vision-onnx"
format = "onnx"
quantization = "fp32"
modalities = ["text", "image"]
streaming_capable = false
context_len = 64
tokenizer = "bytes"
min_vram = "0GiB"

[placement]
policy = "auto"
prefer = ["npu", "gpu", "cpu"]
allow_cpu_fallback = true

[text]
vocab_size = {VOCAB}

[vision]
architecture = "onnx_vlm"
image_token = "<image>"

[[files]]
path = "vision.onnx"
sha256 = "{vision_digest}"
"""
    (OUT_DIR / "manifest-vision.toml").write_text(
        vision_manifest, encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {vision_path.name} "
        f"({vision_path.stat().st_size} bytes, sha256 {vision_digest[:12]}...)"
    )


if __name__ == "__main__":
    main()
