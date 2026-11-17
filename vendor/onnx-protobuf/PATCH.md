# uniinfer-onnx-protobuf

`uniinfer-onnx-protobuf` 0.2.3 generated `src/proto/onnx.rs` with a check for
`protobuf::VERSION_3_4_0`. That runtime is affected by RUSTSEC-2024-0437.

This vendored copy updates the generated runtime version check to
`protobuf::VERSION_3_7_2`, removes a removed `box_pointers` lint, and allows
the Rust 2024 lifetime-elision lint emitted by the generated helpers. The
message definitions are unchanged.

The package is a workspace member and `uniinfer-engine-onnx` depends on it by
package name. Keeping it as a regular path dependency, instead of a
`[patch.crates-io]` entry, lets `cargo outdated` copy the complete workspace
manifest set into its temporary project.
