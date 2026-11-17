use std::path::Path;

use uniinfer_backends::ModelOutput;
use uniinfer_core::{CancelToken, ModelManifest};
use uniinfer_service::{ImageGenerationSpec, ModelExecutor, ModelRequest, Runtime};

#[test]
fn diffusion_slot_returns_png() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-diffusion");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let runtime = Runtime::new().expect("runtime");
    let slot = runtime
        .load_image_manifest(manifest, &dir)
        .expect("load image slot");
    let output = slot
        .execute(
            ModelRequest::Image {
                spec: ImageGenerationSpec {
                    prompt: "a red cube".into(),
                    negative_prompt: None,
                    width: 64,
                    height: 64,
                    steps: 1,
                    guidance: 1.0,
                    seed: Some(7),
                },
            },
            &CancelToken::new(),
            &mut |_| {},
        )
        .expect("generate");
    let ModelOutput::Image(image) = output else {
        panic!("expected image");
    };
    assert_eq!(image.mime, "image/png");
    assert_eq!((image.width, image.height), (64, 64));
    assert!(image.bytes.starts_with(b"\x89PNG"));
}
