use std::path::Path;

use uniinfer_core::ModelManifest;

/// A real-model fixture is usable only when the manifest and every declared
/// file exist. Weight files are downloaded, not tracked, so CI checkouts may
/// have the manifest alone.
pub fn fixture_ready(manifest: &Path) -> bool {
    let Ok(loaded) = ModelManifest::load(manifest) else {
        return false;
    };
    let Some(base) = manifest.parent() else {
        return false;
    };
    loaded
        .files
        .iter()
        .all(|file| base.join(&file.path).is_file())
}
