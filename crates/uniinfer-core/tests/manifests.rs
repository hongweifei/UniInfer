use std::path::PathBuf;

use uniinfer_core::ModelManifest;

#[test]
fn repository_manifests_parse_and_validate() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut manifests = Vec::new();
    collect(&root, &mut manifests);
    manifests.sort();
    assert!(
        !manifests.is_empty(),
        "no manifests under {}",
        root.display()
    );
    for path in manifests {
        ModelManifest::load(&path).unwrap_or_else(|error| {
            panic!("{}: {error}", path.display());
        });
    }
}

fn collect(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("manifest") && name.ends_with(".toml"))
        {
            out.push(path);
        }
    }
}
