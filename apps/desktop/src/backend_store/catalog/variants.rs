use super::GithubAsset;

pub(super) struct WindowsVariant<'a> {
    pub(super) key: String,
    pub(super) label: String,
    pub(super) asset: &'a GithubAsset,
    pub(super) devices: Vec<uniinfer_core::DeviceClass>,
}

pub(super) fn windows_variants<'a>(
    assets: &'a [GithubAsset],
    prefix: &str,
) -> Vec<WindowsVariant<'a>> {
    let mut variants = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for asset in assets {
        let name = asset.name.to_ascii_lowercase();
        if !name.starts_with(prefix)
            || !name.ends_with("-x64.zip")
            || !name.contains("-bin-win-")
            || name.contains("arm64")
        {
            continue;
        }
        let Some((_, rest)) = name.split_once("-bin-win-") else {
            continue;
        };
        let Some(key) = rest.strip_suffix("-x64.zip") else {
            continue;
        };
        if key.is_empty() || !seen.insert(key.to_owned()) {
            continue;
        }
        let key_tokens = key
            .split(|character: char| !character.is_ascii_alphanumeric())
            .collect::<Vec<_>>();
        let devices = if key_tokens.contains(&"cpu") {
            vec![uniinfer_core::DeviceClass::Cpu]
        } else if key_tokens.contains(&"npu") {
            vec![uniinfer_core::DeviceClass::Npu]
        } else {
            vec![uniinfer_core::DeviceClass::Gpu]
        };
        variants.push(WindowsVariant {
            key: key.to_owned(),
            label: variant_label(key),
            asset,
            devices,
        });
    }
    variants
}

fn variant_label(key: &str) -> String {
    key.split('-')
        .flat_map(variant_label_parts)
        .collect::<Vec<_>>()
        .join(" ")
}

fn variant_label_parts(part: &str) -> Vec<String> {
    if part
        .chars()
        .all(|character| character.is_ascii_digit() || character == '.')
    {
        return vec![part.to_owned()];
    }
    if let Some(index) = part.find(|character: char| character.is_ascii_digit()) {
        let (name, version) = part.split_at(index);
        if !name.is_empty() && !version.is_empty() {
            let mut parts = variant_label_parts(name);
            parts.push(version.to_owned());
            return parts;
        }
    }
    let label = match part {
        "cpu" => "CPU",
        "npu" => "NPU",
        "cuda" => "CUDA",
        "rocm" => "ROCm",
        "vulkan" => "Vulkan",
        "sycl" => "SYCL",
        "openvino" => "OpenVINO",
        "opencl" => "OpenCL",
        "adreno" => "Adreno",
        "metal" => "Metal",
        "dml" | "directml" => "DirectML",
        other => {
            let mut chars = other.chars();
            return match chars.next() {
                Some(first) => vec![first.to_uppercase().collect::<String>() + chars.as_str()],
                None => Vec::new(),
            };
        }
    };
    vec![label.to_owned()]
}

pub(super) fn variant_slug(key: &str) -> String {
    key.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_ascii_lowercase()
}

fn cuda_version(key: &str) -> Option<String> {
    let key = key.to_ascii_lowercase();
    key.strip_prefix("cuda")
        .or_else(|| key.strip_prefix("cu"))
        .map(|version| version.trim_start_matches(['-', '_']).to_owned())
        .filter(|version| !version.is_empty())
}

pub(super) fn cudart_asset<'a>(assets: &'a [GithubAsset], key: &str) -> Option<&'a GithubAsset> {
    let version = cuda_version(key)?;
    assets.iter().find(|asset| {
        let name = asset.name.to_ascii_lowercase();
        if !name.starts_with("cudart")
            || !name.ends_with("-x64.zip")
            || !name.contains("-bin-win-")
            || name.contains("arm64")
        {
            return false;
        }
        let Some((_, rest)) = name.split_once("-bin-win-") else {
            return false;
        };
        let Some(cudart_key) = rest.strip_suffix("-x64.zip") else {
            return false;
        };
        cuda_version(cudart_key).as_deref() == Some(version.as_str())
    })
}
