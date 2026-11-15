//! Qwen3-VL image preprocessing.

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use uniinfer_core::{EngineError, Result};

#[derive(Debug, Clone)]
pub(crate) struct PreprocessedImage {
    pub(crate) pixel_values: Vec<f32>,
    pub(crate) grid_thw: [i64; 3],
    pub(crate) image_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ImageProcessor {
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    shortest_edge: u32,
    longest_edge: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImageProcessor {
    pub(crate) fn qwen3_vl(config_path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read_to_string(config_path).map_err(|error| {
            EngineError::ManifestInvalid(format!("preprocessor config: {error}"))
        })?;
        let value: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("preprocessor config: {error}"))
        })?;
        let size = &value["size"];
        Ok(Self {
            patch_size: value["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: value["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            merge_size: value["merge_size"].as_u64().unwrap_or(2) as usize,
            shortest_edge: size["shortest_edge"].as_u64().unwrap_or(65_536) as u32,
            longest_edge: size["longest_edge"].as_u64().unwrap_or(16_777_216) as u32,
            mean: array(&value["image_mean"], 0.5),
            std: array(&value["image_std"], 0.5),
        })
    }

    pub(crate) fn preprocess(&self, bytes: &[u8]) -> Result<PreprocessedImage> {
        let image = image::load_from_memory(bytes)
            .map_err(|error| EngineError::InvalidAudio(format!("image: {error}")))?;
        let (width, height) = image.dimensions();
        let factor = (self.patch_size * self.merge_size) as u32;
        let (new_width, new_height) =
            smart_resize(width, height, factor, self.shortest_edge, self.longest_edge);
        let image = image.resize_exact(new_width, new_height, FilterType::Triangle);
        let rgb: RgbImage = DynamicImage::ImageRgb8(image.to_rgb8()).into_rgb8();
        let grid_h = (new_height / self.patch_size as u32) as i64;
        let grid_w = (new_width / self.patch_size as u32) as i64;
        let temporal = self.temporal_patch_size;
        let patch = self.patch_size;
        let mut pixels =
            Vec::with_capacity((grid_h * grid_w) as usize * 3 * temporal * patch * patch);
        for ph in 0..grid_h as u32 {
            for pw in 0..grid_w as u32 {
                for channel in 0..3usize {
                    for _ in 0..temporal {
                        for y in 0..patch as u32 {
                            for x in 0..patch as u32 {
                                let px =
                                    rgb.get_pixel(pw * patch as u32 + x, ph * patch as u32 + y);
                                let value = f32::from(px[channel]) / 255.0;
                                pixels.push((value - self.mean[channel]) / self.std[channel]);
                            }
                        }
                    }
                }
            }
        }
        let image_tokens =
            (grid_h as usize / self.merge_size) * (grid_w as usize / self.merge_size);
        Ok(PreprocessedImage {
            pixel_values: pixels,
            grid_thw: [1, grid_h, grid_w],
            image_tokens,
        })
    }
}

fn array(value: &serde_json::Value, default: f32) -> [f32; 3] {
    let values = value.as_array();
    std::array::from_fn(|index| {
        values
            .and_then(|values| values.get(index))
            .and_then(serde_json::Value::as_f64)
            .map_or(default, |value| value as f32)
    })
}

fn smart_resize(
    width: u32,
    height: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> (u32, u32) {
    let mut width = width.max(1);
    let mut height = height.max(1);
    if width * height > max_pixels {
        let beta = ((width * height) as f64 / max_pixels as f64).sqrt();
        width = ((width as f64 / beta) as u32).max(factor);
        height = ((height as f64 / beta) as u32).max(factor);
    }
    if width * height < min_pixels {
        let beta = (min_pixels as f64 / (width * height) as f64).sqrt();
        width = ((width as f64 * beta) as u32).max(factor);
        height = ((height as f64 * beta) as u32).max(factor);
    }
    width = (width as f32 / factor as f32).round() as u32 * factor;
    height = (height as f32 / factor as f32).round() as u32 * factor;
    (width.max(factor), height.max(factor))
}

impl ImageProcessor {
    pub(crate) fn merge_size(&self) -> usize {
        self.merge_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocesses_solid_image_to_patches() {
        let mut image = RgbImage::new(64, 64);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgb([255, 0, 0]);
        }
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("png");
        let processor = ImageProcessor {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            shortest_edge: 4096,
            longest_edge: 4096,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let output = processor.preprocess(&bytes).expect("preprocess");
        assert_eq!(output.grid_thw, [1, 4, 4]);
        assert_eq!(output.image_tokens, 4);
        assert_eq!(output.pixel_values.len(), 16 * 3 * 2 * 16 * 16);
    }
}
