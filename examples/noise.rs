use std::path::Path;

use noise_perlin::perlin_2d;

static SQRT_2_OVER_2: f32 = 0.70710678118;

fn main() {
    let width = 256;
    let height = width;

    let mut bytes = vec![0u8; height * width];

    let scale = 1.0;
    let offset = 0.0;

    (bytes.iter_mut().enumerate()).for_each(|(i, pixel)| {
        let x = (i / width) as f32;
        let x = x / width as f32;
        let y = (i % height) as f32;
        let y = y / height as f32;

        let mut v = perlin_2d(x * scale + offset, y * scale + offset);
        v = v * SQRT_2_OVER_2 + 0.5;
        *pixel = (v * 255.0) as u8;
    });

    image::save_buffer(
        &Path::new("image.png"),
        &bytes,
        width as u32,
        height as u32,
        image::ColorType::L8,
    )
    .unwrap();
}
