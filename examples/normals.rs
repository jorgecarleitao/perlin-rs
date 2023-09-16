use std::path::Path;

use noise_perlin::perlin_2d_derivative;

static SQRT_2_OVER_2: f32 = 0.70710678118;

fn main() {
    let width = 256;
    let height = width;

    let mut values = vec![0.0f32; height * width];
    let mut normals = vec![0.0f32; height * width * 3];

    let scale = 2.0;
    let offset = 0.0;

    (values
        .iter_mut()
        .enumerate()
        .zip(normals.chunks_exact_mut(3)))
    .for_each(|((i, pixel), rgb)| {
        let x = (i / width) as f32;
        let x = x / width as f32;
        let y = (i % height) as f32;
        let y = y / height as f32;

        let (v, d) = perlin_2d_derivative(x * scale + offset, y * scale + offset);
        *pixel = v;

        // map the normal to rgb channels
        // first, compute the normal from dx and dy using cross product
        // p = (x, y, z(x,y))
        // dp/dx = (1, 0, dz/dx(x,y))
        // dp/dy = (0, 1, dz/dy(x,y))
        // => normal = (1, 0, dz/dx(x,y)) x (0, 1, dz/dy(x,y))
        // = (-dz/dx(x,y), -dz/dy(x,y), 1)
        rgb[0] = -d[0] * SQRT_2_OVER_2;
        rgb[1] = -d[1] * SQRT_2_OVER_2;
        rgb[2] = 1.0;
    });

    let max = values.iter().cloned().fold(0. / 0., f32::max);
    let min = values.iter().cloned().fold(0. / 0., f32::min);
    let values = values
        .into_iter()
        .map(|x| ((x - min) / (max - min) * 255.0) as u8)
        .collect::<Vec<_>>();

    let max = normals.iter().cloned().fold(0. / 0., f32::max);
    let min = normals.iter().cloned().fold(0. / 0., f32::min);
    let normals = normals
        .into_iter()
        .map(|x| ((x - min) / (max - min) * 255.0) as u8)
        .collect::<Vec<_>>();
    println!("{min} {max}");

    image::save_buffer(
        &Path::new("image.png"),
        &values,
        width as u32,
        height as u32,
        image::ColorType::L8,
    )
    .unwrap();

    image::save_buffer(
        &Path::new("normal.png"),
        &normals,
        width as u32,
        height as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
}
