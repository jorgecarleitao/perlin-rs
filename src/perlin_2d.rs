use super::*;

#[inline]
fn grad_2d_1(hash: usize) -> [f32; 2] {
    // in 2D, we only select from 4 different gradients.
    // http://riven8192.blogspot.com/2010/08/calculate-perlinnoise-twice-as-fast.html
    match hash % 4 {
        0 => [1.0, 1.0],
        1 => [-1.0, 1.0],
        2 => [1.0, -1.0],
        3 => [-1.0, -1.0],
        _ => [0.0, 0.0], // unreachable
    }
}

#[inline]
fn grad_2d(hash: usize, x: f32, y: f32) -> f32 {
    let der = grad_2d_1(hash);
    der[0] * x + der[1] * y
}

#[inline]
fn eval(x: f32, y: f32, g00: usize, g10: usize, g01: usize, g11: usize) -> f32 {
    // compute the gradients
    // note: each corner has its own independent direction (derived from the permutation table)
    // g00 represents the dot product of (x,y) with one of the directions assigned to the corner `(0,0)` (e.g. `(1,1)`)
    let g00 = grad_2d(g00, x, y); // (x,y) - (0,0)
    let g10 = grad_2d(g10, x - 1.0, y); // (x,y) - (1,0)

    let g01 = grad_2d(g01, x, y - 1.0); // (x,y) - (0,1)
    let g11 = grad_2d(g11, x - 1.0, y - 1.0); // (x,y) - (1,1)

    // smoothed x (continuous second derivative)
    let u = fade(x);
    // smoothed y (continuous second derivative)
    let v = fade(y);

    // g00 + f(x) * (g10 - g00) + f(y) * (g01 + f(x) * (g11 - g01) - (g00 + f(x) * (g10 - g00)))
    // g00 + f(x) * (g10 - g00) + f(y) * (g01 - g00 + f(x) * ((g11 - g01) - (g10 - g00)))
    lerp(v, lerp(u, g00, g10), lerp(u, g01, g11))
    // in particular
    // x = 0 and y = 0 => g00 => 0
    // x = 1 and y = 0 => g10 => 0
    // x = 0 and y = 1 => g01 => 0
    // x = 1 and y = 1 => g11 => 0
    // i.e. noise at each corner equals to zero
    // x = 0.5 and y = 0.5 => (g00+g10+g01+g11)/4
    // i.e. noise at the center equals to the average of the gradients
}

/// Returns the noise together with the two partial derivatives
#[inline]
fn eval_grad(x: f32, y: f32, g00: usize, g10: usize, g01: usize, g11: usize) -> (f32, [f32; 2]) {
    // compute the gradients
    // note: each corner has its own independent direction (derived from the permutation table)
    // g00 represents the dot product of (x,y) with one of the directions assigned to the corner `(0,0)` (e.g. `(1,1)`)
    let d00 = grad_2d_1(g00);
    let d10 = grad_2d_1(g10);
    let d01 = grad_2d_1(g01);
    let d11 = grad_2d_1(g11);

    let g00 = grad_2d(g00, x, y); // (x,y) - (0,0)
    let g10 = grad_2d(g10, x - 1.0, y); // (x,y) - (1,0)

    let g01 = grad_2d(g01, x, y - 1.0); // (x,y) - (0,1)
    let g11 = grad_2d(g11, x - 1.0, y - 1.0); // (x,y) - (1,1)

    // smoothed x (continuous second derivative)
    let u = fade(x);
    // smoothed y (continuous second derivative)
    let v = fade(y);

    // lerp(t,a,b) = a + t * (b - a)
    // g00 + f(x) * (g10 - g00) + f(y) * (g01 + f(x) * (g11 - g01) - (g00 + f(x) * (g10 - g00)))
    let n = lerp(v, lerp(u, g00, g10), lerp(u, g01, g11));

    let u_prime = fade_prime(x);
    let v_prime = fade_prime(y);

    // g00(x,y) + f(x) * (g10(x,y) - g00(x,y)) + f(y) * (g01(x,y) + f(x) * (g11(x,y) - g01(x,y)) - (g00(x,y) + f(x) * (g10(x,y) - g00(x,y))))
    // g00(x,y) + f(x) * (g10(x,y) - g00(x,y)) + f(y) * (g01(x,y) - g00(x,y) + f(x) * ((g11(x,y) - g01(x,y)) - (g10(x,y) - g00(x,y))))
    // a1(x,y)  + f(x) * (a2(x,y)  - a1(x,y) ) + f(y) * (a3(x,y)  - a1(x,y)  + f(x) * ((a4(x,y)  - a3(x,y) ) - (a2(x,y)  - a1(x,y) )))
    // https://www.wolframalpha.com/input?i=D%5Ba1%28x%2Cy%29+%2B+f%28x%29+*+%28a2%28x%2Cy%29+-+a1%28x%2Cy%29%29+%2B+f%28y%29+*+%28a3%28x%2Cy%29+-+a1%28x%2Cy%29+%2B+f%28x%29+*+%28%28a4%28x%2Cy%29+-+a3%28x%2Cy%29%29+-+%28a2%28x%2Cy%29+-+a1%28x%2Cy%29%29%29%29%2Cx%5D
    let dn_dx = v
        * (u * (d00[0] - d10[0] - d01[0] + d11[0]) - d00[0]
            + u_prime * (g00 - g10 - g01 + g11)
            + d01[0])
        + u * (d10[0] - d00[0])
        + d00[0]
        + u_prime * (g10 - g00);

    let dn_dy = v * ((u - 1.0) * d00[1] - u * (d10[1] + d01[1] - d11[1]) + d01[1])
        + u * (d10[1] - d00[1])
        + d00[1]
        + v_prime * (u * (g00 - g10 - g01 + g11) - g00 + g01);

    (n, [dn_dx, dn_dy])
}

#[inline]
fn perlin_permutations(x0: usize, y0: usize) -> (usize, usize, usize, usize) {
    let gx = x0 % 256;
    let gy = y0 % 256;

    // derive a permutation from the indices.
    // This behaves like a weak hash
    // note that the +1's must be consistent with the relative position in the box
    let a00 = gy + PERM[gx];
    let a10 = gy + PERM[gx + 1];

    let g00 = PERM[a00];
    let g10 = PERM[a10];
    let g01 = PERM[1 + a00];
    let g11 = PERM[1 + a10];

    (g00, g10, g01, g11)
}

/// Returns the evaluation of perlin noise at position (x, y)
/// This function does not allocate
/// It uses the improved implementation of perlin noise
/// whose reference implementation is available here: https://mrl.cs.nyu.edu/~perlin/noise/
/// The modifications are:
/// * made it 2d, ignoring the z coordinate
/// * the grad computation was modified
pub fn perlin_2d(mut x: f32, mut y: f32) -> f32 {
    let x0 = x as usize;
    let y0 = y as usize;

    x -= x0 as f32;
    y -= y0 as f32;
    // at this point (x, y) is bounded to [0, 1]
    debug_assert!((x >= 0.0) && (x <= 1.0) && (y >= 0.0) && (y <= 1.0));

    let (g00, g10, g01, g11) = perlin_permutations(x0, y0);

    eval(x, y, g00, g10, g01, g11)
}

/// Returns the perlin noise at position and its respective partial derivatives
/// along x and y directions
/// I.e. the first value equals `f(x,y) = perlin_2d(x,y)`, the second is `[df(x,y)/dx, df(x,y)/dy]`.
pub fn perlin_2d_derivative(mut x: f32, mut y: f32) -> (f32, [f32; 2]) {
    let x0 = x as usize;
    let y0 = y as usize;

    x -= x0 as f32;
    y -= y0 as f32;
    // at this point (x, y) is bounded to [0, 1]
    debug_assert!((x >= 0.0) && (x <= 1.0) && (y >= 0.0) && (y <= 1.0));

    let (g00, g10, g01, g11) = perlin_permutations(x0, y0);

    eval_grad(x, y, g00, g10, g01, g11)
}

#[cfg(test)]
mod tests {
    #[test]
    fn perlin() {
        let result = super::perlin_2d(1.5, 1.5);
        assert_eq!(result, 0.0);
    }

    /// the middle point with all gradients point to (1,1) result in:
    ///     (g00+g10+g01+g11)/4
    ///     = ((x + y) + (x-1 + y) + (x + y-1) + (x-1 + y-1))/4
    ///     = (4x + 4y-4)/4
    ///     = x + y - 1 = 0.5 + 0.5 - 1 = 0
    #[test]
    fn eval() {
        let result = super::eval(0.5, 0.5, 0, 0, 0, 0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn eval_grad() {
        // value is correct
        let (f_xy, _) = super::eval_grad(0.5, 0.0, 0, 0, 0, 0);
        assert_eq!(f_xy, 0.0);

        let (f_xy, der) = super::eval_grad(0.5, 0.1, 0, 0, 0, 0);

        // df/dx
        // negative partial derivative over x => noise decreases when x increases
        let h = 0.001;
        let (f_xhy, _) = super::eval_grad(0.5 + h, 0.1, 0, 0, 0, 0);

        // f(x+h,y) - f(x,y) = h * df(x,y)/dx
        let result = f_xhy - f_xy;
        let expected = der[0] * h;
        // (+/- f32 epsilon)
        assert_eq!(
            (result * 100000.0) as isize,
            (expected * 100000.0) as isize
        );

        // df/dy
        let (f_xy, der) = super::eval_grad(0.1, 0.5, 0, 0, 0, 0);
        let h = 0.001;
        let (f_xyh, _) = super::eval_grad(0.1, 0.5 + h, 0, 0, 0, 0);
        // f(x,y+h) - f(x,y) = h * df(x,y)/dy
        let result = f_xyh - f_xy;
        let expected = der[1] * h;
        assert_eq!(
            (result * 100000.0) as isize,
            (expected * 100000.0) as isize
        );

        // df/dy
        let (f_xy, d) = super::eval_grad(0.2, 0.2, 0, 0, 0, 0);
        let hx = 0.0001;
        let hy = 0.0002;
        let (f_xhyh, _) = super::eval_grad(0.2 + hx, 0.2 + hy, 0, 0, 0, 0);
        // f(x+h1,y+h2) - f(x,y) = h1 * df(x,y)/dx + h2 * df(x,y)/dy
        let result = f_xhyh - f_xy;
        let expected = d[0] * hx + d[1] * hy;
        assert_eq!(
            (result * 100000.0) as isize,
            (expected * 100000.0) as isize
        );
    }
}
