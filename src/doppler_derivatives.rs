//! Differentiated Doppler chord integration, including discrete kernel
//! normalization. Source: OpsDoppler.get_x/get_rT/get_kT0/get_kT.
use crate::{
    Map, Result, basis, differentiation, doppler, map, matrix::Matrix, occultation::gauss, require,
};
use std::f64::consts::PI;

/// Rows: kernel value, phase, inclination, equatorial speed, log-grid spacing,
/// then each limb coefficient. Angles are radians, speed is m/s.
/// Spectra and grid size are held fixed. Exact support edges are singular.
#[allow(clippy::too_many_arguments)]
pub fn kernel(
    map: &Map,
    inc: f64,
    phase: f64,
    veq: f64,
    u: &[f64],
    spacing: f64,
    half_width: usize,
) -> Result<Matrix> {
    // Reuse the forward path's domain validation and check value parity in tests.
    doppler::kernel(map, inc, phase, veq, u, spacing, half_width)?;
    let projected = map.projected(inc, 0., phase)?;
    let dphase = differentiation::rotation_derivative(&projected, [0., inc.sin(), inc.cos()])?;
    let dinc = differentiation::rotation_derivative(&projected, [-1., 0., 0.])?;
    let v = veq * inc.sin();
    let vsini = v.max(1.);
    require(
        (v - 1.).abs() > 1e-12,
        "Doppler derivative undefined at the velocity-floor kink",
    )?;
    let n = 2 * half_width + 1;
    let mut result = Matrix::zeros(5 + u.len(), n);
    let (_, norm) = map::limb_polynomial(u)?;
    let nodes = gauss(64);
    let mut base = 0.;
    let mut base_derivatives = [0.; 3];
    for j in 0..n {
        let index = j as f64 - half_width as f64;
        let shift = (index * spacing).tanh();
        let x = -doppler::SPEED_OF_LIGHT * shift / vsini;
        require(
            (x.abs() - 1.).abs() > 1e-12,
            "Doppler derivative undefined at a chord support edge",
        )?;
        if x.abs() >= 1. {
            continue;
        }
        let r = ((1. - x) * (1. + x)).sqrt();
        let dx = [
            if v > 1. {
                -x * veq * inc.cos() / vsini
            } else {
                0.
            },
            if v > 1. { -x * inc.sin() / vsini } else { 0. },
            -doppler::SPEED_OF_LIGHT * index * (1. - shift * shift) / vsini,
        ];
        base += 2. * r / PI;
        for k in 0..3 {
            base_derivatives[k] += -2. * x / (PI * r) * dx[k];
        }
        let mut chord_dx = 0.;
        for &(node, weight) in &nodes {
            let theta = PI / 2. * node;
            let (s, c) = theta.sin_cos();
            let p = [x, r * s, r * c];
            let tangent = [1., -x / r * s, -x / r * c];
            let intensity = projected.intensity_xyz(p)?;
            let limb = map::limb_intensity(p[2], u)? / norm;
            let limb_z = u
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * v * (1. - p[2]).powi(i as i32))
                .sum::<f64>()
                / norm;
            let w = weight * PI / 2. * r * c;
            result[(0, j)] += w * intensity * limb;
            result[(1, j)] += w * dphase.intensity_xyz(p)? * limb;
            result[(2, j)] += w * dinc.intensity_xyz(p)? * limb;
            let di = projected.amplitude()
                * map::dot(
                    &basis::harmonic_directional(projected.degree(), p, tangent)?,
                    projected.coefficients(),
                );
            chord_dx += weight * PI / 2.
                * c
                * (-x / r * intensity * limb + r * (di * limb + intensity * limb_z * tangent[2]));
            for i in 0..u.len() {
                let normalization_derivative = 2. / ((i + 2) * (i + 3)) as f64;
                result[(5 + i, j)] += w
                    * intensity
                    * (-(1. - p[2]).powi(i as i32 + 1) + limb * normalization_derivative)
                    / norm;
            }
        }
        for k in 0..3 {
            result[(2 + k, j)] += chord_dx * dx[k];
        }
    }
    require(base > 0., "empty Doppler support")?;
    for j in 0..n {
        result[(0, j)] /= base;
        for k in 1..result.rows {
            let db = if (2..5).contains(&k) {
                base_derivatives[k - 2]
            } else {
                0.
            };
            result[(k, j)] = (result[(k, j)] - result[(0, j)] * db) / base;
        }
    }
    require(
        result.data.iter().all(|v| v.is_finite()),
        "nonfinite Doppler derivative",
    )?;
    Ok(result)
}

/// Convolved rows plus a final column containing each continuum derivative.
#[allow(clippy::too_many_arguments)]
pub fn spectrum(
    map: &Map,
    inc: f64,
    phase: f64,
    veq: f64,
    u: &[f64],
    spacing: f64,
    half_width: usize,
    rest: &[f64],
) -> Result<Matrix> {
    require(
        rest.len() > 2 * half_width && rest.iter().all(|v| v.is_finite()),
        "invalid rest spectrum",
    )?;
    let kernels = kernel(map, inc, phase, veq, u, spacing, half_width)?;
    let n = rest.len() - 2 * half_width;
    let mut result = Matrix::zeros(kernels.rows, n + 1);
    for row in 0..kernels.rows {
        for j in 0..kernels.cols {
            result[(row, n)] += kernels[(row, j)];
            for i in 0..n {
                result[(row, i)] += kernels[(row, j)] * rest[i + j];
            }
        }
    }
    Ok(result)
}
