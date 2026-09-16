//! Rotations by exact polynomial substitution on the unit sphere.
//! This replaces the upstream Wigner recurrence; it is not a performance port.
use crate::{Result, basis, matrix::Matrix, require};

/// Right-handed active rotation; angles are radians, axis is normalized here.
pub fn axis_angle(axis: [f64; 3], angle: f64) -> Result<[[f64; 3]; 3]> {
    require(
        axis.iter().all(|v| v.is_finite()) && angle.is_finite(),
        "nonfinite rotation",
    )?;
    let norm = axis[0].hypot(axis[1]).hypot(axis[2]);
    require(
        norm > 0. && norm.is_finite(),
        "rotation axis must be nonzero and finite",
    )?;
    let [x, y, z] = axis.map(|v| v / norm);
    let (s, c) = angle.sin_cos();
    let t = 1. - c;
    Ok([
        [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
        [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
        [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
    ])
}
pub fn apply(r: [[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    r.map(|v| v.iter().zip(p).map(|(a, b)| a * b).sum())
}
pub fn transpose(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]))
}

/// Substitute coordinates `p -> r p` in canonical polynomial coefficients.
pub(crate) fn substitute(degree: usize, p: &[f64], r: [[f64; 3]; 3]) -> Vec<f64> {
    let mut out = vec![0.; p.len()];
    let lin = r.map(|v| vec![0., v[0], v[2], v[1]]);
    let powers: Vec<Vec<Vec<f64>>> = lin
        .iter()
        .map(|v| {
            let mut a = vec![vec![1.]];
            for k in 1..=degree {
                a.push(basis::multiply(&a[k - 1], k - 1, v, 1));
            }
            a
        })
        .collect();
    for (i, (a, b, c)) in basis::terms(degree).into_iter().enumerate() {
        if p[i] == 0. {
            continue;
        }
        let xy = basis::multiply(&powers[0][a], a, &powers[1][b], b);
        let xyz = basis::multiply(&xy, a + b, &powers[2][c], c);
        for (j, v) in xyz.into_iter().enumerate() {
            out[j] += p[i] * v;
        }
    }
    out
}

pub fn coefficients(degree: usize, y: &[f64], axis: [f64; 3], angle: f64) -> Result<Vec<f64>> {
    if degree > 8 {
        return projected_coefficients(degree, y, axis, angle);
    }
    let a = basis::a1(degree)?;
    require(
        y.len() == a.cols && y.iter().all(|v| v.is_finite()),
        "invalid harmonic coefficients",
    )?;
    let r = transpose(axis_angle(axis, angle)?);
    let p = substitute(degree, &a.dot(y)?, r);
    let rhs = Matrix {
        rows: p.len(),
        cols: 1,
        data: p,
    };
    Ok(a.solve(&rhs)?.data)
}

/// Exact-in-polynomial-degree spherical projection using Gauss-Legendre and
/// periodic longitude quadrature. Avoids solving ill-conditioned monomials.
pub fn projected_coefficients(
    degree: usize,
    y: &[f64],
    axis: [f64; 3],
    angle: f64,
) -> Result<Vec<f64>> {
    require(
        degree <= basis::MAX_DEGREE
            && y.len() == (degree + 1).pow(2)
            && y.iter().all(|v| v.is_finite()),
        "invalid harmonic coefficients",
    )?;
    let r = transpose(axis_angle(axis, angle)?);
    project_function(degree, |xyz| {
        Ok(crate::map::dot(
            &basis::harmonics(degree, apply(r, xyz))?,
            y,
        ))
    })
}

/// Project a function known to have the supplied band limit onto harmonics.
/// Arbitrary non-band-limited functions require a separate convergence study.
pub fn project_function(
    degree: usize,
    mut function: impl FnMut([f64; 3]) -> Result<f64>,
) -> Result<Vec<f64>> {
    require(
        degree <= basis::MAX_DEGREE,
        "projection degree exceeds supported limit",
    )?;
    let nphi = 2 * degree + 1;
    let mut result = vec![0.; (degree + 1).pow(2)];
    let mut compensation = result.clone();
    for (z, weight) in crate::occultation::gauss(degree + 1) {
        let radius = (1. - z * z).sqrt();
        for j in 0..nphi {
            let phi = 2. * std::f64::consts::PI * j as f64 / nphi as f64;
            let xyz = [radius * phi.cos(), radius * phi.sin(), z];
            let intensity = function(xyz)?;
            require(intensity.is_finite(), "nonfinite projection sample")?;
            let target = basis::harmonics(degree, xyz)?;
            let scale = std::f64::consts::PI.powi(2) / (2. * nphi as f64) * weight * intensity;
            for i in 0..result.len() {
                let term = scale * target[i] - compensation[i];
                let sum = result[i] + term;
                compensation[i] = (sum - result[i]) - term;
                result[i] = sum;
            }
        }
    }
    Ok(result)
}
