//! Equal-area pixel sampling and regularized harmonic transforms.
use crate::{Map, Result, basis, matrix::Matrix, require};
use std::f64::consts::PI;

pub fn grid(degree: usize, oversample: usize) -> Result<Vec<[f64; 2]>> {
    require(
        degree <= basis::MAX_DEGREE && (1..=10).contains(&oversample),
        "invalid pixel grid",
    )?;
    let ny = ((oversample.max(3) * (degree + 1).pow(2)) as f64 * PI / 4.).sqrt() as usize;
    let ny = ny.max(3);
    let nx = 2 * ny;
    let mut points = vec![];
    for ix in 0..nx {
        let x = 2. * ix as f64 / (nx - 1) as f64 - 1.;
        for iy in 0..ny {
            let y = 2. * iy as f64 / (ny - 1) as f64 - 1.;
            if x * x + y * y > 1. || y.abs() == 1. {
                continue;
            }
            let theta = y.asin();
            points.push([
                ((2. * theta + (2. * theta).sin()) / PI)
                    .clamp(-1., 1.)
                    .asin(),
                PI * x / theta.cos(),
            ]);
        }
    }
    points.extend([[-PI / 2., 0.], [0., 0.], [0., PI], [PI / 2., 0.]]);
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    Ok(points)
}

/// Derivative matrices act on harmonic coefficients; combine with inverse
/// to differentiate pixels without storing a quadratic-size pixel operator.
pub struct Transforms {
    pub forward: Matrix,
    pub inverse: Matrix,
    pub latitude: Matrix,
    pub longitude: Matrix,
}
pub fn transforms(degree: usize, points: &[[f64; 2]], ridge: f64) -> Result<Transforms> {
    require(
        !points.is_empty() && points.len() <= 10000 && ridge.is_finite() && ridge >= 0.,
        "invalid pixel transform",
    )?;
    let map = Map::new(degree)?;
    let p = (degree + 1).pow(2);
    let n = points.len();
    let mut forward = Matrix::zeros(n, p);
    let mut latitude = forward.clone();
    let mut longitude = forward.clone();
    for (i, &[lat, lon]) in points.iter().enumerate() {
        require(
            lat.is_finite() && lat.abs() <= PI / 2. && lon.is_finite(),
            "invalid pixel coordinate",
        )?;
        let xyz = [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()];
        let row = map.intensity_design(xyz)?;
        forward.data[i * p..(i + 1) * p].copy_from_slice(&row);
        let dlat = [-lat.sin() * lon.sin(), lat.cos(), -lat.sin() * lon.cos()];
        let dlon = [lat.cos() * lon.cos(), 0., -lat.cos() * lon.sin()];
        for (matrix, tangent) in [(&mut latitude, dlat), (&mut longitude, dlon)] {
            matrix.data[i * p..(i + 1) * p]
                .copy_from_slice(&basis::harmonic_directional(degree, xyz, tangent)?);
        }
    }
    let mut normal = Matrix::zeros(p, p);
    let mut transpose = Matrix::zeros(p, n);
    for i in 0..p {
        for k in 0..n {
            transpose[(i, k)] = forward[(k, i)];
        }
        for j in 0..=i {
            let value = (0..n)
                .map(|k| forward[(k, i)] * forward[(k, j)])
                .sum::<f64>();
            normal[(i, j)] = value;
            normal[(j, i)] = value;
        }
        normal[(i, i)] += ridge;
    }
    Ok(Transforms {
        forward,
        inverse: normal.solve(&transpose)?,
        latitude,
        longitude,
    })
}
