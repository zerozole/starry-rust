//! Surface construction, sampled image transforms, and rendering.
use crate::{Map, Result, basis, matrix::Matrix, require};
use std::f64::consts::PI;

/// Gaussian spherical smoothing transfer function, repeated by harmonic order.
pub fn smoothing_weights(degree: usize, smoothing: f64) -> Result<Vec<f64>> {
    require(
        degree <= basis::MAX_DEGREE && smoothing.is_finite() && smoothing >= 0.,
        "invalid smoothing",
    )?;
    Ok((0..=degree)
        .flat_map(|l| {
            std::iter::repeat_n(
                (-0.5 * (l * (l + 1)) as f64 * smoothing * smoothing).exp(),
                2 * l + 1,
            )
        })
        .collect())
}

pub(crate) fn legendre_values(degree: usize, z: f64) -> Vec<f64> {
    let mut p = vec![1.; degree + 1];
    if degree > 0 {
        p[1] = z;
    }
    for l in 2..=degree {
        p[l] = ((2 * l - 1) as f64 * z * p[l - 1] - (l - 1) as f64 * p[l - 2]) / l as f64;
    }
    p
}
pub(crate) fn axisymmetric_fit(
    degree: usize,
    z: &[f64],
    values: &[f64],
    epsilon: f64,
    smoothing: f64,
) -> Result<Vec<f64>> {
    require(
        degree <= basis::MAX_DEGREE
            && z.len() == values.len()
            && z.len() > degree
            && values.iter().all(|v| v.is_finite()),
        "invalid axisymmetric samples",
    )?;
    let n = degree + 1;
    let mut normal = Matrix::zeros(n, n);
    let mut rhs = Matrix::zeros(n, 1);
    for (&z, &value) in z.iter().zip(values) {
        let mut p = legendre_values(degree, z);
        for (l, v) in p.iter_mut().enumerate() {
            *v *= ((2 * l + 1) as f64).sqrt();
        }
        for i in 0..n {
            rhs[(i, 0)] += p[i] * value;
            for j in 0..n {
                normal[(i, j)] += p[i] * p[j];
            }
        }
    }
    for i in 0..n {
        normal[(i, i)] += epsilon;
    }
    let weights = normal.solve(&rhs)?.data;
    let mut y = vec![0.; n * n];
    for l in 0..n {
        y[basis::index(l, 0)] =
            weights[l] * (-0.5 * (l * (l + 1)) as f64 * smoothing * smoothing).exp();
    }
    Ok(y)
}

/// Sigmoidal circular spot expansion with upstream's default sampling/smoothing.
pub fn add_spot(
    map: &mut Map,
    contrast: f64,
    radius: f64,
    latitude: f64,
    longitude: f64,
    smoothing: Option<f64>,
) -> Result<()> {
    require(
        [contrast, radius, latitude, longitude]
            .iter()
            .all(|v| v.is_finite())
            && (0. ..=PI).contains(&radius)
            && latitude.abs() <= PI / 2.,
        "invalid spot",
    )?;
    let smoothing = smoothing.unwrap_or(if map.degree() < 4 {
        0.5
    } else {
        2. / map.degree() as f64
    });
    require(
        smoothing.is_finite() && smoothing >= 0.,
        "invalid spot smoothing",
    )?;
    let angles: Vec<_> = (0..1000).map(|i| PI * i as f64 / 999.).collect();
    let z: Vec<_> = angles.iter().map(|v| v.cos()).collect();
    let values: Vec<_> = angles
        .iter()
        .map(|theta| -1. / (1. + (300. * (theta - radius)).exp()))
        .collect();
    let mut spot = Map::new(map.degree())?;
    spot.set_coefficients(&axisymmetric_fit(
        map.degree(),
        &z,
        &values,
        1e-9,
        smoothing,
    )?)?;
    spot.rotate([1., 0., 0.], -latitude)?;
    spot.rotate([0., 1., 0.], longitude)?;
    let current = map.coefficients().to_vec();
    let amplitude = map.amplitude();
    let absolute: Vec<_> = current
        .iter()
        .zip(spot.coefficients())
        .map(|(a, b)| amplitude * a + contrast * b)
        .collect();
    // Preserve the represented field even when the monopole becomes exactly zero.
    if absolute[0] != 0. {
        let amplitude = absolute[0];
        map.set_coefficients(&absolute.iter().map(|v| v / amplitude).collect::<Vec<_>>())?;
        map.set_amplitude(amplitude)
    } else {
        map.set_coefficients(&absolute)?;
        map.set_amplitude(1.)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Projection {
    Orthographic,
    Rectangular,
    Mollweide,
}
pub fn render(map: &Map, width: usize, height: usize, projection: Projection) -> Result<Vec<f64>> {
    require(
        width >= 2 && height >= 2 && width.checked_mul(height).is_some_and(|n| n <= 16_000_000),
        "invalid image size",
    )?;
    let mut image = vec![f64::NAN; width * height];
    for iy in 0..height {
        for ix in 0..width {
            let x = 2. * (ix as f64 + 0.5) / width as f64 - 1.;
            let y = 1. - 2. * (iy as f64 + 0.5) / height as f64;
            let value = match projection {
                Projection::Orthographic => {
                    let r2 = x * x + y * y;
                    if r2 > 1. {
                        continue;
                    }
                    map.intensity_xyz([x, y, (1. - r2).sqrt()])?
                }
                Projection::Rectangular => map.intensity(y * PI / 2., x * PI)?,
                Projection::Mollweide => {
                    if x * x + y * y > 1. {
                        continue;
                    }
                    let t = y.asin();
                    let latitude = ((2. * t + (2. * t).sin()) / PI).clamp(-1., 1.).asin();
                    let longitude = PI * x / t.cos();
                    if longitude.abs() > PI {
                        continue;
                    }
                    map.intensity(latitude, longitude)?
                }
            };
            image[iy * width + ix] = value;
        }
    }
    Ok(image)
}

/// Fit map intensities at supplied latitude/longitude samples, with ridge penalty.
pub fn fit_samples(
    degree: usize,
    latitudes: &[f64],
    longitudes: &[f64],
    intensities: &[f64],
    weights: &[f64],
    ridge: f64,
) -> Result<Map> {
    let n = latitudes.len();
    let mut map = Map::new(degree)?;
    let p = (degree + 1).pow(2);
    require(
        n > 0
            && longitudes.len() == n
            && intensities.len() == n
            && weights.len() == n
            && ridge.is_finite()
            && ridge >= 0.,
        "invalid image samples",
    )?;
    let mut normal = Matrix::zeros(p, p);
    let mut rhs = Matrix::zeros(p, 1);
    for k in 0..n {
        require(
            latitudes[k].is_finite()
                && latitudes[k].abs() <= PI / 2.
                && longitudes[k].is_finite()
                && intensities[k].is_finite()
                && weights[k].is_finite()
                && weights[k] >= 0.,
            "invalid surface sample",
        )?;
        let lat = latitudes[k];
        let lon = longitudes[k];
        let row =
            map.intensity_design([lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()])?;
        for i in 0..p {
            rhs[(i, 0)] += weights[k] * row[i] * intensities[k];
            for j in 0..p {
                normal[(i, j)] += weights[k] * row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        normal[(i, i)] += ridge;
    }
    map.set_coefficients(&normal.solve(&rhs)?.data)?;
    Ok(map)
}

#[derive(Clone, Debug)]
pub struct Minimum {
    pub latitude: f64,
    pub longitude: f64,
    pub value: f64,
    pub evaluations: usize,
    pub converged: bool,
}
/// Bounded multistart minimization with analytic surface gradients. Coarse
/// sampling covers the entire supplied box; it is not a global optimality proof.
pub fn minimize(
    map: &Map,
    bounds: [[f64; 2]; 2],
    oversample: usize,
    tries: usize,
) -> Result<Minimum> {
    require(
        bounds.iter().flatten().all(|v| v.is_finite())
            && bounds[0][0] >= -PI / 2.
            && bounds[0][1] <= PI / 2.
            && bounds.iter().all(|p| p[0] < p[1])
            && (1..=20).contains(&oversample)
            && (1..=100).contains(&tries),
        "invalid minimization bounds/settings",
    )?;
    let n = ((map.degree() + 1) * oversample).max(10);
    let mut starts = vec![];
    let mut evaluations = 0;
    for i in 0..=n {
        for j in 0..=2 * n {
            let lat = bounds[0][0] + (bounds[0][1] - bounds[0][0]) * i as f64 / n as f64;
            let lon = bounds[1][0] + (bounds[1][1] - bounds[1][0]) * j as f64 / (2 * n) as f64;
            let value = map.intensity(lat, lon)?;
            evaluations += 1;
            starts.push((value, lat, lon));
        }
    }
    starts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut best = Minimum {
        value: starts[0].0,
        latitude: starts[0].1,
        longitude: starts[0].2,
        evaluations: 0,
        converged: false,
    };
    for &(mut value, mut lat, mut lon) in starts.iter().take(tries) {
        let mut converged = false;
        for _ in 0..300 {
            let g = map.intensity_gradient(lat, lon)?;
            let length = g[0].hypot(g[1]);
            if length < 1e-10 {
                converged = true;
                break;
            }
            let scale = 0.2 / length.max(1.);
            let mut accepted = false;
            for step in 0..35 {
                let a = (lat - scale * 2_f64.powi(-step) * g[0]).clamp(bounds[0][0], bounds[0][1]);
                let b = (lon - scale * 2_f64.powi(-step) * g[1]).clamp(bounds[1][0], bounds[1][1]);
                let next = map.intensity(a, b)?;
                evaluations += 1;
                if next < value {
                    converged = (a - lat).hypot(b - lon) < 1e-10;
                    lat = a;
                    lon = b;
                    value = next;
                    accepted = true;
                    break;
                }
            }
            if !accepted || converged {
                if !accepted {
                    converged = ((lat - g[0]).clamp(bounds[0][0], bounds[0][1]) - lat)
                        .hypot((lon - g[1]).clamp(bounds[1][0], bounds[1][1]) - lon)
                        < 1e-9;
                }
                break;
            }
        }
        if value <= best.value {
            best = Minimum {
                latitude: lat,
                longitude: lon,
                value,
                evaluations: 0,
                converged,
            };
        }
    }
    best.evaluations = evaluations;
    Ok(best)
}

/// Strict positive, monotonically center-brightened limb profile on [0,1].
/// Check extrema of the profile and its derivative by polynomial root isolation.
pub fn limb_is_physical(u: &[f64]) -> Result<bool> {
    crate::map::limb_intensity(1., u)?;
    if u.is_empty() {
        return Ok(true);
    }
    // Work in x=1-mu. Profile p=1-sum(u_n*x^n); -p' must be positive.
    let mut profile = vec![1.];
    profile.extend(u.iter().map(|v| -v));
    let derivative: Vec<_> = u
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1) as f64 * v)
        .collect();
    let second: Vec<_> = derivative
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, v)| i as f64 * v)
        .collect();
    let mut points = vec![0., 1.];
    points.extend(crate::polynomial_roots::real_roots(&derivative, 0., 1.));
    if points
        .iter()
        .any(|x| crate::polynomial_roots::evaluate(&profile, *x) <= 0.)
    {
        return Ok(false);
    }
    let mut points = vec![0., 1.];
    points.extend(crate::polynomial_roots::real_roots(&second, 0., 1.));
    Ok(points
        .iter()
        .all(|x| crate::polynomial_roots::evaluate(&derivative, *x) > 0.))
}
