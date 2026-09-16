//! Reflected illumination derivatives, including its moving terminator.
//! Scalar Oren-Nayar coefficients mirror reflected::illumination_polynomial;
//! forward jets differentiate those coefficients, never sampled flux values.
use crate::oren_coefficients::COEFFICIENTS;
use crate::{
    Map, Result, autodiff::Jet, basis, map, occultation::Occultor, quadrature, reflected, require,
    rotation,
};

fn illumination(source: [Jet<4>; 3], rough: Jet<4>) -> impl Fn([f64; 3]) -> Result<[f64; 4]> {
    let c = Jet::constant;
    let distance = (source[0] * source[0] + source[1] * source[1] + source[2] * source[2]).sqrt();
    let lambert = source.map(|v| v / distance.powi(3));
    let b = if rough.value == 0. {
        c(0.)
    } else {
        -source[2] / distance
    };
    let bc = (c(1.) - b * b).sqrt();
    let angle = if rough.value == 0. {
        c(0.)
    } else {
        source[0].atan2(source[1])
    };
    let (sine, cosine) = (angle.sin(), angle.cos());
    let sig2 = rough * rough;
    let ca = c(1.) - c(0.5) * sig2 / (sig2 + c(0.33));
    let cb = c(0.45) * sig2 / (sig2 + c(0.09));
    let mut coefficients = vec![c(0.); 36];
    coefficients[2] = -ca * b;
    coefficients[3] = ca * bc;
    if b.value < 0. {
        for (n, coefficient) in coefficients.iter_mut().enumerate() {
            for a in 0..4 {
                for z in 0..4 {
                    *coefficient = *coefficient
                        + cb * c(COEFFICIENTS[n * 16 + a * 4 + z]) * b.powi(a + 1) * bc.powi(z);
                }
            }
        }
    }
    for v in &mut coefficients {
        *v = *v / distance.powi(2);
    }
    move |p| {
        if rough.value == 0. {
            return Ok(std::array::from_fn(|k| {
                (0..3).map(|j| lambert[j].gradient[k] * p[j]).sum()
            }));
        }
        let q = [
            cosine.value * p[0] - sine.value * p[1],
            sine.value * p[0] + cosine.value * p[1],
            p[2],
        ];
        let terms = basis::polynomial(5, q)?;
        let gradients = basis::polynomial_gradient(5, q)?;
        Ok(std::array::from_fn(|k| {
            let dx = cosine.gradient[k] * p[0] - sine.gradient[k] * p[1];
            let dy = sine.gradient[k] * p[0] + cosine.gradient[k] * p[1];
            coefficients
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    c.gradient[k] * terms[j]
                        + c.value * (gradients[0][j] * dx + gradients[1][j] * dy)
                })
                .sum()
        }))
    }
}

/// First derivatives for a fixed observer-frame map and a point source.
/// Returns source x,y,z and roughness derivatives. Angles are radians.
pub fn point(
    map: &Map,
    source: [f64; 3],
    roughness: f64,
    occultors: &[Occultor],
) -> Result<[f64; 4]> {
    point_ellipses(map, source, roughness, &quadrature::circles(occultors)?)
}

pub fn point_ellipses(
    map: &Map,
    source: [f64; 3],
    roughness: f64,
    silhouettes: &[crate::conic::Ellipse],
) -> Result<[f64; 4]> {
    let distance = source[0].hypot(source[1]).hypot(source[2]);
    require(
        source.iter().all(|x| x.is_finite())
            && distance > 0.
            && distance.is_finite()
            && roughness.is_finite()
            && roughness >= 0.,
        "invalid reflected derivative inputs",
    )?;
    require(
        roughness == 0. || source[0].hypot(source[1]) > 1e-12 * distance,
        "rough-source derivatives at an exact phase pole require a directional limit",
    )?;
    let b = -source[2] / distance;
    let angle = source[0].atan2(source[1]);
    let r = rotation::axis_angle([0., 0., 1.], angle)?;
    let inverse = rotation::transpose(r);
    let ellipses: Vec<_> = silhouettes.iter().map(|e| e.rotated(angle)).collect();
    let variables = std::array::from_fn(|i| Jet::variable(source[i], i));
    let rough = Jet::variable(roughness, 3);
    let illumination = illumination(variables, rough);
    let mut result = [0.; 4];
    for (k, derivative) in result.iter_mut().enumerate() {
        *derivative = quadrature::surface(1., b, &ellipses, 2e-11, |q| {
            let p = rotation::apply(inverse, q);
            Ok(map.intensity_xyz(p)? * illumination(p)?[k])
        })?;
        // The source approximation may have residual intensity on its
        // terminator. Include its shape term rather than assuming it vanishes.
        if k < 3 && roughness > 0. {
            let bc = (1. - b * b).max(0.).sqrt();
            let (degree, poly) = reflected::illumination_polynomial(b, roughness)?;
            *derivative += quadrature::terminator(b, &ellipses, |q| {
                let p = rotation::apply(inverse, q);
                Ok(bc * p[k] / distance
                    * map.intensity_xyz(p)?
                    * map::dot(&poly, &basis::polynomial(degree, q)?)
                    / distance.powi(2))
            })?;
        }
    }
    require(
        result.iter().all(|v| v.is_finite()),
        "nonfinite reflected derivative",
    )?;
    Ok(result)
}

/// Finite-source chain rule through the source's effective disk samples.
/// Returns source x,y,z, roughness and source-radius derivatives.
pub fn extended(
    map: &Map,
    source: [f64; 3],
    radius: f64,
    samples: usize,
    roughness: f64,
    occultors: &[Occultor],
) -> Result<[f64; 5]> {
    extended_ellipses(
        map,
        source,
        radius,
        samples,
        roughness,
        &quadrature::circles(occultors)?,
    )
}

pub fn extended_ellipses(
    map: &Map,
    source: [f64; 3],
    radius: f64,
    samples: usize,
    roughness: f64,
    ellipses: &[crate::conic::Ellipse],
) -> Result<[f64; 5]> {
    let points = reflected::source_points(source, radius, samples)?;
    if samples == 1 || radius == 0. {
        let g = point_ellipses(map, source, roughness, ellipses)?;
        let mut radius_gradient = 0.;
        if samples > 1 {
            let distance = source[0].hypot(source[1]).hypot(source[2]);
            require(
                distance >= 1.,
                "zero-radius source has no adjacent valid finite-source domain",
            )?;
            let q2 = 1. - 1. / distance.powi(2);
            let n = (2. + (samples as f64 * 4. / std::f64::consts::PI).sqrt()) as usize;
            let mut count = 0;
            for iy in 0..n {
                for ix in 0..n {
                    let x = -1. + 2. * ix as f64 / (n - 1) as f64;
                    let y = -1. + 2. * iy as f64 / (n - 1) as f64;
                    if x * x + y * y < 1. {
                        radius_gradient -= g[2] * (1. - q2 * (x * x + y * y)).sqrt();
                        count += 1;
                    }
                }
            }
            radius_gradient /= count as f64;
        }
        // Transverse sample displacements cancel by symmetry. The source's
        // facing hemisphere also moves toward the observer; that term remains.
        return Ok([g[0], g[1], g[2], g[3], radius_gradient]);
    }
    let c = Jet::<4>::constant;
    let variables: [Jet<4>; 3] = std::array::from_fn(|i| Jet::variable(source[i], i));
    let rs = Jet::variable(radius, 3);
    let distance = (variables[0].powi(2) + variables[1].powi(2) + variables[2].powi(2)).sqrt();
    let effective = rs * (c(1.) - ((rs - c(1.)) / distance).powi(2)).sqrt();
    require(
        effective.value > 0.,
        "degenerate finite-source effective disk",
    )?;
    let mut result = [0.; 5];
    for p in &points {
        let x = (p[0] - source[0]) / effective.value;
        let y = (p[1] - source[1]) / effective.value;
        let dx = effective * c(x);
        let dy = effective * c(y);
        let dz = -(rs.powi(2) - dx.powi(2) - dy.powi(2)).sqrt();
        let jets = [variables[0] + dx, variables[1] + dy, variables[2] + dz];
        let gradient = point_ellipses(map, *p, roughness, ellipses)?;
        for k in 0..4 {
            result[if k == 3 { 4 } else { k }] += (0..3)
                .map(|i| gradient[i] * jets[i].gradient[k])
                .sum::<f64>();
        }
        result[3] += gradient[3];
    }
    for v in &mut result {
        *v /= points.len() as f64;
    }
    require(
        result.iter().all(|v| v.is_finite()),
        "nonfinite finite-source derivative",
    )?;
    Ok(result)
}
