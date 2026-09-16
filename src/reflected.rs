//! Reflected light, including the upstream polynomial Oren-Nayar approximation.
//! Phase moments are a port of reflected/phasecurve.h. Occultations integrate
//! the exact clipped polynomial domain with the independent adaptive integrator.
use crate::oren_coefficients::COEFFICIENTS;
use crate::{
    Map, Result, basis,
    occultation::{Integration, Occultor},
    require, rotation,
};
use std::f64::consts::PI;

/// Integrals on the visible dayside y >= b sqrt(1-x²), before illumination.
pub fn dayside_moments(degree: usize, b: f64) -> Result<Vec<f64>> {
    require(
        degree <= basis::POLYNOMIAL_LIMIT && b.is_finite() && (-1. ..=1.).contains(&b),
        "invalid reflected degree/terminator",
    )?;
    let size = degree + 3;
    let mut l = vec![vec![0.; size]; size];
    let mut m = l.clone();
    l[0][0] = PI;
    l[0][1] = 4. / 3.;
    m[0][0] = 4. / 3.;
    m[0][1] = 3. * PI / 8.;
    for j in 0..=degree {
        l[0][j + 2] = l[0][j] * (j + 1) as f64 / (j + 4) as f64;
        m[0][j + 2] = m[0][j] * (j + 4) as f64 / (j + 5) as f64;
    }
    for j in 0..=degree {
        for i in (0..degree).step_by(2) {
            l[i + 2][j] = l[i][j] * (i + 1) as f64 / (i + j + 4) as f64;
            m[i + 2][j] = m[i][j] * (i + 1) as f64 / (i + j + 5) as f64;
        }
    }
    let mut k = vec![0.; size];
    let w = (1. - b * b).max(0.).sqrt();
    k[0] = 0.5 * (b.acos() - b * w);
    k[1] = w.powi(3) / 3.;
    for j in 0..degree {
        k[j + 2] = (b.powi(j as i32 + 1) * w.powi(3) + (j + 1) as f64 * k[j]) / (j + 4) as f64;
    }
    Ok(basis::terms(degree)
        .into_iter()
        .map(|(i, j, z)| {
            if z == 0 {
                0.5 * (1. - b.powi(j as i32 + 1)) * l[i][j]
            } else {
                k[j] * m[i][j]
            }
        })
        .collect())
}

pub fn illumination_polynomial(b: f64, roughness: f64) -> Result<(usize, Vec<f64>)> {
    require(
        b.is_finite() && (-1. ..=1.).contains(&b) && roughness.is_finite() && roughness >= 0.,
        "invalid scattering arguments",
    )?;
    let bc = (1. - b * b).max(0.).sqrt();
    if roughness == 0. {
        return Ok((1, vec![0., 0., -b, bc]));
    }
    let sig2 = roughness * roughness;
    let ca = 1. - 0.5 * sig2 / (sig2 + 0.33);
    let cb = 0.45 * sig2 / (sig2 + 0.09);
    let mut p = vec![0.; 36];
    if b < 0. {
        for n in 0..36 {
            for i in 0..4 {
                for j in 0..4 {
                    p[n] += cb
                        * COEFFICIENTS[n * 16 + i * 4 + j]
                        * b.powi(i as i32 + 1)
                        * bc.powi(j as i32);
                }
            }
        }
    }
    p[2] -= ca * b;
    p[3] += ca * bc;
    Ok((5, p))
}

/// Reflected flux; source is a position vector in units of the map radius.
/// A uniform unit-albedo sphere has full-phase flux 2/(3*source_distance²).
pub fn flux(map: &Map, source: [f64; 3], roughness: f64, occ: Option<Occultor>) -> Result<f64> {
    flux_many(map, source, roughness, &occ.into_iter().collect::<Vec<_>>())
}

/// Finite-source sampling from OpsReflected.X: interior square-grid points on
/// the effective source disk, offset toward the observer along negative z.
pub fn source_points(source: [f64; 3], radius: f64, requested: usize) -> Result<Vec<[f64; 3]>> {
    let distance = source[0].hypot(source[1]).hypot(source[2]);
    require(
        source.iter().all(|v| v.is_finite())
            && distance.is_finite()
            && distance > 0.
            && radius.is_finite()
            && radius >= 0.
            && requested > 0
            && requested <= 100_000,
        "invalid finite source",
    )?;
    if radius == 0. || requested == 1 {
        return Ok(vec![source]);
    }
    let ratio = (radius - 1.) / distance;
    require(ratio.abs() <= 1., "invalid finite-source effective radius")?;
    let effective = radius * (1. - ratio * ratio).sqrt();
    let n = (2. + (requested as f64 * 4. / PI).sqrt()) as usize;
    let mut points = vec![];
    for iy in 0..n {
        for ix in 0..n {
            let x = -1. + 2. * ix as f64 / (n - 1) as f64;
            let y = -1. + 2. * iy as f64 / (n - 1) as f64;
            if x * x + y * y < 1. {
                let dx = effective * x;
                let dy = effective * y;
                let dz = -(radius * radius - dx * dx - dy * dy).max(0.).sqrt();
                points.push([source[0] + dx, source[1] + dy, source[2] + dz]);
            }
        }
    }
    require(!points.is_empty(), "empty finite-source grid")?;
    Ok(points)
}

pub fn flux_extended(
    map: &Map,
    source: [f64; 3],
    radius: f64,
    requested: usize,
    roughness: f64,
    occultors: &[Occultor],
) -> Result<f64> {
    let points = source_points(source, radius, requested)?;
    let mut sum = 0.;
    for p in &points {
        sum += flux_many(map, *p, roughness, occultors)?;
    }
    Ok(sum / points.len() as f64)
}

pub fn flux_many(
    map: &Map,
    source: [f64; 3],
    roughness: f64,
    occultors: &[Occultor],
) -> Result<f64> {
    flux_silhouettes(map, source, roughness, occultors, &[])
}

pub fn flux_extended_ellipses(
    map: &Map,
    source: [f64; 3],
    radius: f64,
    requested: usize,
    roughness: f64,
    ellipses: &[crate::conic::Ellipse],
) -> Result<f64> {
    let points = source_points(source, radius, requested)?;
    let mut flux = 0.;
    for point in &points {
        flux += flux_silhouettes(map, *point, roughness, &[], ellipses)?;
    }
    Ok(flux / points.len() as f64)
}

fn flux_silhouettes(
    map: &Map,
    source: [f64; 3],
    roughness: f64,
    occultors: &[Occultor],
    ellipses: &[crate::conic::Ellipse],
) -> Result<f64> {
    require(source.iter().all(|v| v.is_finite()), "nonfinite source")?;
    let distance = source[0].hypot(source[1]).hypot(source[2]);
    require(
        distance > 0. && distance.is_finite(),
        "source must be nonzero",
    )?;
    let b = (-source[2] / distance).clamp(-1., 1.);
    let angle = source[0].atan2(source[1]);
    let r = rotation::axis_angle([0., 0., 1.], angle)?;
    let (id, illum) = illumination_polynomial(b, roughness)?;
    if map.degree() + id > 12 {
        let mut silhouettes = crate::quadrature::circles(occultors)?;
        silhouettes.extend_from_slice(ellipses);
        let silhouettes: Vec<_> = silhouettes.iter().map(|e| e.rotated(angle)).collect();
        let inverse = rotation::transpose(r);
        return Ok(crate::quadrature::surface(1., b, &silhouettes, 2e-11, |p| {
            Ok(map.intensity_xyz(rotation::apply(inverse, p))?
                * crate::map::dot(&illum, &basis::polynomial(id, p)?))
        })? / distance.powi(2));
    }
    let p = rotation::substitute(
        map.degree(),
        &map.polynomial_coefficients()?,
        rotation::transpose(r),
    );
    let degree = map.degree() + id;
    require(
        degree <= basis::POLYNOMIAL_LIMIT,
        "map plus illumination degree exceeds 20",
    )?;
    let product = basis::multiply(&p, map.degree(), &illum, id);
    let mut moments = dayside_moments(degree, b)?;
    let mut aligned = vec![];
    for o in occultors {
        Occultor::new(o.x, o.y, o.radius)?;
        let q = rotation::apply(r, [o.x, o.y, 0.]);
        aligned.push(Occultor::new(q[0], q[1], o.radius)?);
    }
    if !ellipses.is_empty() {
        let ellipses: Vec<_> = ellipses.iter().map(|e| e.rotated(angle)).collect();
        moments = crate::occultation::projected_clipped(
            degree,
            1.,
            &ellipses,
            Integration::default(),
            b,
            moments,
        )?
        .values;
    } else if !aligned.is_empty() {
        moments = crate::occultation::clipped_union(
            degree,
            &aligned,
            Integration::default(),
            b,
            moments,
        )?
        .values;
    }
    Ok(map.amplitude() * crate::map::dot(&product, &moments) / (distance * distance))
}

/// Apparent local reflected intensity. The returned value is albedo * cos(i) / pi.
pub fn intensity(map: &Map, point: [f64; 3], source: [f64; 3], roughness: f64) -> Result<f64> {
    let distance = source[0].hypot(source[1]).hypot(source[2]);
    require(
        source.iter().all(|v| v.is_finite()) && distance > 0. && distance.is_finite(),
        "invalid source",
    )?;
    let angle = source[0].atan2(source[1]);
    let b = (-source[2] / distance).clamp(-1., 1.);
    let q = rotation::apply(rotation::axis_angle([0., 0., 1.], angle)?, point);
    let mu = crate::map::dot(&source, &point) / distance;
    if mu <= 0. {
        map.intensity_xyz(point)?;
        return Ok(0.);
    }
    let (degree, p) = illumination_polynomial(b, roughness)?;
    Ok(
        map.intensity_xyz(point)? * crate::map::dot(&p, &basis::polynomial(degree, q)?)
            / (distance * distance),
    )
}
