//! Oblate projected maps with gravity-darkening spectral filters.
//! Gravity profile/SHT follow OpsOblate. Projected ellipse integration is numerical.
use crate::{
    Map, Result, basis,
    occultation::{Integration, Occultor},
    require,
};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct GravityDarkening {
    pub degree: usize,
    pub omega: f64,
    pub flattening: f64,
    pub beta: f64,
    pub polar_temperature: f64,
    pub wavelength_meters: f64,
}
impl GravityDarkening {
    pub fn profile(&self, z: f64) -> Result<f64> {
        require(
            [
                self.omega,
                self.flattening,
                self.beta,
                self.polar_temperature,
                self.wavelength_meters,
                z,
            ]
            .iter()
            .all(|v| v.is_finite())
                && self.degree <= basis::MAX_DEGREE
                && (0. ..1.).contains(&self.flattening)
                && self.omega >= 0.
                && self.omega <= 1.
                && self.beta >= 0.
                && self.polar_temperature > 0.
                && self.wavelength_meters > 0.
                && z.abs() <= 1.,
            "invalid gravity darkening",
        )?;
        let b = 1. - self.flattening;
        let z2 = z * z;
        let term = (-self.omega * self.omega * (z2 * b * b - z2 + 1.).powf(1.5) + 1.).powi(2);
        let ratio = (-z2 * b * b + (z2 - 1.) * term) / (-z2 * b * b + z2 - 1.).powi(3);
        let temp =
            self.polar_temperature * b.powf(2. * self.beta) * ratio.max(0.).powf(0.5 * self.beta);
        if temp == 0. {
            return Ok(0.);
        }
        Ok(1. / (1.43877735e-2 / (self.wavelength_meters * temp)).exp_m1())
    }
    pub fn filter(&self) -> Result<Map> {
        let n = 4 * (self.degree + 1).pow(2);
        let z: Vec<_> = (0..n)
            .map(|i| -1. + 2. * i as f64 / (n - 1) as f64)
            .collect();
        let values = z
            .iter()
            .map(|&z| self.profile(z))
            .collect::<Result<Vec<_>>>()?;
        let y = crate::surface::axisymmetric_fit(self.degree, &z, &values, 1e-9, 0.)?;
        let mut map = Map::new(self.degree)?;
        map.set_coefficients(&y)?;
        map.rotate([1., 0., 0.], -PI / 2.)?;
        Ok(map)
    }
}

#[allow(clippy::too_many_arguments)] // Explicit physical inputs matching the upstream op.
pub fn flux(
    map: &Map,
    flattening: f64,
    inclination: f64,
    obliquity: f64,
    phase: f64,
    u: &[f64],
    gravity: Option<GravityDarkening>,
    occ: Option<Occultor>,
    normalized: bool,
) -> Result<f64> {
    require(
        flattening.is_finite() && (0. ..1.).contains(&flattening),
        "invalid oblateness",
    )?;
    let q = (1. - flattening * (2. - flattening) * inclination.sin().powi(2)).sqrt();
    if map.degree() + u.len() + gravity.map_or(0, |g| g.degree) > 12 {
        let view = observed_map(map, flattening, inclination, phase, u, gravity, normalized)?;
        let silhouettes = crate::quadrature::circles(&occ.into_iter().collect::<Vec<_>>())?;
        let silhouettes: Vec<_> = silhouettes.iter().map(|e| e.rotated(-obliquity)).collect();
        return crate::quadrature::surface(q, -1., &silhouettes, 2e-11, |p| view.intensity_xyz(p));
    }
    let weighted = if let Some(g) = gravity {
        let filter = g.filter()?;
        let d = map.degree() + g.degree;
        require(
            d <= basis::POLYNOMIAL_LIMIT,
            "combined gravity degree exceeds 20",
        )?;
        let p = basis::multiply(
            &map.polynomial_coefficients()?,
            map.degree(),
            &filter.polynomial_coefficients()?,
            g.degree,
        );
        let rhs = crate::matrix::Matrix {
            rows: p.len(),
            cols: 1,
            data: p,
        };
        let mut result = Map::new(d)?;
        result.set_coefficients(&basis::a1(d)?.solve(&rhs)?.data)?;
        result.set_amplitude(map.amplitude() * PI)?;
        result
    } else {
        map.clone()
    };
    let view = weighted.projected(inclination, 0., phase)?;
    let (limb, norm) = crate::map::limb_polynomial(u)?;
    let degree = view.degree() + u.len();
    require(
        degree <= basis::POLYNOMIAL_LIMIT,
        "combined oblate degree exceeds 20",
    )?;
    let p = basis::multiply(
        &view.polynomial_coefficients()?,
        view.degree(),
        &limb,
        u.len(),
    );
    let aligned = occ.map(|o| {
        let (s, c) = obliquity.sin_cos();
        Occultor {
            x: c * o.x + s * o.y,
            y: -s * o.x + c * o.y,
            radius: o.radius,
        }
    });
    let moments =
        crate::occultation::ellipse_moments(degree, q, aligned, Integration::default())?.values;
    let mut value = view.amplitude() * crate::map::dot(&p, &moments) / norm;
    if let Some(g) = gravity {
        value *= 1.19104295e-16 / g.wavelength_meters.powi(5);
    }
    if normalized {
        let uniform = Map::new(0)?;
        let base = flux(
            &uniform,
            flattening,
            inclination,
            obliquity,
            0.,
            u,
            gravity,
            None,
            false,
        )?;
        require(base > 0., "nonpositive oblate normalization")?;
        value /= base;
    }
    Ok(value)
}

/// Observer-frame intensity map in ellipse coordinates (x,y/q,z). Obliquity
/// rotates the silhouette and coordinates separately. Its disk moments carry q.
pub fn observed_map(
    map: &Map,
    flattening: f64,
    inclination: f64,
    phase: f64,
    u: &[f64],
    gravity: Option<GravityDarkening>,
    normalized: bool,
) -> Result<Map> {
    require(
        flattening.is_finite() && (0. ..1.).contains(&flattening),
        "invalid oblateness",
    )?;
    let weighted = if let Some(g) = gravity {
        let filter = g.filter()?;
        let degree = map.degree() + g.degree;
        require(
            degree <= basis::MAX_DEGREE,
            "combined gravity degree exceeds 32",
        )?;
        let mut weighted = Map::new(degree)?;
        if degree > 12 {
            // Project a product of stable harmonic fields rather than invert
            // the ill-conditioned monomial transform at high degree.
            let mut unit = map.clone();
            unit.set_amplitude(1.)?;
            weighted.set_coefficients(&crate::rotation::project_function(degree, |p| {
                Ok(unit.intensity_xyz(p)? * filter.intensity_xyz(p)?)
            })?)?;
        } else {
            let product = basis::multiply(
                &map.polynomial_coefficients()?,
                map.degree(),
                &filter.polynomial_coefficients()?,
                g.degree,
            );
            weighted.set_coefficients(
                &basis::a1(degree)?
                    .solve(&crate::matrix::Matrix {
                        rows: product.len(),
                        cols: 1,
                        data: product,
                    })?
                    .data,
            )?;
        }
        weighted
            .set_amplitude(map.amplitude() * PI * 1.19104295e-16 / g.wavelength_meters.powi(5))?;
        weighted
    } else {
        map.clone()
    };
    let mut projected = weighted
        .projected(inclination, 0., phase)?
        .limb_filtered(u)?;
    if normalized {
        let baseline = flux(
            &Map::new(0)?,
            flattening,
            inclination,
            0.,
            0.,
            u,
            gravity,
            None,
            false,
        )?;
        require(baseline > 0., "nonpositive oblate normalization")?;
        projected.set_amplitude(projected.amplitude() / baseline)?;
    }
    Ok(projected)
}
