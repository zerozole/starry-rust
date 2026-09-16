//! Emitted-light maps with starry's real spherical harmonic convention.
use crate::{
    Result, basis,
    matrix::Matrix,
    occultation::{self, Integration, Occultor},
    require, rotation,
};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Map {
    degree: usize,
    coefficients: Vec<f64>,
    polynomial: Vec<f64>,
    a1: Matrix,
    green_transform: Matrix,
    amplitude: f64,
}
impl Map {
    pub(crate) fn polynomial_coefficients(&self) -> Result<Vec<f64>> {
        Ok(self.polynomial.clone())
    }
    /// Transform a body-frame map using starry's inclination, obliquity, phase.
    pub fn projected(&self, inclination: f64, obliquity: f64, phase: f64) -> Result<Self> {
        require(
            inclination.is_finite()
                && (0. ..=PI).contains(&inclination)
                && obliquity.is_finite()
                && phase.is_finite(),
            "invalid viewing orientation",
        )?;
        let mut map = self.clone();
        map.rotate([0., 1., 0.], phase)?;
        map.rotate([1., 0., 0.], PI / 2. - inclination)?;
        map.rotate([0., 0., 1.], obliquity)?;
        Ok(map)
    }
    pub fn flux_at(
        &self,
        phase: f64,
        inclination: f64,
        obliquity: f64,
        occ: Option<Occultor>,
    ) -> Result<f64> {
        self.projected(inclination, obliquity, phase)?.flux(occ)
    }
    pub fn new(degree: usize) -> Result<Self> {
        let a1 = basis::a1(degree)?;
        let mut coefficients = vec![0.; a1.cols];
        coefficients[0] = 1.;
        let green_transform = if degree <= 12 {
            basis::a2_inverse(degree)?.solve(&a1)?
        } else {
            Matrix::zeros(0, 0)
        };
        let polynomial = a1.dot(&coefficients)?;
        Ok(Self {
            degree,
            coefficients,
            polynomial,
            a1,
            green_transform,
            amplitude: 1.,
        })
    }
    pub fn degree(&self) -> usize {
        self.degree
    }
    /// Exact-degree harmonic representation of a product of two surface fields.
    pub fn multiplied(&self, other: &Self) -> Result<Self> {
        let degree = self.degree + other.degree;
        require(
            degree <= basis::MAX_DEGREE,
            "combined product degree exceeds32",
        )?;
        let mut result = Self::new(degree)?;
        if degree > 12 {
            result.set_coefficients(&rotation::project_function(degree, |p| {
                Ok(dot(&basis::harmonics(self.degree, p)?, &self.coefficients)
                    * dot(&basis::harmonics(other.degree, p)?, &other.coefficients))
            })?)?;
        } else {
            let p = basis::multiply(
                &self.polynomial_coefficients()?,
                self.degree,
                &other.polynomial_coefficients()?,
                other.degree,
            );
            let rhs = Matrix {
                rows: p.len(),
                cols: 1,
                data: p,
            };
            result.set_coefficients(&result.a1.solve(&rhs)?.data)?;
        }
        result.set_amplitude(self.amplitude * other.amplitude)?;
        Ok(result)
    }
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }
    pub fn set_amplitude(&mut self, amplitude: f64) -> Result<()> {
        require(amplitude.is_finite(), "amplitude must be finite")?;
        self.amplitude = amplitude;
        Ok(())
    }
    /// Set raw coefficients, including Y00 (no implicit amplitude renormalization).
    pub fn set_coefficients(&mut self, coefficients: &[f64]) -> Result<()> {
        require(
            coefficients.len() == self.coefficients.len()
                && coefficients.iter().all(|v| v.is_finite()),
            "invalid coefficients",
        )?;
        let polynomial = self.a1.dot(coefficients)?;
        require(
            polynomial.iter().all(|v| v.is_finite()),
            "nonfinite map polynomial",
        )?;
        self.coefficients.copy_from_slice(coefficients);
        self.polynomial = polynomial;
        Ok(())
    }
    pub fn set(&mut self, l: usize, m: isize, value: f64) -> Result<()> {
        require(
            l <= self.degree && m.unsigned_abs() <= l && value.is_finite(),
            "invalid harmonic coefficient",
        )?;
        let mut coefficients = self.coefficients.clone();
        coefficients[basis::index(l, m)] = value;
        self.set_coefficients(&coefficients)
    }
    /// Unit-sphere Cartesian coordinates: x east, y north, z toward observer.
    pub fn intensity_design(&self, xyz: [f64; 3]) -> Result<Vec<f64>> {
        require(
            xyz.iter().all(|v| v.is_finite())
                && (xyz.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-10,
            "intensity point must lie on unit sphere",
        )?;
        basis::harmonics(self.degree, xyz)
    }
    pub fn intensity_xyz(&self, xyz: [f64; 3]) -> Result<f64> {
        require(
            xyz.iter().all(|v| v.is_finite())
                && (xyz.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-10,
            "intensity point must lie on unit sphere",
        )?;
        Ok(self.amplitude * dot(&basis::harmonics(self.degree, xyz)?, &self.coefficients))
    }
    /// Latitude/longitude in radians, lon=0 on the +z meridian (as in starry).
    pub fn intensity(&self, latitude: f64, longitude: f64) -> Result<f64> {
        require(
            latitude.is_finite() && longitude.is_finite() && latitude.abs() <= PI / 2.,
            "invalid latitude/longitude",
        )?;
        self.intensity_xyz([
            latitude.cos() * longitude.sin(),
            latitude.sin(),
            latitude.cos() * longitude.cos(),
        ])
    }
    /// Analytic derivatives with respect to latitude and longitude (radians).
    pub fn intensity_gradient(&self, lat: f64, lon: f64) -> Result<[f64; 2]> {
        self.intensity(lat, lon)?;
        let xyz = [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()];
        let tangents = [
            [-lat.sin() * lon.sin(), lat.cos(), -lat.sin() * lon.cos()],
            [lat.cos() * lon.cos(), 0., -lat.cos() * lon.sin()],
        ];
        let a = basis::harmonic_directional(self.degree, xyz, tangents[0])?;
        let b = basis::harmonic_directional(self.degree, xyz, tangents[1])?;
        Ok([
            dot(&a, &self.coefficients) * self.amplitude,
            dot(&b, &self.coefficients) * self.amplitude,
        ])
    }
    /// Rotate the map actively. Map values satisfy I_new(p)=I_old(R^-1 p).
    pub fn rotate(&mut self, axis: [f64; 3], angle: f64) -> Result<()> {
        let coefficients = rotation::coefficients(self.degree, &self.coefficients, axis, angle)?;
        self.set_coefficients(&coefficients)
    }
    /// Linear flux response to each coefficient; excludes amplitude.
    pub fn flux_design_numerical(
        &self,
        occ: Option<Occultor>,
        integration: Integration,
    ) -> Result<Vec<f64>> {
        if self.degree > 12 {
            return crate::rings::flux_design(self.degree, occ, &[], integration);
        }
        let moments = match occ {
            Some(o) => occultation::visible_moments(self.degree, o, integration)?.values,
            None => basis::disk_moments(self.degree)?,
        };
        self.a1.left_dot(&moments)
    }
    /// Emitted-light response, excluding amplitude. Uses stable harmonic-ring
    /// integration above degree 12 to avoid polynomial cancellation.
    pub fn flux_design(&self, occ: Option<Occultor>) -> Result<Vec<f64>> {
        if self.degree > 12 {
            return crate::rings::flux_design(self.degree, occ, &[], Integration::default());
        }
        match occ {
            Some(o) => {
                crate::solver::flux_design_cached(&self.a1, &self.green_transform, self.degree, o)
            }
            None => self.a1.left_dot(&basis::disk_moments(self.degree)?),
        }
    }
    pub fn flux(&self, occ: Option<Occultor>) -> Result<f64> {
        Ok(self.amplitude * dot(&self.flux_design(occ)?, &self.coefficients))
    }
    /// Derivatives with respect to foreground occultor x, y, radius.
    pub fn flux_gradient(&self, occ: Occultor) -> Result<[f64; 3]> {
        let rows = crate::gradients::flux_design(self.degree, &self.a1, occ, 1e-12)?;
        Ok(rows.map(|row| self.amplitude * dot(&row, &self.coefficients)))
    }
    /// Apply limb darkening in the observer frame, after any map rotations.
    /// The profile is normalized by its uniform-map disk average, as in starry.
    pub fn flux_limb_darkened(&self, u: &[f64], occ: Option<Occultor>) -> Result<f64> {
        require(
            self.degree + u.len() <= basis::MAX_DEGREE,
            "combined map and limb degree exceeds 32",
        )?;
        if self.degree + u.len() > 12 {
            return Ok(self.amplitude
                * dot(
                    &crate::rings::flux_design(self.degree, occ, u, Integration::default())?,
                    &self.coefficients,
                ));
        }
        let (limb, norm) = limb_polynomial(u)?;
        let p = basis::multiply(
            &self.a1.dot(&self.coefficients)?,
            self.degree,
            &limb,
            u.len(),
        );
        Ok(self.amplitude * crate::solver::polynomial_flux(self.degree + u.len(), &p, occ)? / norm)
    }
    /// Materialize an observer-frame normalized limb filter as a harmonic map.
    pub fn limb_filtered(&self, u: &[f64]) -> Result<Self> {
        if u.is_empty() {
            return Ok(self.clone());
        }
        let degree = self.degree + u.len();
        require(
            degree <= basis::MAX_DEGREE,
            "combined limb degree exceeds 32",
        )?;
        let (limb, norm) = limb_polynomial(u)?;
        if degree > 12 {
            let y = rotation::project_function(degree, |xyz| {
                let profile = 1.
                    - u.iter()
                        .enumerate()
                        .map(|(i, v)| v * (1. - xyz[2]).powi(i as i32 + 1))
                        .sum::<f64>();
                Ok(dot(&basis::harmonics(self.degree, xyz)?, &self.coefficients) * profile / norm)
            })?;
            let mut map = Self::new(degree)?;
            map.set_coefficients(&y)?;
            map.set_amplitude(self.amplitude)?;
            return Ok(map);
        }
        let p = basis::multiply(
            &self.polynomial_coefficients()?,
            self.degree,
            &limb,
            u.len(),
        );
        let rhs = Matrix {
            rows: p.len(),
            cols: 1,
            data: p,
        };
        let mut map = Self::new(degree)?;
        map.set_coefficients(&map.a1.solve(&rhs)?.data)?;
        map.set_amplitude(self.amplitude / norm)?;
        Ok(map)
    }
    /// Batch of single-occultor geometries. Occultors must be in front of the map.
    pub fn light_curve(&self, occultors: &[Occultor]) -> Result<Vec<f64>> {
        occultors.iter().map(|&o| self.flux(Some(o))).collect()
    }
}
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Center-normalized limb profile: 1 - sum u[n-1]*(1-mu)^n.
pub fn limb_intensity(mu: f64, u: &[f64]) -> Result<f64> {
    require(
        mu.is_finite()
            && (0. ..=1.).contains(&mu)
            && u.iter().all(|v| v.is_finite())
            && u.len() <= basis::MAX_DEGREE,
        "invalid limb profile",
    )?;
    Ok(1.
        - u.iter()
            .enumerate()
            .map(|(i, v)| v * (1. - mu).powi(i as i32 + 1))
            .sum::<f64>())
}

/// Unit-flux normalization of a polynomial limb-darkened star.
pub(crate) fn limb_polynomial(u: &[f64]) -> Result<(Vec<f64>, f64)> {
    limb_intensity(1., u)?;
    let norm = 1.
        - u.iter()
            .enumerate()
            .map(|(i, v)| 2. * v / ((i + 2) * (i + 3)) as f64)
            .sum::<f64>();
    require(
        norm.is_finite() && norm > 0.,
        "limb profile has nonpositive disk integral",
    )?;
    // Build 1 - sum u_n (1-z)^n in the canonical basis.
    let base = vec![1., 0., -1., 0.];
    let mut pow = vec![1.];
    let mut p = vec![0.; (u.len() + 1).pow(2)];
    p[0] = 1.;
    for (k, &v) in u.iter().enumerate() {
        pow = basis::multiply(&pow, k, &base, 1);
        for (j, &term) in pow.iter().enumerate() {
            p[j] -= v * term;
        }
    }
    Ok((p, norm))
}

pub fn limb_flux(u: &[f64], occ: Option<Occultor>) -> Result<f64> {
    let (p, norm) = limb_polynomial(u)?;
    Ok(crate::solver::polynomial_flux(u.len(), &p, occ)? / (PI * norm))
}
