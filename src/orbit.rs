//! Elliptic Kepler orbits, in days, solar radii, solar masses, and radians.
//! Observer axes follow exoplanet's KeplerianOrbit (used by upstream starry).
use crate::{Result, require};
use std::f64::consts::PI;
pub const SOLAR_RADIUS_METERS: f64 = 6.957e8;
pub const DAY_SECONDS: f64 = 86400.;
pub const GM_SUN: f64 = 1.3271244e20;
pub const C_SOLAR_RADII_PER_DAY: f64 = 299792458. * DAY_SECONDS / SOLAR_RADIUS_METERS;

#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub semimajor_axis: f64,
    pub period: f64,
    pub eccentricity: f64,
    pub inclination: f64,
    pub omega: f64,
    pub ascending_node: f64,
    pub transit_epoch: f64,
}
impl Orbit {
    pub fn circular(semimajor_axis: f64, period: f64) -> Result<Self> {
        let o = Self {
            semimajor_axis,
            period,
            eccentricity: 0.,
            inclination: PI / 2.,
            omega: PI / 2.,
            ascending_node: 0.,
            transit_epoch: 0.,
        };
        o.validate()?;
        Ok(o)
    }
    pub fn from_period(period: f64, total_mass: f64) -> Result<Self> {
        require(
            total_mass.is_finite() && total_mass > 0. && period.is_finite() && period > 0.,
            "invalid orbit period/mass",
        )?;
        let a = (GM_SUN * total_mass * (period * DAY_SECONDS / (2. * PI)).powi(2)).cbrt()
            / SOLAR_RADIUS_METERS;
        Self::circular(a, period)
    }
    pub fn from_semimajor_axis(a: f64, total_mass: f64) -> Result<Self> {
        require(
            total_mass.is_finite() && total_mass > 0. && a.is_finite() && a > 0.,
            "invalid orbit axis/mass",
        )?;
        Self::circular(
            a,
            2. * PI * ((a * SOLAR_RADIUS_METERS).powi(3) / (GM_SUN * total_mass)).sqrt()
                / DAY_SECONDS,
        )
    }
    pub fn validate(&self) -> Result<()> {
        require(
            [
                self.semimajor_axis,
                self.period,
                self.eccentricity,
                self.inclination,
                self.omega,
                self.ascending_node,
                self.transit_epoch,
            ]
            .iter()
            .all(|v| v.is_finite())
                && self.semimajor_axis > 0.
                && self.period > 0.
                && (0. ..1.).contains(&self.eccentricity)
                && (0. ..=PI).contains(&self.inclination),
            "invalid Keplerian elements",
        )
    }
    fn rotate(&self, x: f64, y: f64) -> [f64; 3] {
        let (s, c) = self.omega.sin_cos();
        let x1 = c * x - s * y;
        let y1 = s * x + c * y;
        let y2 = self.inclination.cos() * y1;
        let z = -self.inclination.sin() * y1;
        let (s, c) = self.ascending_node.sin_cos();
        [c * x1 - s * y2, s * x1 + c * y2, z]
    }
    pub fn state(&self, time: f64) -> Result<([f64; 3], [f64; 3])> {
        self.validate()?;
        require(time.is_finite(), "nonfinite observation time")?;
        let e = self.eccentricity;
        let f0 = PI / 2. - self.omega;
        let e0 = 2. * ((1. - e).sqrt() * (f0 / 2.).sin()).atan2((1. + e).sqrt() * (f0 / 2.).cos());
        let mean = (e0 - e * e0.sin()
            + 2. * PI * ((time - self.transit_epoch) / self.period).rem_euclid(1.)
            + PI)
            .rem_euclid(2. * PI)
            - PI;
        let eccentric = solve_kepler(mean, e)?;
        let (s, c) = eccentric.sin_cos();
        let q = (1. - e * e).sqrt();
        let a = self.semimajor_axis;
        let edot = 2. * PI / (self.period * (1. - e * c));
        Ok((
            self.rotate(-a * (c - e), -a * q * s),
            self.rotate(a * s * edot, -a * q * c * edot),
        ))
    }

    /// Position/velocity values, Jacobians and Hessians. Parameter order:
    /// a, period, eccentricity, inclination, omega, node, transit_epoch, time.
    /// The semimajor axis and period are independent elements here.
    pub fn state_derivatives(&self, time: f64) -> Result<[crate::autodiff::Jet<8>; 6]> {
        use crate::autodiff::Jet;
        self.validate()?;
        require(time.is_finite(), "nonfinite observation time")?;
        let [a, period, e, inc, omega, node, epoch, t] = std::array::from_fn(|i| {
            Jet::variable(
                [
                    self.semimajor_axis,
                    self.period,
                    self.eccentricity,
                    self.inclination,
                    self.omega,
                    self.ascending_node,
                    self.transit_epoch,
                    time,
                ][i],
                i,
            )
        });
        let constant = Jet::constant;
        let f0 = constant(PI / 2.) - omega;
        let e0 = constant(2.)
            * ((constant(1.) - e).sqrt() * (f0 / constant(2.)).sin())
                .atan2((constant(1.) + e).sqrt() * (f0 / constant(2.)).cos());
        let mut phase = (t - epoch) / period;
        phase.value = phase.value.rem_euclid(1.);
        let mut mean = e0 - e * e0.sin() + constant(2. * PI) * phase;
        mean.value = (mean.value + PI).rem_euclid(2. * PI) - PI;
        let value = solve_kepler(mean.value, e.value)?;
        let (s, c) = value.sin_cos();
        let denominator = 1. - e.value * c;
        let mut eccentric = constant(value);
        for i in 0..8 {
            eccentric.gradient[i] = (mean.gradient[i] + s * e.gradient[i]) / denominator;
        }
        for i in 0..8 {
            for j in 0..8 {
                eccentric.hessian[i][j] = (mean.hessian[i][j]
                    + s * e.hessian[i][j]
                    + c * (e.gradient[i] * eccentric.gradient[j]
                        + e.gradient[j] * eccentric.gradient[i])
                    - e.value * s * eccentric.gradient[i] * eccentric.gradient[j])
                    / denominator;
            }
        }
        let s = eccentric.sin();
        let c = eccentric.cos();
        let q = (constant(1.) - e * e).sqrt();
        let edot = constant(2. * PI) / (period * (constant(1.) - e * c));
        let rotate = |x: Jet<8>, y: Jet<8>| {
            let x1 = omega.cos() * x - omega.sin() * y;
            let y1 = omega.sin() * x + omega.cos() * y;
            let y2 = inc.cos() * y1;
            let z = -inc.sin() * y1;
            [
                node.cos() * x1 - node.sin() * y2,
                node.sin() * x1 + node.cos() * y2,
                z,
            ]
        };
        let p = rotate(-a * (c - e), -a * q * s);
        let v = rotate(a * s * edot, -a * q * c * edot);
        let out = [p[0], p[1], p[2], v[0], v[1], v[2]];
        require(
            out.iter().all(|j| {
                j.value.is_finite()
                    && j.gradient
                        .iter()
                        .chain(j.hessian.iter().flatten())
                        .all(|v| v.is_finite())
            }),
            "nonfinite orbital derivatives",
        )?;
        Ok(out)
    }
}

/// Bracketed Newton iteration, robust up to eccentricity just below one.
pub fn solve_kepler(mean: f64, eccentricity: f64) -> Result<f64> {
    require(
        mean.is_finite() && eccentricity.is_finite() && (0. ..1.).contains(&eccentricity),
        "invalid Kepler arguments",
    )?;
    let m = (mean + PI).rem_euclid(2. * PI) - PI;
    let e = eccentricity;
    let mut lo = -PI;
    let mut hi = PI;
    let mut x = if e < 0.8 { m } else { m.signum() * PI / 2. };
    for _ in 0..100 {
        let f = x - e * x.sin() - m;
        if f.abs() < 4e-15 {
            return Ok(x);
        }
        if f > 0. {
            hi = x;
        } else {
            lo = x;
        }
        let candidate = x - f / (1. - e * x.cos());
        x = if candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    Err(crate::Error("Kepler equation did not converge".into()))
}
