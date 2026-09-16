//! Rotating luminous/reflected bodies on independent Kepler orbits.
//! Multiple occultor overlaps are integrated as a union, never double-subtracted.
use crate::{
    Map, Result,
    occultation::{self, Integration, Occultor},
    orbit::{C_SOLAR_RADII_PER_DAY, Orbit},
    require,
};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Body {
    pub map: Map,
    pub radius: f64,
    pub mass: f64,
    pub rotation_period: f64,
    pub reference_epoch: f64,
    pub phase0: f64,
    pub inclination: f64,
    pub obliquity: f64,
    pub limb_darkening: Vec<f64>,
    pub reflected: bool,
    pub roughness: f64,
    pub source_samples: usize,
    pub radial_velocity: bool,
    pub equatorial_velocity: f64,
    pub differential_rotation: f64,
    pub flattening: Option<f64>,
    pub gravity: Option<crate::oblate::GravityDarkening>,
    pub normalized: bool,
}
impl Body {
    pub fn new(map: Map, radius: f64, mass: f64) -> Result<Self> {
        require(
            radius.is_finite() && radius >= 0. && mass.is_finite() && mass >= 0.,
            "invalid body radius/mass",
        )?;
        Ok(Self {
            map,
            radius,
            mass,
            rotation_period: 1.,
            reference_epoch: 0.,
            phase0: 0.,
            inclination: PI / 2.,
            obliquity: 0.,
            limb_darkening: vec![],
            reflected: false,
            roughness: 0.,
            source_samples: 1,
            radial_velocity: false,
            equatorial_velocity: 0.,
            differential_rotation: 0.,
            flattening: None,
            gravity: None,
            normalized: true,
        })
    }
    fn projected(&self, t: f64) -> Result<Map> {
        require(
            self.radius.is_finite()
                && self.radius >= 0.
                && self.mass.is_finite()
                && self.mass >= 0.
                && self.rotation_period.is_finite()
                && self.reference_epoch.is_finite()
                && self.phase0.is_finite(),
            "invalid body parameters",
        )?;
        self.map
            .projected(self.inclination, self.obliquity, self.phase(t))
    }
    pub fn axis_ratio(&self) -> Result<f64> {
        let f = self.flattening.unwrap_or(0.);
        require(
            f.is_finite() && (0. ..1.).contains(&f),
            "invalid body flattening",
        )?;
        Ok((1. - f * (2. - f) * self.inclination.sin().powi(2)).sqrt())
    }
    pub(crate) fn phase(&self, t: f64) -> f64 {
        if self.rotation_period == 0. {
            return self.phase0;
        }
        self.phase0 + 2. * PI * ((t - self.reference_epoch) / self.rotation_period).rem_euclid(1.)
    }
}
#[derive(Clone, Debug)]
pub struct Secondary {
    pub body: Body,
    pub orbit: Orbit,
}
#[derive(Clone, Debug)]
pub struct System {
    pub primary: Body,
    pub secondaries: Vec<Secondary>,
    pub light_delay: bool,
    pub source_convention: bool,
    pub exposure: f64,
    pub exposure_samples: usize,
    /// 0: midpoint; 1: trapezoid; 2: Simpson. Endpoint rules use an odd count.
    pub exposure_order: u32,
}
impl System {
    pub fn new(primary: Body, secondaries: Vec<Secondary>) -> Result<Self> {
        require(!primary.reflected, "primary must be an emission source")?;
        require(
            primary.flattening.is_none()
                || (!primary.radial_velocity
                    && secondaries.iter().all(|s| !s.body.radial_velocity)),
            "rotational RV with an oblate primary is not supported",
        )?;
        for s in &secondaries {
            s.orbit.validate()?;
            require(
                s.body.flattening.is_none(),
                "upstream does not support oblate secondary bodies",
            )?;
        }
        Ok(Self {
            primary,
            secondaries,
            light_delay: false,
            source_convention: false,
            exposure: 0.,
            exposure_samples: 31,
            exposure_order: 0,
        })
    }
    pub fn bodies(&self) -> Vec<&Body> {
        std::iter::once(&self.primary)
            .chain(self.secondaries.iter().map(|s| &s.body))
            .collect()
    }
    /// System barycentric coordinates. These are independent Kepler trajectories,
    /// with the primary displacement chosen to keep the center of mass at zero.
    pub fn states(&self, time: f64) -> Result<Vec<([f64; 3], [f64; 3])>> {
        require(time.is_finite(), "nonfinite observation time")?;
        let mut relative = vec![];
        let mut position = [0.; 3];
        let mut velocity = [0.; 3];
        let mut mass = self.primary.mass;
        for s in &self.secondaries {
            require(s.body.mass.is_finite() && s.body.mass >= 0., "invalid mass")?;
            let state = s.orbit.state(time)?;
            mass += s.body.mass;
            for k in 0..3 {
                position[k] -= s.body.mass * state.0[k];
                velocity[k] -= s.body.mass * state.1[k];
            }
            relative.push(state);
        }
        if mass > 0. {
            for k in 0..3 {
                position[k] /= mass;
                velocity[k] /= mass;
            }
        }
        let mut states = vec![(position, velocity)];
        states.extend(relative.into_iter().map(|(p, v)| {
            (
                std::array::from_fn(|k| p[k] + position[k]),
                std::array::from_fn(|k| v[k] + velocity[k]),
            )
        }));
        Ok(states)
    }
    pub(crate) fn apparent(&self, time: f64) -> Result<(Vec<[f64; 3]>, Vec<f64>)> {
        if self.source_convention {
            let mut positions = vec![[0.; 3]];
            for s in &self.secondaries {
                positions.push(
                    crate::source_orbits::position(&s.orbit, time, 1., self.light_delay)?
                        .map(|j| j.value),
                );
            }
            return Ok((positions, vec![time; self.secondaries.len() + 1]));
        }
        let mut times = vec![time; self.secondaries.len() + 1];
        let mut positions = vec![[0.; 3]; times.len()];
        for i in 0..times.len() {
            let mut converged = !self.light_delay;
            for _ in 0..20 {
                positions[i] = self.states(times[i])?[i].0;
                if !self.light_delay {
                    break;
                }
                let next = time + positions[i][2] / C_SOLAR_RADII_PER_DAY;
                if (next - times[i]).abs() < 1e-12 {
                    times[i] = next;
                    converged = true;
                    break;
                }
                times[i] = next;
            }
            require(converged, "light travel time iteration did not converge")?;
            positions[i] = self.states(times[i])?[i].0;
        }
        Ok((positions, times))
    }
    pub fn flux_instantaneous(&self, time: f64) -> Result<Vec<f64>> {
        let (positions, times) = self.apparent(time)?;
        let bodies = self.bodies();
        let mut fluxes = vec![];
        for (i, body) in bodies.iter().enumerate() {
            let map = body.projected(times[i])?;
            let mut foreground = vec![];
            let mut silhouettes = vec![];
            let mut point_blocked = false;
            for (j, other) in bodies.iter().enumerate() {
                if i == j || positions[j][2] <= positions[i][2] || other.radius == 0. {
                    continue;
                }
                let x = positions[j][0] - positions[i][0];
                let y = positions[j][1] - positions[i][1];
                if body.radius == 0. {
                    let ellipse = crate::conic::Ellipse {
                        x,
                        y,
                        major: other.radius,
                        minor: other.radius * other.axis_ratio()?,
                        angle: other.obliquity,
                    };
                    if ellipse.contains(0., 0.) {
                        point_blocked = true;
                    }
                    continue;
                }
                if x.hypot(y) < body.radius + other.radius {
                    silhouettes.push(crate::conic::Ellipse {
                        x: x / body.radius,
                        y: y / body.radius,
                        major: other.radius / body.radius,
                        minor: other.radius * other.axis_ratio()? / body.radius,
                        angle: other.obliquity,
                    });
                    foreground.push(Occultor::new(
                        x / body.radius,
                        y / body.radius,
                        other.radius / body.radius,
                    )?);
                }
            }
            let flux = if body.reflected {
                if body.radius == 0. {
                    0.
                } else {
                    let source =
                        std::array::from_fn(|k| (positions[0][k] - positions[i][k]) / body.radius);
                    if silhouettes.iter().any(|e| e.major != e.minor) {
                        self.primary.map.amplitude()
                            * crate::reflected::flux_extended_ellipses(
                                &map.limb_filtered(&body.limb_darkening)?,
                                source,
                                self.primary.radius / body.radius,
                                body.source_samples,
                                body.roughness,
                                &silhouettes,
                            )?
                    } else {
                        self.primary.map.amplitude()
                            * crate::reflected::flux_extended(
                                &map.limb_filtered(&body.limb_darkening)?,
                                source,
                                self.primary.radius / body.radius,
                                body.source_samples,
                                body.roughness,
                                &foreground,
                            )?
                    }
                }
            } else if let Some(flattening) = body.flattening {
                let view = crate::oblate::observed_map(
                    &body.map,
                    flattening,
                    body.inclination,
                    body.phase(times[i]),
                    &body.limb_darkening,
                    body.gravity,
                    body.normalized,
                )?;
                let ellipses: Vec<_> = silhouettes
                    .iter()
                    .map(|e| e.rotated(-body.obliquity))
                    .collect();
                if view.degree() > 12 {
                    fluxes.push(crate::quadrature::surface(
                        body.axis_ratio()?,
                        -1.,
                        &ellipses,
                        2e-11,
                        |p| view.intensity_xyz(p),
                    )?);
                    continue;
                }
                let moments = occultation::projected_moments(
                    view.degree(),
                    body.axis_ratio()?,
                    &ellipses,
                    Integration::default(),
                )?
                .values;
                view.amplitude() * crate::map::dot(&view.polynomial_coefficients()?, &moments)
            } else if point_blocked {
                0.
            } else if silhouettes.iter().any(|e| e.major != e.minor) {
                let view = map.limb_filtered(&body.limb_darkening)?;
                if view.degree() > 12 {
                    fluxes.push(crate::quadrature::surface(
                        1.,
                        -1.,
                        &silhouettes,
                        2e-11,
                        |p| view.intensity_xyz(p),
                    )?);
                    continue;
                }
                let moments = occultation::projected_moments(
                    view.degree(),
                    1.,
                    &silhouettes,
                    Integration::default(),
                )?
                .values;
                view.amplitude() * crate::map::dot(&view.polynomial_coefficients()?, &moments)
            } else if foreground.len() <= 1 {
                map.flux_limb_darkened(&body.limb_darkening, foreground.first().copied())?
            } else if map.degree() + body.limb_darkening.len() > 12 {
                let row = crate::rings::flux_design_many(
                    map.degree(),
                    &foreground,
                    &body.limb_darkening,
                    Integration::default(),
                )?;
                map.amplitude() * crate::map::dot(&row, map.coefficients())
            } else {
                let (limb, norm) = crate::map::limb_polynomial(&body.limb_darkening)?;
                let d = map.degree() + body.limb_darkening.len();
                require(
                    d <= crate::basis::MAX_DEGREE,
                    "combined body degree exceeds 20",
                )?;
                let p = crate::basis::multiply(
                    &map.polynomial_coefficients()?,
                    map.degree(),
                    &limb,
                    body.limb_darkening.len(),
                );
                let moments =
                    occultation::union_moments(d, &foreground, Integration::default())?.values;
                map.amplitude() * crate::map::dot(&p, &moments) / norm
            };
            fluxes.push(flux);
        }
        Ok(fluxes)
    }
    pub fn exposure_stencil(&self) -> Result<Vec<(f64, f64)>> {
        require(
            self.exposure.is_finite()
                && self.exposure >= 0.
                && self.exposure_samples > 0
                && self.exposure_samples <= 100001
                && self.exposure_order <= 2,
            "invalid exposure settings",
        )?;
        if self.exposure == 0. || self.exposure_samples == 1 {
            return Ok(vec![(0., 1.)]);
        }
        let n = if self.exposure_order == 0 {
            self.exposure_samples
        } else {
            self.exposure_samples | 1
        };
        let mut stencil = Vec::with_capacity(n);
        for j in 0..n {
            let (offset, weight) = match self.exposure_order {
                0 => ((j as f64 + 0.5) / n as f64 - 0.5, 1.),
                1 => (
                    j as f64 / (n - 1) as f64 - 0.5,
                    if j == 0 || j == n - 1 { 1. } else { 2. },
                ),
                _ => (
                    j as f64 / (n - 1) as f64 - 0.5,
                    if j == 0 || j == n - 1 {
                        1.
                    } else if j % 2 == 1 {
                        4.
                    } else {
                        2.
                    },
                ),
            };
            stencil.push((self.exposure * offset, weight));
        }
        let sum: f64 = stencil.iter().map(|p| p.1).sum();
        for p in &mut stencil {
            p.1 /= sum;
        }
        Ok(stencil)
    }
    pub fn flux(&self, time: f64) -> Result<Vec<f64>> {
        let stencil = self.exposure_stencil()?;
        if self.exposure == 0. {
            return self.flux_instantaneous(time);
        }
        let mut flux = vec![0.; self.secondaries.len() + 1];
        for (dt, weight) in stencil {
            for (a, b) in flux.iter_mut().zip(self.flux_instantaneous(time + dt)?) {
                *a += b * weight;
            }
        }
        Ok(flux)
    }
    pub fn light_curve(&self, times: &[f64]) -> Result<Vec<f64>> {
        times
            .iter()
            .map(|&t| Ok(self.flux(t)?.iter().sum()))
            .collect()
    }
    /// Per-body rotational/Rossiter-McLaughlin velocities, optionally including
    /// orbital velocities, in m/s (positive away from the observer).
    pub fn radial_velocities(&self, time: f64, keplerian: bool) -> Result<Vec<f64>> {
        let stencil = self.exposure_stencil()?;
        if self.exposure == 0. {
            return self.radial_velocities_instantaneous(time, keplerian);
        }
        let mut numerators = vec![0.; self.secondaries.len() + 1];
        let mut denominators = numerators.clone();
        for (dt, weight) in stencil {
            let t = time + dt;
            let velocities = self.radial_velocities_instantaneous(t, false)?;
            let fluxes = self.flux_instantaneous(t)?;
            for i in 0..numerators.len() {
                numerators[i] += weight * velocities[i] * fluxes[i];
                denominators[i] += weight * fluxes[i];
            }
        }
        for i in 0..numerators.len() {
            numerators[i] = if denominators[i] == 0. {
                0.
            } else {
                numerators[i] / denominators[i]
            };
        }
        if keplerian {
            let full = self.radial_velocities_instantaneous(time, true)?;
            let rotational = self.radial_velocities_instantaneous(time, false)?;
            for i in 0..numerators.len() {
                numerators[i] += full[i] - rotational[i];
            }
        }
        Ok(numerators)
    }

    pub fn radial_velocities_instantaneous(&self, time: f64, keplerian: bool) -> Result<Vec<f64>> {
        let (positions, times) = self.apparent(time)?;
        let bodies = self.bodies();
        let mut values = Vec::with_capacity(bodies.len());
        for (i, body) in bodies.iter().enumerate() {
            let mut value = if keplerian && self.source_convention {
                crate::source_orbits::reflex(self, time)?[i]
            } else if keplerian {
                -self.states(times[i])?[i].1[2] * crate::orbit::SOLAR_RADIUS_METERS / 86400.
            } else {
                0.
            };
            if body.radial_velocity && body.radius > 0. {
                require(!body.reflected, "reflected RV body is unsupported")?;
                let mut occultors = vec![];
                for (j, other) in bodies.iter().enumerate() {
                    if i != j && positions[j][2] > positions[i][2] && other.radius > 0. {
                        occultors.push(Occultor::new(
                            (positions[j][0] - positions[i][0]) / body.radius,
                            (positions[j][1] - positions[i][1]) / body.radius,
                            other.radius / body.radius,
                        )?);
                    }
                }
                value += crate::rv::radial_velocity_many(
                    &body.projected(times[i])?,
                    body.inclination,
                    body.obliquity,
                    body.equatorial_velocity,
                    body.differential_rotation,
                    &body.limb_darkening,
                    &occultors,
                )?;
            }
            values.push(value);
        }
        Ok(values)
    }
}
