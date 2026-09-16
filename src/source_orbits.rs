//! The independent-pair conventions used by starry's exoplanet adapter.
use crate::{
    Result,
    autodiff::Jet,
    orbit::{C_SOLAR_RADII_PER_DAY, Orbit},
    system::System,
};

/// Exoplanet's quadratic light-time approximation, rationalized to avoid
/// cancellation. Scale is one for relative positions, m_star/m_total for planets.
/// Only values and first gradients are consumed; Hessians are not composed here.
pub fn position(orbit: &Orbit, time: f64, scale: f64, delayed: bool) -> Result<[Jet<8>; 3]> {
    let state = orbit.state_derivatives(time)?;
    let mut shift = Jet::constant(0.);
    if delayed {
        let z = state[2] * Jet::constant(scale);
        let vz = state[5] * Jet::constant(scale);
        let radius = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
        let n = Jet::constant(2. * std::f64::consts::PI) / Jet::variable(orbit.period, 1);
        let az = -n.powi(2) * (Jet::variable(orbit.semimajor_axis, 0) / radius).powi(3) * z;
        let c = Jet::constant(C_SOLAR_RADII_PER_DAY) - vz;
        shift = if az.value.abs() < 1e-10 {
            z / c
        } else {
            z * Jet::constant(2.) / (c + (c.powi(2) + az * z * Jet::constant(2.)).sqrt())
        };
    }
    let retarded = orbit.state_derivatives(time + shift.value)?;
    Ok(std::array::from_fn(|axis| {
        let mut result = retarded[axis] * Jet::constant(scale);
        for k in 0..8 {
            result.gradient[k] += retarded[axis].gradient[7] * scale * shift.gradient[k];
        }
        result
    }))
}

pub fn positions(system: &System, time: f64) -> Result<Vec<[f64; 3]>> {
    let mut result = vec![[0.; 3]];
    for secondary in &system.secondaries {
        let total = system.primary.mass + secondary.body.mass;
        let state = secondary.orbit.state(time)?;
        for (k, value) in result[0].iter_mut().enumerate() {
            *value -= secondary.body.mass / total * state.0[k];
        }
        result.push(
            position(
                &secondary.orbit,
                time,
                system.primary.mass / total,
                system.light_delay,
            )?
            .map(|j| j.value),
        );
    }
    Ok(result)
}

pub fn reflex(system: &System, time: f64) -> Result<Vec<f64>> {
    let mut result = vec![0.];
    for secondary in &system.secondaries {
        result.push(
            secondary.body.mass / (system.primary.mass + secondary.body.mass)
                * secondary.orbit.state(time)?.1[2]
                * crate::orbit::SOLAR_RADIUS_METERS
                / crate::orbit::DAY_SECONDS,
        );
    }
    Ok(result)
}
