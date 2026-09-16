//! Chain orbital state derivatives through light delay, shape integrals,
//! reflection, surface rotation, overlapping silhouettes and exposures.
use crate::{
    Result, conic::Ellipse, differentiation, imaging, matrix::Matrix, orbit::C_SOLAR_RADII_PER_DAY,
    reflected_derivatives, system::System,
};
use std::f64::consts::PI;

/// Columns: observation time, exposure duration, seven independent parameters
/// for each orbit (a,period,e,inc,omega,node,transit_epoch), then three per body
/// (phase0,rotation_period,reference_epoch), then (mass,radius) and (veq,alpha)
/// per body. Surface mode appends nine physical map parameters, y and u per body.
/// Native orbital elements are independent of mass; the Python API chains Kepler's law.
pub struct Jacobian {
    pub values: Vec<f64>,
    pub derivatives: Matrix,
}

// mode: 0 flux, 1 rotational velocity moment, 2 Keplerian velocity.
fn surface_offsets(system: &System) -> Vec<usize> {
    let bodies = system.bodies();
    let n = bodies.len();
    let mut offset = 2 + 7 * (n - 1) + 7 * n;
    let mut result = vec![];
    for b in bodies {
        result.push(offset);
        offset += 9 + b.map.coefficients().len() + b.limb_darkening.len();
    }
    result.push(offset);
    result
}

fn instantaneous(system: &System, time: f64, mode: u8, surface: bool) -> Result<Jacobian> {
    let bodies = system.bodies();
    let nb = bodies.len();
    let physical = 2 + 7 * system.secondaries.len() + 3 * nb;
    let offsets = surface_offsets(system);
    let count = if surface {
        offsets[nb]
    } else {
        physical + 4 * nb
    };
    let (positions, times) = system.apparent(time)?;
    let mut values = if mode == 0 {
        system.flux_instantaneous(time)?
    } else {
        vec![0.; nb]
    };
    let mut derivatives = Matrix::zeros(nb, count);
    let mass: f64 = bodies.iter().map(|b| b.mass).sum();
    let mut position_derivatives = vec![Matrix::zeros(3, count); nb];
    let mut time_derivatives = vec![vec![0.; count]; nb];
    if system.source_convention && mode == 2 {
        let conversion = crate::orbit::SOLAR_RADIUS_METERS / crate::orbit::DAY_SECONDS;
        for (j, s) in system.secondaries.iter().enumerate() {
            let i = j + 1;
            let total = system.primary.mass + s.body.mass;
            let fraction = s.body.mass / total;
            let v = s.orbit.state_derivatives(time)?[5];
            values[i] = fraction * v.value * conversion;
            derivatives[(i, 0)] = fraction * v.gradient[7] * conversion;
            for k in 0..7 {
                derivatives[(i, 2 + 7 * j + k)] = fraction * v.gradient[k] * conversion;
            }
            derivatives[(i, physical)] = -s.body.mass / total.powi(2) * v.value * conversion;
            derivatives[(i, physical + 2 * i)] =
                system.primary.mass / total.powi(2) * v.value * conversion;
        }
        return Ok(Jacobian {
            values,
            derivatives,
        });
    }
    for i in 0..nb {
        let states = system.states(times[i])?;
        let velocity = states[i].1;
        let mut velocity_derivatives = vec![0.; count];
        let mut acceleration_z = 0.;
        for (j, secondary) in system.secondaries.iter().enumerate() {
            let jets = secondary.orbit.state_derivatives(times[i])?;
            let factor = if i == j + 1 { 1. } else { 0. }
                - if mass > 0. {
                    secondary.body.mass / mass
                } else {
                    0.
                };
            acceleration_z += factor * jets[5].gradient[7];
            for k in 0..7 {
                velocity_derivatives[2 + 7 * j + k] = factor * jets[5].gradient[k];
            }
            if mass > 0. {
                for k in 0..nb {
                    let dm = if k == j + 1 { 1. } else { 0. };
                    velocity_derivatives[physical + 2 * k] +=
                        (secondary.body.mass / mass.powi(2) - dm / mass) * jets[5].value;
                }
            }
            for axis in 0..3 {
                for k in 0..7 {
                    position_derivatives[i][(axis, 2 + 7 * j + k)] =
                        factor * jets[axis].gradient[k];
                }
                if mass > 0. {
                    for k in 0..nb {
                        let dm = if k == j + 1 { 1. } else { 0. };
                        position_derivatives[i][(axis, physical + 2 * k)] +=
                            (secondary.body.mass / mass.powi(2) - dm / mass) * jets[axis].value;
                    }
                }
            }
        }
        for k in 0..count {
            let observed = if k == 0 { 1. } else { 0. };
            let dt = if system.light_delay {
                (observed + position_derivatives[i][(2, k)] / C_SOLAR_RADII_PER_DAY)
                    / (1. - velocity[2] / C_SOLAR_RADII_PER_DAY)
            } else {
                observed
            };
            time_derivatives[i][k] = dt;
            derivatives[(i, k)] = -(velocity_derivatives[k] + acceleration_z * dt)
                * crate::orbit::SOLAR_RADIUS_METERS
                / crate::orbit::DAY_SECONDS;
            for (axis, &v) in velocity.iter().enumerate() {
                position_derivatives[i][(axis, k)] += v * dt;
            }
        }
        if mode == 2 {
            values[i] =
                -velocity[2] * crate::orbit::SOLAR_RADIUS_METERS / crate::orbit::DAY_SECONDS;
        }
    }
    if mode == 2 {
        return Ok(Jacobian {
            values,
            derivatives,
        });
    }
    if system.source_convention {
        position_derivatives = vec![Matrix::zeros(3, count); nb];
        time_derivatives = vec![vec![0.; count]; nb];
        for row in &mut time_derivatives {
            row[0] = 1.;
        }
        for (j, s) in system.secondaries.iter().enumerate() {
            let position = crate::source_orbits::position(&s.orbit, time, 1., system.light_delay)?;
            for axis in 0..3 {
                position_derivatives[j + 1][(axis, 0)] = position[axis].gradient[7];
                for k in 0..7 {
                    position_derivatives[j + 1][(axis, 2 + 7 * j + k)] = position[axis].gradient[k];
                }
            }
        }
    }
    derivatives = Matrix::zeros(nb, count);
    for (i, body) in bodies.iter().enumerate() {
        if mode == 1 && (!body.radial_velocity || body.radius == 0.) {
            continue;
        }
        let mut silhouettes = vec![];
        let mut foreground = vec![];
        let mut blocked = false;
        for (j, other) in bodies.iter().enumerate() {
            if i == j || positions[j][2] <= positions[i][2] || other.radius == 0. {
                continue;
            }
            let x = positions[j][0] - positions[i][0];
            let y = positions[j][1] - positions[i][1];
            if body.radius == 0. {
                blocked |= Ellipse {
                    x,
                    y,
                    major: other.radius,
                    minor: other.radius * other.axis_ratio()?,
                    angle: other.obliquity,
                }
                .contains(0., 0.);
            } else if x.hypot(y) < body.radius + other.radius {
                silhouettes.push(Ellipse {
                    x: x / body.radius,
                    y: y / body.radius,
                    major: other.radius / body.radius,
                    minor: other.radius * other.axis_ratio()? / body.radius,
                    angle: other.obliquity,
                });
                foreground.push(j);
            }
        }
        if blocked || (body.reflected && body.radius == 0.) {
            continue;
        }
        let mut view = imaging::View {
            inclination: body.inclination,
            obliquity: body.obliquity,
            phase: body.phase(times[i]),
            limb: body.limb_darkening.clone(),
            ..Default::default()
        };
        if let Some(f) = body.flattening {
            view.oblateness = Some((f, body.gravity, body.normalized));
        }
        if body.reflected {
            view.reflection = Some(imaging::Reflection {
                source: std::array::from_fn(|k| (positions[0][k] - positions[i][k]) / body.radius),
                radius: system.primary.radius / body.radius,
                samples: body.source_samples,
                roughness: body.roughness,
                illuminate: true,
                exact: false,
            });
        }
        let phase_map = differentiation::rotation_derivative(&body.map, [0., 1., 0.])?;
        let (shape, phase_derivative) = if mode == 1 {
            crate::require(!body.reflected, "reflected RV body is unsupported")?;
            let observed = body
                .map
                .projected(body.inclination, body.obliquity, view.phase)?
                .limb_filtered(&view.limb)?;
            let filter = crate::rv_derivatives::filters(
                body.inclination,
                body.obliquity,
                body.equatorial_velocity,
                body.differential_rotation,
            )?;
            let moment = observed.multiplied(&filter[0])?;
            let identity = imaging::View::default();
            values[i] = imaging::flux(&moment, &identity, &silhouettes)?;
            for j in 0..2 {
                derivatives[(i, physical + 2 * nb + 2 * i + j)] = imaging::flux(
                    &observed.multiplied(&filter[3 + j])?,
                    &identity,
                    &silhouettes,
                )?;
            }
            (
                imaging::silhouette_gradients(&moment, &identity, &silhouettes)?,
                imaging::flux(
                    &phase_map
                        .projected(body.inclination, body.obliquity, view.phase)?
                        .limb_filtered(&view.limb)?
                        .multiplied(&filter[0])?,
                    &identity,
                    &silhouettes,
                )?,
            )
        } else {
            (
                imaging::silhouette_gradients(&body.map, &view, &silhouettes)?,
                imaging::flux(&phase_map, &view, &silhouettes)?,
            )
        };
        if surface {
            let velocity = if mode == 1 {
                Some((body.equatorial_velocity, body.differential_rotation))
            } else {
                None
            };
            let (unscaled, gradient) =
                crate::system_surface::flux(&body.map, &view, &silhouettes, velocity)?;
            let lamp_scale = if body.reflected {
                system.primary.map.amplitude()
            } else {
                1.
            };
            for (j, g) in gradient.into_iter().enumerate() {
                derivatives[(i, offsets[i] + j)] += lamp_scale * g;
            }
            if body.reflected {
                derivatives[(i, offsets[0] + 2)] += unscaled;
            }
            for (&j, g) in foreground.iter().zip(&shape) {
                if let Some(f) = bodies[j].flattening {
                    let inc = bodies[j].inclination;
                    let q = bodies[j].axis_ratio()?;
                    let scale = lamp_scale * bodies[j].radius / body.radius;
                    derivatives[(i, offsets[j])] -=
                        g[3] * scale * f * (2. - f) * inc.sin() * inc.cos() / q;
                    derivatives[(i, offsets[j] + 4)] -=
                        g[3] * scale * (1. - f) * inc.sin().powi(2) / q;
                    derivatives[(i, offsets[j] + 1)] += lamp_scale * g[4];
                }
            }
        }
        let rate = if body.rotation_period == 0. {
            0.
        } else {
            2. * PI / body.rotation_period
        };
        let mut phase_chain: Vec<_> = time_derivatives[i].iter().map(|v| rate * v).collect();
        let spin = 2 + 7 * system.secondaries.len() + 3 * i;
        phase_chain[spin] = 1.;
        phase_chain[spin + 2] = -rate;
        phase_chain[spin + 1] = if body.rotation_period == 0. {
            0.
        } else {
            -rate * (times[i] - body.reference_epoch) / body.rotation_period
        };
        let mut source_gradient = [0.; 3];
        let mut source_radius_gradient = 0.;
        if let Some(lamp) = view.reflection {
            let map = body
                .map
                .projected(body.inclination, body.obliquity, view.phase)?
                .limb_filtered(&body.limb_darkening)?;
            if lamp.radius == 0. {
                source_gradient.copy_from_slice(
                    &reflected_derivatives::point_ellipses(
                        &map,
                        lamp.source,
                        lamp.roughness,
                        &silhouettes,
                    )?[..3],
                );
            } else {
                let gradient = reflected_derivatives::extended_ellipses(
                    &map,
                    lamp.source,
                    lamp.radius,
                    lamp.samples,
                    lamp.roughness,
                    &silhouettes,
                )?;
                source_gradient.copy_from_slice(&gradient[..3]);
                source_radius_gradient = gradient[4];
            }
        }
        for k in 0..count {
            let mut value = phase_derivative * phase_chain[k];
            for (j, g) in foreground.iter().zip(&shape) {
                value += (g[0]
                    * (position_derivatives[*j][(0, k)] - position_derivatives[i][(0, k)])
                    + g[1] * (position_derivatives[*j][(1, k)] - position_derivatives[i][(1, k)]))
                    / body.radius;
            }
            for ((&j, g), e) in foreground.iter().zip(&shape).zip(&silhouettes) {
                if k == physical + 2 * j + 1 {
                    value += (g[2] * e.major + g[3] * e.minor) / bodies[j].radius;
                }
                if k == physical + 2 * i + 1 {
                    value -=
                        (g[0] * e.x + g[1] * e.y + g[2] * e.major + g[3] * e.minor) / body.radius;
                }
            }
            if body.reflected {
                for axis in 0..3 {
                    value += source_gradient[axis]
                        * (position_derivatives[0][(axis, k)] - position_derivatives[i][(axis, k)])
                        / body.radius;
                }
                let lamp = view.reflection.expect("reflected body has a lamp");
                if k == physical + 1 {
                    value += source_radius_gradient / body.radius;
                }
                if k == physical + 2 * i + 1 {
                    value -= (crate::map::dot(&source_gradient, &lamp.source)
                        + source_radius_gradient * lamp.radius)
                        / body.radius;
                }
                value *= system.primary.map.amplitude();
            }
            derivatives[(i, k)] += value;
        }
    }
    crate::require(
        derivatives.data.iter().all(|v| v.is_finite()),
        "nonfinite system Jacobian",
    )?;
    Ok(Jacobian {
        values,
        derivatives,
    })
}

pub fn flux(system: &System, time: f64) -> Result<Jacobian> {
    integrate(system, time, 0, false)
}

pub fn flux_surface(system: &System, time: f64) -> Result<Jacobian> {
    integrate(system, time, 0, true)
}

fn integrate(system: &System, time: f64, mode: u8, surface: bool) -> Result<Jacobian> {
    let stencil = system.exposure_stencil()?;
    let nb = system.secondaries.len() + 1;
    let count = if surface {
        surface_offsets(system)[nb]
    } else {
        2 + 7 * (nb - 1) + 7 * nb
    };
    let mut result = Jacobian {
        values: vec![0.; nb],
        derivatives: Matrix::zeros(nb, count),
    };
    for (dt, weight) in stencil {
        let sample = instantaneous(system, time + dt, mode, surface)?;
        for i in 0..nb {
            result.values[i] += weight * sample.values[i];
            for k in 0..count {
                result.derivatives[(i, k)] += weight * sample.derivatives[(i, k)];
            }
            if system.exposure > 0. {
                result.derivatives[(i, 1)] +=
                    weight * sample.derivatives[(i, 0)] * dt / system.exposure;
            }
        }
    }
    Ok(result)
}

/// RV quotient averaged with flux weights; Keplerian term uses the central
/// apparent time exactly as System::radial_velocities does.
pub fn radial_velocity(system: &System, time: f64, keplerian: bool) -> Result<Jacobian> {
    radial_velocity_surface(system, time, keplerian, false)
}

pub fn radial_velocity_surface(
    system: &System,
    time: f64,
    keplerian: bool,
    surface: bool,
) -> Result<Jacobian> {
    let mut unit = system.clone();
    if system.exposure == 0. {
        unit.primary.map.set_amplitude(1.)?;
        for s in &mut unit.secondaries {
            s.body.map.set_amplitude(1.)?;
        }
    }
    let denominator = integrate(&unit, time, 0, surface)?;
    let mut numerator = integrate(&unit, time, 1, surface)?;
    for i in 0..numerator.values.len() {
        let d = denominator.values[i];
        let rv = if d == 0. { 0. } else { numerator.values[i] / d };
        for k in 0..numerator.derivatives.cols {
            numerator.derivatives[(i, k)] = if d == 0. {
                0.
            } else {
                (numerator.derivatives[(i, k)] - rv * denominator.derivatives[(i, k)]) / d
            };
        }
        numerator.values[i] = rv;
    }
    if keplerian {
        let orbital = instantaneous(system, time, 2, surface)?;
        for i in 0..numerator.values.len() {
            numerator.values[i] += orbital.values[i];
            for k in 0..numerator.derivatives.cols {
                numerator.derivatives[(i, k)] += orbital.derivatives[(i, k)];
            }
        }
    }
    Ok(numerator)
}
