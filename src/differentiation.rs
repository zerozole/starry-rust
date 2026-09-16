//! Analytic first derivatives of emitted-light maps, including all view,
//! coefficient, amplitude, limb-profile and occultor parameters.
use crate::{Map, Result, basis, matrix::Matrix, occultation::Occultor, require};
use std::collections::BTreeMap;

/// Infinitesimal active rotation: dI/dtheta = -(axis cross p) dot grad I.
pub fn rotation_derivative(map: &Map, axis: [f64; 3]) -> Result<Map> {
    require(
        axis.iter().all(|v| v.is_finite()),
        "invalid rotation derivative axis",
    )?;
    if map.degree() > 12 {
        let y = crate::rotation::project_function(map.degree(), |p| {
            let tangent = [
                axis[2] * p[1] - axis[1] * p[2],
                axis[0] * p[2] - axis[2] * p[0],
                axis[1] * p[0] - axis[0] * p[1],
            ];
            Ok(crate::map::dot(
                &basis::harmonic_directional(map.degree(), p, tangent)?,
                map.coefficients(),
            ))
        })?;
        let mut result = Map::new(map.degree())?;
        result.set_coefficients(&y)?;
        result.set_amplitude(map.amplitude())?;
        return Ok(result);
    }
    let terms = basis::terms(map.degree());
    let lookup: BTreeMap<_, _> = terms.iter().enumerate().map(|(n, &t)| (t, n)).collect();
    let p = map.polynomial_coefficients()?;
    let mut derivative = vec![0.; p.len()];
    // Six terms of the negative cross-product vector, in Cartesian components.
    let products = [
        (0, 2, -axis[1]),
        (0, 1, axis[2]),
        (1, 0, -axis[2]),
        (1, 2, axis[0]),
        (2, 1, -axis[0]),
        (2, 0, axis[1]),
    ];
    for (n, &(i, j, k)) in terms.iter().enumerate() {
        for &(d, m, a) in &products {
            let mut powers = [i, j, k];
            let exponent = powers[d];
            if exponent == 0 || a == 0. {
                continue;
            }
            powers[d] -= 1;
            powers[m] += 1;
            let coefficient = p[n] * a * exponent as f64;
            if powers[2] < 2 {
                derivative[lookup[&(powers[0], powers[1], powers[2])]] += coefficient;
            } else {
                powers[2] -= 2;
                derivative[lookup[&(powers[0], powers[1], powers[2])]] += coefficient;
                derivative[lookup[&(powers[0] + 2, powers[1], powers[2])]] -= coefficient;
                derivative[lookup[&(powers[0], powers[1] + 2, powers[2])]] -= coefficient;
            }
        }
    }
    let mut result = Map::new(map.degree())?;
    result.set_coefficients(
        &basis::a1(map.degree())?
            .solve(&Matrix {
                rows: p.len(),
                cols: 1,
                data: derivative,
            })?
            .data,
    )?;
    result.set_amplitude(map.amplitude())?;
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct FluxDerivatives {
    pub value: f64,
    pub phase: f64,
    pub inclination: f64,
    pub obliquity: f64,
    pub occultor: [f64; 3],
    pub amplitude: f64,
    pub coefficients: Vec<f64>,
    pub limb: Vec<f64>,
}
pub fn emitted(
    map: &Map,
    inclination: f64,
    obliquity: f64,
    phase: f64,
    u: &[f64],
    occ: Option<Occultor>,
) -> Result<FluxDerivatives> {
    let view = map.projected(inclination, obliquity, phase)?;
    let value = view.flux_limb_darkened(u, occ)?;
    let (s, c) = obliquity.sin_cos();
    let (si, ci) = inclination.sin_cos();
    let dphase = rotation_derivative(&view, [-s * si, c * si, ci])?.flux_limb_darkened(u, occ)?;
    let dinc = rotation_derivative(&view, [-c, -s, 0.])?.flux_limb_darkened(u, occ)?;
    let dobl = rotation_derivative(&view, [0., 0., 1.])?.flux_limb_darkened(u, occ)?;
    let geometry = if let Some(o) = occ {
        view.limb_filtered(u)?.flux_gradient(o)?
    } else {
        [0.; 3]
    };
    let mut unit = view.clone();
    unit.set_amplitude(1.)?;
    let amplitude = unit.flux_limb_darkened(u, occ)?;
    // All harmonic degrees have the same norm, so the adjoint response is
    // obtained by applying the inverse map rotation to the observer response.
    let mut response = Map::new(map.degree())?;
    response.set_coefficients(&crate::rings::flux_design(
        map.degree(),
        occ,
        u,
        Default::default(),
    )?)?;
    response.rotate([0., 0., 1.], -obliquity)?;
    response.rotate([1., 0., 0.], inclination - std::f64::consts::PI / 2.)?;
    response.rotate([0., 1., 0.], -phase)?;
    let coefficients = response
        .coefficients()
        .iter()
        .map(|v| v * map.amplitude())
        .collect();
    let (_, normalization) = crate::map::limb_polynomial(u)?;
    let unfiltered = view.flux(occ)?;
    let mut limb = vec![];
    for i in 0..u.len() {
        let degree = i + 1;
        let mut one = vec![0.; degree];
        one[i] = 1.;
        let scale = 2. / ((i + 2) * (i + 3)) as f64;
        let direct = (1. - scale) * view.flux_limb_darkened(&one, occ)? - unfiltered;
        limb.push((direct + value * scale) / normalization);
    }
    Ok(FluxDerivatives {
        value,
        phase: dphase,
        inclination: dinc,
        obliquity: dobl,
        occultor: geometry,
        amplitude,
        coefficients,
        limb,
    })
}
