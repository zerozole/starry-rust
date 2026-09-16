//! Radial velocity maps. Harmonic velocity filter ported from core.py OpsRV.
use crate::{Map, Result, basis, occultation::Occultor, require};
use std::f64::consts::PI;
pub fn velocity_polynomial(inc: f64, obl: f64, veq: f64, alpha: f64) -> Result<Vec<f64>> {
    require(
        [inc, obl, veq, alpha].iter().all(|v| v.is_finite()) && (0. ..=PI).contains(&inc),
        "invalid RV arguments",
    )?;
    let a = inc.sin() * obl.cos();
    let b = inc.sin() * obl.sin();
    let c = inc.cos();
    let mut y = vec![0.; 16];
    y[1] = veq * 3_f64.sqrt() * b * (-(a * a + b * b + c * c) * alpha + 5.) / 15.;
    y[3] = veq * 3_f64.sqrt() * a * (-(a * a + b * b + c * c) * alpha + 5.) / 15.;
    y[9] = veq * alpha * 70_f64.sqrt() * b * (3. * a * a - b * b) / 70.;
    y[10] = veq * alpha * 2. * 105_f64.sqrt() * c * (-a * a + b * b) / 105.;
    y[11] = veq * alpha * 42_f64.sqrt() * b * (a * a + b * b - 4. * c * c) / 210.;
    y[13] = veq * alpha * 42_f64.sqrt() * a * (a * a + b * b - 4. * c * c) / 210.;
    y[14] = veq * alpha * 4. * 105_f64.sqrt() * a * b * c / 105.;
    y[15] = veq * alpha * 70_f64.sqrt() * a * (a * a - 3. * b * b) / 70.;
    for v in &mut y {
        *v *= PI;
    }
    basis::a1(3)?.dot(&y)
}

#[allow(clippy::too_many_arguments)] // Mirrors the upstream RV operation's arguments.
pub fn radial_velocity(
    map: &Map,
    inc: f64,
    obl: f64,
    phase: f64,
    veq: f64,
    alpha: f64,
    u: &[f64],
    occ: Option<Occultor>,
) -> Result<f64> {
    let map = map.projected(inc, obl, phase)?;
    if map.degree() + u.len() + 3 > 12 {
        return radial_velocity_many(
            &map,
            inc,
            obl,
            veq,
            alpha,
            u,
            &occ.into_iter().collect::<Vec<_>>(),
        );
    }
    require(
        map.degree() + u.len() + 3 <= basis::POLYNOMIAL_LIMIT,
        "combined RV degree exceeds 20",
    )?;
    let (limb, norm) = crate::map::limb_polynomial(u)?;
    let p = basis::multiply(
        &map.polynomial_coefficients()?,
        map.degree(),
        &limb,
        u.len(),
    );
    let d = map.degree() + u.len();
    let numerator = basis::multiply(&p, d, &velocity_polynomial(inc, obl, veq, alpha)?, 3);
    let denominator = crate::solver::polynomial_flux(d, &p, occ)? / norm;
    if denominator == 0. {
        return Ok(0.);
    }
    Ok(crate::solver::polynomial_flux(d + 3, &numerator, occ)? / (norm * denominator))
}

/// RV of an already observer-projected map with a union of foreground disks.
pub fn radial_velocity_many(
    projected: &Map,
    inc: f64,
    obl: f64,
    veq: f64,
    alpha: f64,
    u: &[f64],
    occultors: &[Occultor],
) -> Result<f64> {
    let map = projected.limb_filtered(u)?;
    let d = map.degree();
    require(d + 3 <= basis::MAX_DEGREE, "combined RV degree exceeds 32")?;
    if d + 3 > 12 {
        let velocity = velocity_polynomial(inc, obl, veq, alpha)?;
        // The scalar amplitude cancels in the velocity ratio, including at zero.
        let mut unit = map.clone();
        unit.set_amplitude(1.)?;
        let product = crate::rotation::project_function(d + 3, |p| {
            Ok(unit.intensity_xyz(p)? * crate::map::dot(&velocity, &basis::polynomial(3, p)?))
        })?;
        let row = crate::rings::flux_design_many(d + 3, occultors, &[], Default::default())?;
        let denominator = crate::map::dot(unit.coefficients(), &row[..unit.coefficients().len()]);
        if denominator == 0. {
            return Ok(0.);
        }
        return Ok(crate::map::dot(&product, &row) / denominator);
    }
    let p = map.polynomial_coefficients()?;
    let product = basis::multiply(&p, d, &velocity_polynomial(inc, obl, veq, alpha)?, 3);
    let moments = crate::occultation::union_moments(d + 3, occultors, Default::default())?.values;
    let denominator = crate::map::dot(&p, &moments[..p.len()]);
    if denominator == 0. {
        return Ok(0.);
    }
    Ok(crate::map::dot(&product, &moments) / denominator)
}
