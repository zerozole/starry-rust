//! Native derivatives of the upstream OpsRV flux ratio and velocity filter.
use crate::{
    Map, Result, autodiff::Jet, differentiation::rotation_derivative, imaging::View,
    occultation::Occultor, require,
};
use std::f64::consts::PI;

/// Velocity filter followed by its inc, obl, veq and alpha derivatives.
pub fn filters(inc: f64, obl: f64, veq: f64, alpha: f64) -> Result<Vec<Map>> {
    crate::rv::velocity_polynomial(inc, obl, veq, alpha)?;
    let [inc, obl, veq, alpha] =
        std::array::from_fn(|i| Jet::<4>::variable([inc, obl, veq, alpha][i], i));
    let k = Jet::constant;
    let a = inc.sin() * obl.cos();
    let b = inc.sin() * obl.sin();
    let c = inc.cos();
    let mut y = [k(0.); 16];
    y[1] = veq * k(3_f64.sqrt()) * b * (-(a * a + b * b + c * c) * alpha + k(5.)) / k(15.);
    y[3] = veq * k(3_f64.sqrt()) * a * (-(a * a + b * b + c * c) * alpha + k(5.)) / k(15.);
    y[9] = veq * alpha * k(70_f64.sqrt()) * b * (k(3.) * a * a - b * b) / k(70.);
    y[10] = veq * alpha * k(2. * 105_f64.sqrt()) * c * (-a * a + b * b) / k(105.);
    y[11] = veq * alpha * k(42_f64.sqrt()) * b * (a * a + b * b - k(4.) * c * c) / k(210.);
    y[13] = veq * alpha * k(42_f64.sqrt()) * a * (a * a + b * b - k(4.) * c * c) / k(210.);
    y[14] = veq * alpha * k(4. * 105_f64.sqrt()) * a * b * c / k(105.);
    y[15] = veq * alpha * k(70_f64.sqrt()) * a * (a * a - k(3.) * b * b) / k(70.);
    (0..5)
        .map(|i| {
            let mut m = Map::new(3)?;
            m.set_coefficients(
                &y.iter()
                    .map(|v| PI * if i == 0 { v.value } else { v.gradient[i - 1] })
                    .collect::<Vec<_>>(),
            )?;
            Ok(m)
        })
        .collect()
}

/// Layout: rv, phase, inc, obl, xo, yo, ro, amp, y..., u..., veq, alpha.
pub fn jacobian(
    map: &Map,
    view: &View,
    veq: f64,
    alpha: f64,
    occ: Option<Occultor>,
) -> Result<Vec<f64>> {
    require(
        map.degree() + view.limb.len() + 3 <= crate::basis::MAX_DEGREE,
        "combined RV degree exceeds 32",
    )?;
    // Amplitude cancels, including the forward API's zero-amplitude extension.
    let mut intrinsic = map.clone();
    intrinsic.set_amplitude(1.)?;
    let projected = intrinsic.projected(view.inclination, view.obliquity, view.phase)?;
    let field = projected.limb_filtered(&view.limb)?;
    let filters = filters(view.inclination, view.obliquity, veq, alpha)?;
    let numerator = field.multiplied(&filters[0])?;
    let denominator = field.flux(occ)?;
    let mut out = vec![0.; 10 + map.coefficients().len() + view.limb.len()];
    if denominator == 0. {
        return Ok(out);
    }
    let value = numerator.flux(occ)? / denominator;
    out[0] = value;
    let derivative = |delta: &Map| -> Result<f64> {
        Ok((delta.multiplied(&filters[0])?.flux(occ)? - value * delta.flux(occ)?) / denominator)
    };
    let (s, c) = view.obliquity.sin_cos();
    let (si, ci) = view.inclination.sin_cos();
    for (j, axis) in [[-s * si, c * si, ci], [-c, -s, 0.], [0., 0., 1.]]
        .into_iter()
        .enumerate()
    {
        out[1 + j] =
            derivative(&rotation_derivative(&projected, axis)?.limb_filtered(&view.limb)?)?;
        if j > 0 {
            out[1 + j] += field.multiplied(&filters[j])?.flux(occ)? / denominator;
        }
    }
    if let Some(o) = occ {
        let ng = numerator.flux_gradient(o)?;
        let dg = field.flux_gradient(o)?;
        for j in 0..3 {
            out[4 + j] = (ng[j] - value * dg[j]) / denominator;
        }
    }
    for j in 0..map.coefficients().len() {
        let mut basis = Map::new(map.degree())?;
        let mut y = vec![0.; map.coefficients().len()];
        y[j] = 1.;
        basis.set_coefficients(&y)?;
        out[8 + j] = derivative(
            &basis
                .projected(view.inclination, view.obliquity, view.phase)?
                .limb_filtered(&view.limb)?,
        )?;
    }
    let unfiltered = derivative(&projected)?;
    let (_, norm) = crate::map::limb_polynomial(&view.limb)?;
    for i in 0..view.limb.len() {
        let mut one = vec![0.; i + 1];
        one[i] = 1.;
        let scale = 2. / ((i + 2) * (i + 3)) as f64;
        out[8 + map.coefficients().len() + i] =
            ((1. - scale) * derivative(&projected.limb_filtered(&one)?)? - unfiltered) / norm;
    }
    let n = out.len();
    out[n - 2] = field.multiplied(&filters[3])?.flux(occ)? / denominator;
    out[n - 1] = field.multiplied(&filters[4])?.flux(occ)? / denominator;
    require(out.iter().all(|v| v.is_finite()), "nonfinite RV Jacobian")?;
    Ok(out)
}
