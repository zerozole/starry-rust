//! Assemble full first map Jacobians from native field and shape derivatives.
use crate::{
    Map, Result, basis, differentiation::rotation_derivative, imaging, map, oblate,
    oblate_derivatives, occultation::Occultor, reflected, reflected_derivatives,
};

/// Layout: flux, phase/inc/obl, occultor x/y/r, amplitude, y..., u...,
/// then source x/y/z, roughness, source radius.
pub fn reflection(map: &Map, view: &imaging::View, occ: Option<Occultor>) -> Result<Vec<f64>> {
    let lamp = view
        .reflection
        .ok_or(crate::Error("missing illumination".into()))?;
    let occultors: Vec<_> = occ.into_iter().collect();
    let evaluate = |m: &Map, u: &[f64]| {
        reflected::flux_extended(
            &m.limb_filtered(u)?,
            lamp.source,
            lamp.radius,
            lamp.samples,
            lamp.roughness,
            &occultors,
        )
    };
    let projected = map.projected(view.inclination, view.obliquity, view.phase)?;
    let value = evaluate(&projected, &view.limb)?;
    let (s, c) = view.obliquity.sin_cos();
    let (si, ci) = view.inclination.sin_cos();
    let orientation = [[-s * si, c * si, ci], [-c, -s, 0.], [0., 0., 1.]]
        .iter()
        .map(|&axis| evaluate(&rotation_derivative(&projected, axis)?, &view.limb))
        .collect::<Result<Vec<_>>>()?;
    let shape = if let Some(o) = occ {
        imaging::occultor_gradient(map, view, o)?
    } else {
        [0.; 3]
    };
    let mut unit = projected.clone();
    unit.set_amplitude(1.)?;
    let mut out = vec![value];
    out.extend(orientation);
    out.extend(shape);
    out.push(evaluate(&unit, &view.limb)?);
    let mut coefficient_map = Map::new(map.degree())?;
    coefficient_map.set_amplitude(map.amplitude())?;
    for j in 0..map.coefficients().len() {
        let mut y = vec![0.; map.coefficients().len()];
        y[j] = 1.;
        coefficient_map.set_coefficients(&y)?;
        out.push(evaluate(
            &coefficient_map.projected(view.inclination, view.obliquity, view.phase)?,
            &view.limb,
        )?);
    }
    let unfiltered = evaluate(&projected, &[])?;
    let (_, norm) = map::limb_polynomial(&view.limb)?;
    for i in 0..view.limb.len() {
        let mut one = vec![0.; i + 1];
        one[i] = 1.;
        let scale = 2. / ((i + 2) * (i + 3)) as f64;
        out.push(((1. - scale) * evaluate(&projected, &one)? - unfiltered + value * scale) / norm);
    }
    out.extend(reflected_derivatives::extended(
        &projected.limb_filtered(&view.limb)?,
        lamp.source,
        lamp.radius,
        lamp.samples,
        lamp.roughness,
        &occultors,
    )?);
    Ok(out)
}

/// Same common layout, then flattening, omega, beta, polar temperature, wavelength.
pub fn oblate(map: &Map, view: &imaging::View, occ: Option<Occultor>) -> Result<Vec<f64>> {
    let (f, gravity, normalized) = view
        .oblateness
        .ok_or(crate::Error("missing oblateness".into()))?;
    let evaluate = |m: &Map, u: &[f64], occ, normalized| {
        oblate::flux(
            m,
            f,
            view.inclination,
            view.obliquity,
            view.phase,
            u,
            gravity,
            occ,
            normalized,
        )
    };
    let physical = oblate_derivatives::flux(
        map,
        f,
        view.inclination,
        view.obliquity,
        view.phase,
        &view.limb,
        gravity,
        occ,
        normalized,
    )?;
    let shape = if let Some(o) = occ {
        imaging::occultor_gradient(map, view, o)?
    } else {
        [0.; 3]
    };
    let mut out = physical[..4].to_vec();
    out.extend(shape);
    out.push(physical[9]);
    let mut coefficient_map = Map::new(map.degree())?;
    coefficient_map.set_amplitude(map.amplitude())?;
    for j in 0..map.coefficients().len() {
        let mut y = vec![0.; map.coefficients().len()];
        y[j] = 1.;
        coefficient_map.set_coefficients(&y)?;
        out.push(evaluate(&coefficient_map, &view.limb, occ, normalized)?);
    }
    let (_, norm) = map::limb_polynomial(&view.limb)?;
    let raw = evaluate(map, &view.limb, occ, false)?;
    let unfiltered = evaluate(map, &[], occ, false)?;
    let uniform = Map::new(0)?;
    let baseline = if normalized {
        evaluate(&uniform, &view.limb, None, false)?
    } else {
        1.
    };
    let base_unfiltered = if normalized {
        evaluate(&uniform, &[], None, false)?
    } else {
        0.
    };
    for i in 0..view.limb.len() {
        let mut one = vec![0.; i + 1];
        one[i] = 1.;
        let scale = 2. / ((i + 2) * (i + 3)) as f64;
        let derivative =
            ((1. - scale) * evaluate(map, &one, occ, false)? - unfiltered + raw * scale) / norm;
        let base_derivative = if normalized {
            ((1. - scale) * evaluate(&uniform, &one, None, false)? - base_unfiltered
                + baseline * scale)
                / norm
        } else {
            0.
        };
        out.push((derivative - raw * base_derivative / baseline) / baseline);
    }
    out.extend_from_slice(&physical[4..9]);
    crate::require(
        out.len() == 13 + (map.degree() + 1).pow(2) + view.limb.len()
            && map.degree() <= basis::MAX_DEGREE,
        "invalid oblate derivative dimensions",
    )?;
    Ok(out)
}
