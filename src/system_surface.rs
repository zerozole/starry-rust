//! Surface-parameter derivatives with arbitrary overlapping silhouettes.
use crate::{
    Map, Result,
    conic::Ellipse,
    differentiation,
    imaging::{self, View},
    oblate_derivatives, reflected_derivatives, rv_derivatives,
};

/// Layout: inc, obl, amp, roughness, f, omega, beta, tpole, wav, y..., u....
pub fn flux(
    map: &Map,
    view: &View,
    ellipses: &[Ellipse],
    velocity: Option<(f64, f64)>,
) -> Result<(f64, Vec<f64>)> {
    let mut out = vec![0.; 9 + map.coefficients().len() + view.limb.len()];
    let filters = velocity
        .map(|(v, a)| rv_derivatives::filters(view.inclination, view.obliquity, v, a))
        .transpose()?;
    let evaluate = |m: &Map, v: &View, e: &[Ellipse]| -> Result<f64> {
        if let Some(f) = &filters {
            let p = m
                .projected(v.inclination, v.obliquity, v.phase)?
                .limb_filtered(&v.limb)?
                .multiplied(&f[0])?;
            imaging::flux(&p, &View::default(), e)
        } else {
            imaging::flux(m, v, e)
        }
    };
    let value = evaluate(map, view, ellipses)?;
    let mut unit = map.clone();
    unit.set_amplitude(1.)?;
    out[2] = evaluate(&unit, view, ellipses)?;
    if let Some((f, g, normalized)) = view.oblateness {
        let physical = oblate_derivatives::flux_ellipses(
            map,
            f,
            view.inclination,
            view.obliquity,
            view.phase,
            &view.limb,
            g,
            ellipses,
            normalized,
        )?;
        out[0] = physical[2];
        out[1] = physical[3];
        out[4..9].copy_from_slice(&physical[4..9]);
    } else {
        let projected = map.projected(view.inclination, view.obliquity, view.phase)?;
        let mut identity = view.clone();
        identity.inclination = std::f64::consts::FRAC_PI_2;
        identity.obliquity = 0.;
        identity.phase = 0.;
        let (s, c) = view.obliquity.sin_cos();
        for (i, axis) in [[-c, -s, 0.], [0., 0., 1.]].into_iter().enumerate() {
            out[i] = evaluate(
                &differentiation::rotation_derivative(&projected, axis)?,
                &identity,
                ellipses,
            )?;
            if let Some(f) = &filters {
                out[i] += imaging::flux(
                    &projected.limb_filtered(&view.limb)?.multiplied(&f[1 + i])?,
                    &View::default(),
                    ellipses,
                )?;
            }
        }
        if let Some(lamp) = view.reflection {
            let observed = projected.limb_filtered(&view.limb)?;
            out[3] = if lamp.radius == 0. {
                reflected_derivatives::point_ellipses(
                    &observed,
                    lamp.source,
                    lamp.roughness,
                    ellipses,
                )?[3]
            } else {
                reflected_derivatives::extended_ellipses(
                    &observed,
                    lamp.source,
                    lamp.radius,
                    lamp.samples,
                    lamp.roughness,
                    ellipses,
                )?[3]
            };
        }
    }
    let mut basis = Map::new(map.degree())?;
    basis.set_amplitude(map.amplitude())?;
    for j in 0..map.coefficients().len() {
        let mut y = vec![0.; map.coefficients().len()];
        y[j] = 1.;
        basis.set_coefficients(&y)?;
        out[9 + j] = evaluate(&basis, view, ellipses)?;
    }
    let mut raw_view = view.clone();
    let normalized = view.oblateness.is_some_and(|(_, _, n)| n);
    if let Some((f, g, _)) = view.oblateness {
        raw_view.oblateness = Some((f, g, false));
    }
    let raw = evaluate(map, &raw_view, ellipses)?;
    let mut unfiltered = raw_view.clone();
    unfiltered.limb.clear();
    let unfiltered_flux = evaluate(map, &unfiltered, ellipses)?;
    let uniform = Map::new(0)?;
    let baseline = if normalized {
        evaluate(&uniform, &raw_view, &[])?
    } else {
        1.
    };
    let unfiltered_base = if normalized {
        evaluate(&uniform, &unfiltered, &[])?
    } else {
        0.
    };
    let (_, norm) = crate::map::limb_polynomial(&view.limb)?;
    for i in 0..view.limb.len() {
        let mut one = raw_view.clone();
        one.limb = vec![0.; i + 1];
        one.limb[i] = 1.;
        let scale = 2. / ((i + 2) * (i + 3)) as f64;
        let direct =
            ((1. - scale) * evaluate(map, &one, ellipses)? - unfiltered_flux + scale * raw) / norm;
        let db = if normalized {
            ((1. - scale) * evaluate(&uniform, &one, &[])? - unfiltered_base + scale * baseline)
                / norm
        } else {
            0.
        };
        out[9 + map.coefficients().len() + i] = (direct - raw * db / baseline) / baseline;
    }
    crate::require(
        out.iter().all(|v| v.is_finite()),
        "nonfinite system surface Jacobian",
    )?;
    Ok((value, out))
}
