//! Native oblate/gravity derivatives: differentiated profile fits, rotation
//! generators, and shape derivatives of the projected elliptical disk.
use crate::{
    Map, Result, autodiff::Jet, basis, differentiation, oblate::GravityDarkening,
    occultation::Occultor, quadrature, require, surface,
};
use std::f64::consts::PI;

fn gravity_profile(g: GravityDarkening, z: f64) -> Result<Jet<5>> {
    g.profile(z)?;
    let c = Jet::constant;
    let [omega, f, beta, tpole, wav] = std::array::from_fn(|i| {
        Jet::variable(
            [
                g.omega,
                g.flattening,
                g.beta,
                g.polar_temperature,
                g.wavelength_meters,
            ][i],
            i,
        )
    });
    let b = c(1.) - f;
    let z2 = c(z * z);
    let term = (c(1.) - omega.powi(2) * (z2 * b.powi(2) - z2 + c(1.)).pow(c(1.5))).powi(2);
    let ratio = (-z2 * b.powi(2) + (z2 - c(1.)) * term) / (-z2 * b.powi(2) + z2 - c(1.)).powi(3);
    // For beta>0 the Planck factor tends to zero faster than any power as
    // effective gravity tends to zero. Its first derivatives have zero limit.
    if ratio.value == 0. && beta.value > 0. {
        return Ok(c(0.));
    }
    require(
        ratio.value > 0.,
        "gravity derivative undefined at zero effective gravity",
    )?;
    let temperature = tpole * b.pow(c(2.) * beta) * ratio.pow(c(0.5) * beta);
    let exponent = c(1.43877735e-2) / (wav * temperature);
    if exponent.value > 700. {
        return Ok(c(0.));
    }
    Ok(exponent.exp_m1().reciprocal())
}

/// Value and five differentiated intrinsic gravity filters, ordered omega,
/// flattening, beta, polar temperature, wavelength. Same ridge/SHT as source.
fn filters(g: GravityDarkening) -> Result<(Map, Vec<Map>)> {
    let n = 4 * (g.degree + 1).pow(2);
    let z: Vec<_> = (0..n)
        .map(|i| -1. + 2. * i as f64 / (n - 1) as f64)
        .collect();
    let profiles = z
        .iter()
        .map(|&z| gravity_profile(g, z))
        .collect::<Result<Vec<_>>>()?;
    let mut derivatives = vec![];
    for k in 0..5 {
        let values: Vec<_> = profiles.iter().map(|p| p.gradient[k]).collect();
        let y = surface::axisymmetric_fit(g.degree, &z, &values, 1e-9, 0.)?;
        let mut m = Map::new(g.degree)?;
        m.set_coefficients(&y)?;
        m.rotate([1., 0., 0.], -PI / 2.)?;
        derivatives.push(m);
    }
    Ok((g.filter()?, derivatives))
}

// Unit-amplitude flux and derivatives: phase, inc, obl, f, omega, beta, T, wav.
#[allow(clippy::too_many_arguments)]
fn raw(
    map: &Map,
    f: f64,
    inc: f64,
    obl: f64,
    phase: f64,
    u: &[f64],
    gravity: Option<GravityDarkening>,
    occultors: &[crate::conic::Ellipse],
) -> Result<[f64; 9]> {
    require(
        f.is_finite() && (0. ..1.).contains(&f),
        "invalid flattening",
    )?;
    require(
        map.degree() + u.len() + gravity.map_or(0, |g| g.degree) <= basis::MAX_DEGREE,
        "combined oblate degree exceeds32",
    )?;
    let mut unit = map.clone();
    unit.set_amplitude(1.)?;
    let mut gravity_derivatives = vec![];
    let weighted = if let Some(mut g) = gravity {
        g.flattening = f;
        let (filter, derivatives) = filters(g)?;
        let scale = PI * 1.19104295e-16 / g.wavelength_meters.powi(5);
        for d in derivatives {
            let mut product = unit.multiplied(&d)?;
            product.set_amplitude(scale)?;
            gravity_derivatives.push(product);
        }
        let mut product = unit.multiplied(&filter)?;
        product.set_amplitude(scale)?;
        product
    } else {
        unit
    };
    let projected = weighted.projected(inc, 0., phase)?;
    let view = projected.limb_filtered(u)?;
    let q = (1. - f * (2. - f) * inc.sin().powi(2)).sqrt();
    let ellipses: Vec<_> = occultors.iter().map(|e| e.rotated(-obl)).collect();
    let integrate = |m: &Map| {
        let coefficient_scale = m.coefficients().iter().map(|v| v.abs()).fold(0., f64::max);
        if coefficient_scale == 0. || m.amplitude() == 0. {
            return Ok(0.);
        }
        let mut unit = m.clone();
        unit.set_coefficients(
            &m.coefficients()
                .iter()
                .map(|v| v / coefficient_scale)
                .collect::<Vec<_>>(),
        )?;
        unit.set_amplitude(1.)?;
        Ok(
            quadrature::surface(q, -1., &ellipses, 2e-12, |p| unit.intensity_xyz(p))?
                * coefficient_scale
                * m.amplitude(),
        )
    };
    let value = integrate(&view)?;
    let mut q_derivative = value / q;
    let mut obliquity_derivative = 0.;
    for (j, &e) in ellipses.iter().enumerate() {
        let others: Vec<_> = ellipses
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != j)
            .map(|(_, e)| *e)
            .collect();
        let shape = quadrature::ellipse_boundary(q, e, &others, |p| view.intensity_xyz(p))?;
        q_derivative -=
            quadrature::ellipse_boundary(q, e, &others, |p| Ok(view.intensity_xyz(p)? * p[1]))?[1];
        obliquity_derivative += shape[0] * e.y - shape[1] * e.x - shape[4];
    }
    // A uniform map times the axisymmetric gravity field is phase invariant.
    let dphase = if map.coefficients()[1..].iter().all(|v| *v == 0.) {
        0.
    } else {
        integrate(
            &differentiation::rotation_derivative(&projected, [0., inc.sin(), inc.cos()])?
                .limb_filtered(u)?,
        )?
    };
    let inc_map =
        differentiation::rotation_derivative(&projected, [-1., 0., 0.])?.limb_filtered(u)?;
    let mut result = [
        value,
        dphase,
        integrate(&inc_map)? - q_derivative * f * (2. - f) * inc.sin() * inc.cos() / q,
        obliquity_derivative,
        -q_derivative * (1. - f) * inc.sin().powi(2) / q,
        0.,
        0.,
        0.,
        0.,
    ];
    if let Some(g) = gravity {
        for (k, m) in gravity_derivatives.iter().enumerate() {
            let derivative = integrate(&m.projected(inc, 0., phase)?.limb_filtered(u)?)?;
            result[[5, 4, 6, 7, 8][k]] += derivative;
        }
        result[8] -= 5. * value / g.wavelength_meters;
    }
    Ok(result)
}

/// Returns flux, phase, inclination, obliquity, flattening, omega, beta,
/// polar temperature, wavelength and map-amplitude derivatives.
#[allow(clippy::too_many_arguments)]
pub fn flux(
    map: &Map,
    f: f64,
    inc: f64,
    obl: f64,
    phase: f64,
    u: &[f64],
    gravity: Option<GravityDarkening>,
    occ: Option<Occultor>,
    normalized: bool,
) -> Result<[f64; 10]> {
    flux_ellipses(
        map,
        f,
        inc,
        obl,
        phase,
        u,
        gravity,
        &quadrature::circles(&occ.into_iter().collect::<Vec<_>>())?,
        normalized,
    )
}

/// Oblate physical derivatives with a union of foreground ellipses.
#[allow(clippy::too_many_arguments)]
pub fn flux_ellipses(
    map: &Map,
    f: f64,
    inc: f64,
    obl: f64,
    phase: f64,
    u: &[f64],
    gravity: Option<GravityDarkening>,
    occultors: &[crate::conic::Ellipse],
    normalized: bool,
) -> Result<[f64; 10]> {
    require(
        gravity.is_none_or(|g| g.flattening == f),
        "gravity and silhouette flattening must agree",
    )?;
    let mut values = raw(map, f, inc, obl, phase, u, gravity, occultors)?;
    if normalized {
        let mut base = raw(&Map::new(0)?, f, inc, obl, 0., u, gravity, &[])?;
        base[1] = 0.;
        base[3] = 0.;
        require(base[0] > 0., "nonpositive oblate normalization")?;
        for k in 1..9 {
            values[k] = (values[k] - values[0] * base[k] / base[0]) / base[0];
        }
        values[0] /= base[0];
    }
    let mut out = [0.; 10];
    for i in 0..9 {
        out[i] = map.amplitude() * values[i];
    }
    out[9] = values[0];
    require(
        out.iter().all(|v| v.is_finite()),
        "nonfinite oblate derivative",
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_gravity_has_zero_planck_derivative_limit() {
        let g = crate::oblate::GravityDarkening {
            degree: 4,
            omega: 1.,
            flattening: 0.2,
            beta: 0.25,
            polar_temperature: 5000.,
            wavelength_meters: 600e-9,
        };
        let p = super::gravity_profile(g, 0.).unwrap();
        assert_eq!(p.value, 0.);
        assert_eq!(p.gradient, [0.; 5]);
    }
}
