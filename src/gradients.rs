//! Geometry gradients from the shape derivative of the occultor boundary.
//!
//! Differentiates the integration domain, not a sampled flux curve. The boundary
//! integral is evaluated by converged Gaussian quadrature. This is independent
//! of upstream's forward-mode AD of the analytic Green solver.
use crate::{
    Error, Result, basis,
    matrix::Matrix,
    occultation::{Occultor, gauss},
    require,
};
use std::f64::consts::PI;

/// Derivative of the flux design row with respect to (xo, yo, ro).
/// Error target is absolute per polynomial moment before the basis transform
/// through degree 12, and directly per harmonic above degree 12.
pub fn flux_design(
    degree: usize,
    a1: &Matrix,
    occ: Occultor,
    tolerance: f64,
) -> Result<[Vec<f64>; 3]> {
    Occultor::new(occ.x, occ.y, occ.radius)?;
    require(
        degree <= basis::MAX_DEGREE && tolerance.is_finite() && tolerance > 0.,
        "invalid gradient options",
    )?;
    let b = occ.x.hypot(occ.y);
    let r = occ.radius;
    let n = (degree + 1).pow(2);
    require(
        !(b == 0. && r == 1.),
        "radius gradient is undefined for coincident unit disks",
    )?;
    if r == 0. || b >= 1. + r || r >= 1. + b {
        return Ok(std::array::from_fn(|_| vec![0.; n]));
    }
    let half = if b + r <= 1. {
        PI
    } else {
        ((b * b + r * r - 1.) / (2. * b * r)).clamp(-1., 1.).acos()
    };
    let mid = occ.y.atan2(occ.x) + PI;
    let ts = basis::terms(degree);
    let eval = |order| -> Result<[Vec<f64>; 3]> {
        let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.; n]);
        for (node, weight) in gauss(order) {
            let t = node * PI / 2.;
            let theta = mid + half * t.sin();
            let w = weight * half * PI / 2. * t.cos() * r;
            let (sy, cx) = theta.sin_cos();
            let x = occ.x + r * cx;
            let y = occ.y + r * sy;
            let z = (1. - x * x - y * y).max(0.).sqrt();
            if degree > 12 {
                for (i, p) in basis::harmonics(degree, [x, y, z])?.iter().enumerate() {
                    out[0][i] -= w * p * cx;
                    out[1][i] -= w * p * sy;
                    out[2][i] -= w * p;
                }
                continue;
            }
            for (i, &(a, b, c)) in ts.iter().enumerate() {
                let p = x.powi(a as i32) * y.powi(b as i32) * z.powi(c as i32) * w;
                out[0][i] -= p * cx;
                out[1][i] -= p * sy;
                out[2][i] -= p;
            }
        }
        Ok(out)
    };
    let mut previous = eval(32)?;
    for order in [64, 128, 256, 512] {
        let current = eval(order)?;
        let error = previous
            .iter()
            .flatten()
            .zip(current.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if error <= tolerance {
            if degree > 12 {
                return Ok(current);
            }
            return Ok([
                a1.left_dot(&current[0])?,
                a1.left_dot(&current[1])?,
                a1.left_dot(&current[2])?,
            ]);
        }
        previous = current;
    }
    Err(Error(
        "boundary gradient quadrature did not converge".into(),
    ))
}
