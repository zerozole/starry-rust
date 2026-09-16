//! Bulirsch CEL, translated from upstream ellip.h (MIT).
use crate::{Error, Result, require};
use std::f64::consts::PI;

pub fn cel(mut ksq: f64, mut kc: f64, mut p: f64, mut a: f64, mut b: f64) -> Result<f64> {
    require(
        [ksq, kc, p, a, b].iter().all(|v| v.is_finite()) && ksq <= 1.,
        "invalid CEL argument",
    )?;
    ksq = ksq.max(0.);
    kc = kc.max(0.);
    if ksq < 1e-5 {
        kc = (1. - ksq).sqrt();
    }
    if ksq == 1. || kc == 0. {
        kc = f64::EPSILON * ksq;
    }
    let ca = (f64::EPSILON * ksq).sqrt().max(f64::MIN_POSITIVE);
    let mut m = 1.;
    let mut ee = kc;
    if p > 0. {
        p = p.sqrt();
        b /= p;
    } else {
        let g = 1. - p;
        let f = g - ksq;
        let q = ksq * (b - a * p);
        p = (f / g).sqrt();
        a = (a - b) / g;
        b = -q / (g * g * p) + a * p;
    }
    let f = a;
    a += b / p;
    let mut g = ee / p;
    b = 2. * (b + f * g);
    p += g;
    m += kc;
    for _ in 0..200 {
        kc = 2. * ee.sqrt();
        ee = kc * m;
        let f = a;
        a += b / p;
        g = ee / p;
        b = 2. * (b + f * g);
        p += g;
        g = m;
        m += kc;
        if (g - kc).abs() <= g * ca {
            let v = 0.5 * PI * (a * m + b) / (m * (m + p));
            require(v.is_finite(), "singular CEL result")?;
            return Ok(v);
        }
    }
    Err(Error("CEL did not converge".into()))
}
