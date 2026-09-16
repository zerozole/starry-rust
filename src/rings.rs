//! Stable circular-occultation integration directly in the harmonic basis.
//! Azimuthal integrals are analytic. Radial integration uses adaptive Gaussian
//! quadrature with endpoint transformations at circle tangencies.
use crate::{
    Error, Result, basis, map,
    occultation::{self, Integration, Occultor},
    require,
};
use std::f64::consts::PI;

struct Integral<'a> {
    degree: usize,
    b: f64,
    r: f64,
    phi: f64,
    occultors: Option<&'a [Occultor]>,
    limb: &'a [f64],
    norm: f64,
    nodes16: Vec<(f64, f64)>,
    nodes32: Vec<(f64, f64)>,
    max_depth: usize,
}
impl Integral<'_> {
    fn sample(&self, rho: f64) -> Result<Vec<f64>> {
        let mu = ((1. - rho) * (1. + rho)).max(0.).sqrt();
        let h = basis::harmonics(self.degree, [rho, 0., mu])?;
        if let Some(occultors) = self.occultors {
            let mut arcs = vec![];
            for o in occultors {
                let b = o.x.hypot(o.y);
                let r = o.radius;
                if b + rho <= r {
                    return Ok(vec![0.; h.len()]);
                }
                if b >= rho + r || rho >= b + r {
                    continue;
                }
                let area = ((r + b + rho) * (r + b - rho) * (r - b + rho) * (-r + b + rho))
                    .max(0.)
                    .sqrt();
                let half = area.atan2(rho * rho + b * b - r * r);
                let left = (o.y.atan2(o.x) - half).rem_euclid(2. * PI);
                let right = left + 2. * half;
                if right > 2. * PI {
                    arcs.push((left, 2. * PI));
                    arcs.push((0., right - 2. * PI));
                } else {
                    arcs.push((left, right));
                }
            }
            arcs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut merged: Vec<(f64, f64)> = vec![];
            for (a, b) in arcs {
                if let Some(last) = merged.last_mut().filter(|last| a <= last.1) {
                    last.1 = last.1.max(b);
                } else {
                    merged.push((a, b));
                }
            }
            let filter = map::limb_intensity(mu, self.limb)? / self.norm;
            let mut row = vec![0.; h.len()];
            for m in 0..=self.degree {
                let (cosine, sine) = if m == 0 {
                    (2. * PI - merged.iter().map(|(a, b)| b - a).sum::<f64>(), 0.)
                } else {
                    let m = m as f64;
                    (
                        -merged
                            .iter()
                            .map(|(a, b)| ((m * b).sin() - (m * a).sin()) / m)
                            .sum::<f64>(),
                        -merged
                            .iter()
                            .map(|(a, b)| ((m * a).cos() - (m * b).cos()) / m)
                            .sum::<f64>(),
                    )
                };
                for l in m..=self.degree {
                    let p = filter * h[basis::index(l, m as isize)];
                    row[basis::index(l, m as isize)] = p * cosine;
                    if m > 0 {
                        row[basis::index(l, -(m as isize))] = p * sine;
                    }
                }
            }
            return Ok(row);
        }
        let alpha = if self.b + rho <= self.r {
            PI
        } else if self.b >= rho + self.r || rho >= self.b + self.r {
            0.
        } else {
            let area = ((self.r + self.b + rho)
                * (self.r + self.b - rho)
                * (self.r - self.b + rho)
                * (-self.r + self.b + rho))
                .max(0.)
                .sqrt();
            area.atan2(rho * rho + self.b * self.b - self.r * self.r)
        };
        let mut row = vec![0.; h.len()];
        let filter = map::limb_intensity(mu, self.limb)? / self.norm;
        for m in 0..=self.degree {
            let angular = if m == 0 {
                2. * (PI - alpha)
            } else {
                -2. * (m as f64 * alpha).sin() / m as f64
            };
            let (sin, cos) = (m as f64 * self.phi).sin_cos();
            for l in m..=self.degree {
                let value = filter * h[basis::index(l, m as isize)] * angular;
                row[basis::index(l, m as isize)] = value * cos;
                if m > 0 {
                    row[basis::index(l, -(m as isize))] = value * sin;
                }
            }
        }
        Ok(row)
    }
    fn quadrature(
        &self,
        radial: [f64; 2],
        interval: [f64; 2],
        nodes: &[(f64, f64)],
    ) -> Result<Vec<f64>> {
        let mut result = vec![0.; (self.degree + 1).pow(2)];
        let half = (interval[1] - interval[0]) / 2.;
        let mid = (interval[1] + interval[0]) / 2.;
        for &(node, weight) in nodes {
            let t = mid + half * node;
            let rho = radial[0] + (radial[1] - radial[0]) * t.sin().powi(2);
            let scale = half * weight * rho * (radial[1] - radial[0]) * (2. * t).sin();
            for (value, sample) in result.iter_mut().zip(self.sample(rho)?) {
                *value += scale * sample;
            }
        }
        Ok(result)
    }
    fn adaptive(
        &self,
        radial: [f64; 2],
        interval: [f64; 2],
        tol: f64,
        depth: usize,
    ) -> Result<Vec<f64>> {
        let low = self.quadrature(radial, interval, &self.nodes16)?;
        let high = self.quadrature(radial, interval, &self.nodes32)?;
        let error = low
            .iter()
            .zip(&high)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if error <= tol {
            return Ok(high);
        }
        if depth >= self.max_depth {
            return Err(Error(format!(
                "harmonic ring quadrature did not converge: {error:e}"
            )));
        }
        let mid = (interval[0] + interval[1]) / 2.;
        let mut a = self.adaptive(radial, [interval[0], mid], tol / 2., depth + 1)?;
        let b = self.adaptive(radial, [mid, interval[1]], tol / 2., depth + 1)?;
        for (a, b) in a.iter_mut().zip(b) {
            *a += b;
        }
        Ok(a)
    }
}

/// Harmonic flux design, with optional normalized observer-frame limb profile.
/// Tolerance is an estimated absolute error per response coefficient.
pub fn flux_design(
    degree: usize,
    occ: Option<Occultor>,
    limb: &[f64],
    options: Integration,
) -> Result<Vec<f64>> {
    require(
        degree + limb.len() <= basis::MAX_DEGREE
            && options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid harmonic integration settings",
    )?;
    let (_, norm) = map::limb_polynomial(limb)?;
    let occ = match occ {
        Some(o) => Occultor::new(o.x, o.y, o.radius)?,
        None => Occultor::new(0., 0., 0.)?,
    };
    let b = occ.x.hypot(occ.y);
    let r = occ.radius;
    let mut result = vec![0.; (degree + 1).pow(2)];
    if r >= b + 1. {
        return Ok(result);
    }
    // Axially symmetric domains require only ordinary Legendre moments, and
    // Gaussian quadrature is exact up to the combined polynomial degree.
    if b == 0. || b >= 1. + r || r == 0. {
        let top = if b == 0. {
            ((1. - r) * (1. + r)).max(0.).sqrt()
        } else {
            1.
        };
        for (node, weight) in occultation::gauss((degree + limb.len() + 3).div_ceil(2)) {
            let mu = top * (node + 1.) / 2.;
            let h = basis::harmonics(degree, [(1. - mu * mu).sqrt(), 0., mu])?;
            let scale = PI * top * weight * mu * map::limb_intensity(mu, limb)? / norm;
            for l in 0..=degree {
                let index = basis::index(l, 0);
                result[index] += scale * h[index];
            }
        }
        return Ok(result);
    }
    let integral = Integral {
        degree,
        b,
        r,
        phi: occ.y.atan2(occ.x),
        occultors: None,
        limb,
        norm,
        nodes16: occultation::gauss(16),
        nodes32: occultation::gauss(32),
        max_depth: options.max_depth,
    };
    let mut cuts = vec![0., 1.];
    for cut in [(b - r).abs(), b + r] {
        if cut > 0. && cut < 1. {
            cuts.push(cut);
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    for cut in cuts.windows(2) {
        let row = integral.adaptive(
            [cut[0], cut[1]],
            [0., PI / 2.],
            options.absolute_tolerance / (cuts.len() - 1) as f64,
            0,
        )?;
        for (a, b) in result.iter_mut().zip(row) {
            *a += b;
        }
    }
    Ok(result)
}

/// Harmonic response to the union of foreground circles, without double counting.
pub fn flux_design_many(
    degree: usize,
    occultors: &[Occultor],
    limb: &[f64],
    options: Integration,
) -> Result<Vec<f64>> {
    if occultors.len() <= 1 {
        return flux_design(degree, occultors.first().copied(), limb, options);
    }
    require(
        occultors.len() <= 100
            && degree + limb.len() <= basis::MAX_DEGREE
            && options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid harmonic union options",
    )?;
    let (_, norm) = map::limb_polynomial(limb)?;
    let mut active = vec![];
    let mut cuts = vec![0., 1.];
    for o in occultors {
        Occultor::new(o.x, o.y, o.radius)?;
        let b = o.x.hypot(o.y);
        let r = o.radius;
        if r >= b + 1. {
            return Ok(vec![0.; (degree + 1).pow(2)]);
        }
        if r == 0. || b >= 1. + r {
            continue;
        }
        for cut in [(b - r).abs(), b + r] {
            if cut > 0. && cut < 1. {
                cuts.push(cut);
            }
        }
        active.push(*o);
    }
    if active.len() <= 1 {
        return flux_design(degree, active.first().copied(), limb, options);
    }
    for (i, a) in active.iter().enumerate() {
        for b in &active[i + 1..] {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let distance = dx.hypot(dy);
            if distance == 0.
                || distance >= a.radius + b.radius
                || distance <= (a.radius - b.radius).abs()
            {
                continue;
            }
            let along =
                (a.radius * a.radius - b.radius * b.radius + distance * distance) / (2. * distance);
            let height = (a.radius * a.radius - along * along).max(0.).sqrt();
            for sign in [-1., 1.] {
                let x = a.x + along * dx / distance - sign * height * dy / distance;
                let y = a.y + along * dy / distance + sign * height * dx / distance;
                let cut = x.hypot(y);
                if cut > 0. && cut < 1. {
                    cuts.push(cut);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let integral = Integral {
        degree,
        b: 0.,
        r: 0.,
        phi: 0.,
        occultors: Some(&active),
        limb,
        norm,
        nodes16: occultation::gauss(16),
        nodes32: occultation::gauss(32),
        max_depth: options.max_depth,
    };
    let mut result = vec![0.; (degree + 1).pow(2)];
    for c in cuts.windows(2) {
        let row = integral.adaptive(
            [c[0], c[1]],
            [0., PI / 2.],
            options.absolute_tolerance / (cuts.len() - 1) as f64,
            0,
        )?;
        for (a, b) in result.iter_mut().zip(row) {
            *a += b;
        }
    }
    Ok(result)
}
