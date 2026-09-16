//! Analytic emitted-light Green solver, ported from upstream solver.h.
//! Uses the same I/J recurrences and Bulirsch elliptic integrals.
//! f64 value path only; upstream geometry autodiff is not yet ported.
use crate::{
    Result, basis, elliptic,
    occultation::{Integration, Occultor},
    require,
};
use std::f64::consts::PI;

struct Geometry {
    b: f64,
    r: f64,
    k2: f64,
    k: f64,
    kc: f64,
    kc2: f64,
    kkc: f64,
    invk2: f64,
    area: f64,
    kap0: f64,
    kap1: f64,
    coslam: f64,
    sinlam: f64,
    qcond: bool,
}
impl Geometry {
    fn new(b: f64, r: f64) -> Self {
        let bmr = b - r;
        let bpr = b + r;
        let inv4br = 0.25 / (b * r);
        let oneplus = (1. + bpr) * (1. - bpr);
        let qcond = (1. - r).abs() >= b || bmr >= 1.;
        let (sinlam, coslam) = if qcond {
            (1., 0.)
        } else {
            let s = 0.5 * (1. / b + bmr * (1. + r / b));
            if s > 0.5 {
                let del = 1. - bpr;
                let eps = del / b * (r + 0.5 * del);
                (1. + eps, (-eps * (2. + eps)).max(0.).sqrt())
            } else {
                (s, (1. - s * s).max(0.).sqrt())
            }
        };
        let k2 = (oneplus * inv4br + 1.).max(0.);
        let k = k2.sqrt();
        let (kc2, kkc, area, kap0, kap1) = if k2 > 1. {
            let kc2 = oneplus / ((1. + bmr) * (1. - bmr));
            (kc2, k * kc2.sqrt(), 0., 0., 0.)
        } else {
            let mut sides = [1., b, r];
            sides.sort_by(|a, b| b.total_cmp(a));
            let [p0, p1, p2] = sides;
            let sq = (p0 + (p1 + p2)) * (p2 - (p0 - p1)) * (p2 + (p0 - p1)) * (p0 + (p1 - p2));
            let area = sq.max(0.).sqrt();
            (
                (-oneplus * inv4br).max(0.),
                area * inv4br,
                area,
                area.atan2((r - 1.) * (r + 1.) + b * b),
                area.atan2((1. - r) * (1. + r) + b * b),
            )
        };
        Self {
            b,
            r,
            k2,
            k,
            kc: kc2.sqrt(),
            kc2,
            kkc,
            invk2: 1. / k2,
            area,
            kap0,
            kap1,
            coslam,
            sinlam,
            qcond,
        }
    }
    fn s0(&self) -> f64 {
        if self.k2 > 1. {
            PI * (1. - self.r * self.r)
        } else {
            PI - self.kap1 - self.r * self.r * self.kap0 + self.area * 0.5
        }
    }
    // Returns the linear limb term and E, (E-(1-m)K)/m.
    fn s2(&self) -> Result<(f64, f64, f64)> {
        let Self {
            b,
            r,
            k2,
            kc,
            kc2,
            invk2,
            ..
        } = *self;
        let r2 = r * r;
        let bmr = b - r;
        let bpr = b + r;
        let one = (1. + bmr) * (1. - bmr);
        let lambda;
        let e;
        let ek;
        if b == r {
            if r == 0.5 {
                lambda = PI - 4. / 3.;
                e = 1.;
                ek = 1.;
            } else if r < 0.5 {
                let m = 4. * r2;
                e = elliptic::cel(m, (1. - m).sqrt(), 1., 1., 1. - m)?;
                ek = elliptic::cel(m, (1. - m).sqrt(), 1., 1., 0.)?;
                lambda = PI + 2. / 3. * ((2. * m - 3.) * e - m * ek);
            } else {
                let m = 4. * r2;
                let mi = 1. / m;
                e = elliptic::cel(mi, (1. - mi).sqrt(), 1., 1., 1. - mi)?;
                ek = elliptic::cel(mi, (1. - mi).sqrt(), 1., 1., 0.)?;
                lambda = PI + (-m * e + (2. * m - 3.) * ek) / (3. * r);
            }
        } else if k2 < 1. {
            let p = elliptic::cel(k2, kc, bmr * bmr * kc2, 0., 3. * kc2 * bmr * bpr)?;
            e = elliptic::cel(k2, kc, 1., 1., kc2)?;
            ek = elliptic::cel(k2, kc, 1., 1., 0.)?;
            lambda = one * (p + (-3. + 6. * r2 + 2. * b * r) * ek - 4. * b * r * e)
                / (3. * (b * r).sqrt());
        } else if k2 > 1. {
            let plus = (1. + bpr) * (1. - bpr);
            let ratio = bmr / bpr;
            let mu = 3. * ratio / one;
            let p = ratio * ratio * plus / one;
            let pp = elliptic::cel(invk2, kc, p, 1. + mu, p + mu)?;
            e = elliptic::cel(invk2, kc, 1., 1., kc2)?;
            ek = elliptic::cel(invk2, kc, 1., 1., 0.)?;
            lambda = 2. * one.sqrt() * (plus * pp - (4. - 7. * r2 - b * b) * e) / 3.;
        } else {
            lambda = 2. * (1. - 2. * r).acos()
                - 4. / 3. * (3. + 2. * r - 8. * r2) * (r * (1. - r)).sqrt()
                - if r > 0.5 { 2. * PI } else { 0. };
            e = 1.;
            ek = 1.;
        }
        Ok(((if r > b { 0. } else { 2. * PI } - lambda) / 3., e, ek))
    }
}

struct HIntegral {
    values: Vec<Vec<Option<f64>>>,
    cos: f64,
    sin: f64,
}
impl HIntegral {
    fn new(degree: usize, cos: f64, sin: f64) -> Self {
        let mut values = vec![vec![None; degree.max(1) + 2]; degree + 4];
        values[0][0] = Some(if cos == 0. {
            2. * PI
        } else if sin < 0.5 {
            2. * sin.asin() + PI
        } else {
            2. * cos.acos() + PI
        });
        values[0][1] = Some(-2. * cos);
        Self { values, cos, sin }
    }
    fn get(&mut self, u: usize, v: usize) -> f64 {
        if let Some(value) = self.values[u][v] {
            return value;
        }
        if u % 2 == 1 || (self.cos == 0. && v % 2 == 1) {
            return 0.;
        }
        let value = if u < 2 {
            (-2. * self.cos.powi(u as i32 + 1) * self.sin.powi(v as i32 - 1)
                + (v - 1) as f64 * self.get(u, v - 2))
                / (u + v) as f64
        } else {
            (2. * self.cos.powi(u as i32 - 1) * self.sin.powi(v as i32 + 1)
                + (u - 1) as f64 * self.get(u - 2, v))
                / (u + v) as f64
        };
        self.values[u][v] = Some(value);
        value
    }
}

// Coefficients of (1-t)^u (delta+t)^v, equal to upstream Vieta A.
fn vieta(u: usize, v: usize, delta: f64) -> Vec<f64> {
    let mut coeff = vec![1.];
    for _ in 0..u {
        let mut next = vec![0.; coeff.len() + 1];
        for (i, &a) in coeff.iter().enumerate() {
            next[i] += a;
            next[i + 1] -= a;
        }
        coeff = next;
    }
    for _ in 0..v {
        let mut next = vec![0.; coeff.len() + 1];
        for (i, &a) in coeff.iter().enumerate() {
            next[i] += delta * a;
            next[i + 1] += a;
        }
        coeff = next;
    }
    coeff
}
fn integral(u: usize, v: usize, t: usize, delta: f64, primitives: &[f64]) -> f64 {
    vieta(u, v, delta)
        .iter()
        .enumerate()
        .map(|(i, c)| c * primitives[u + t + i])
        .sum()
}

fn i_integrals(degree: usize, g: &Geometry) -> Result<Vec<f64>> {
    let top = degree + 2;
    let mut out = vec![0.; top + 1];
    if g.k2 >= 1. {
        out[0] = PI;
        for v in 1..=top {
            out[v] = out[v - 1] * (v as f64 - 0.5) / v as f64;
        }
    } else if g.k2 < 0.5 {
        let mut coeff = 2. / (2. * top as f64 + 1.);
        let mut sum = coeff;
        let mut converged = false;
        for n in 1..200 {
            let n = n as f64;
            let top = top as f64;
            coeff *= (2. * n - 1.) * 0.5 * (2. * n + 2. * top - 1.)
                / (n * (2. * n + 2. * top + 1.))
                * g.k2;
            sum += coeff;
            if coeff.abs() <= f64::EPSILON * g.k2 {
                converged = true;
                break;
            }
        }
        require(converged, "I series did not converge")?;
        out[top] = g.k2.powi(top as i32) * g.k * sum;
        for v in (0..top).rev() {
            out[v] = 2. / (2. * v as f64 + 1.)
                * ((v + 1) as f64 * out[v + 1] + g.k2.powi(v as i32) * g.kkc);
        }
    } else {
        out[0] = g.kap0;
        for v in 1..=top {
            out[v] = (0.5 * (2 * v - 1) as f64 * out[v - 1] - g.k2.powi(v as i32 - 1) * g.kkc)
                / v as f64;
        }
    }
    Ok(out)
}
fn j_integrals(degree: usize, g: &Geometry, e: f64, ek: f64) -> Result<Vec<f64>> {
    let top = degree.saturating_sub(1).max(1);
    let mut out = vec![0.; top + 1];
    let low = g.k2 < 1.;
    if g.k2 < 0.5 || g.k2 > 2. {
        for (v, output) in out.iter_mut().enumerate().skip(top - 1) {
            let mut coeff = if low { 3. * PI / 8. } else { PI };
            for i in 1..=v {
                coeff *= if low {
                    (2 * i - 1) as f64 / (2. * (i + 2) as f64)
                } else {
                    1. - 0.5 / i as f64
                };
            }
            let mut sum = coeff;
            let mut converged = false;
            let tol = f64::EPSILON * if low { g.k2 } else { g.invk2 };
            for n in 1..200 {
                let n = n as f64;
                let v = v as f64;
                coeff *= if low {
                    (2. * n - 1.) * (2. * (n + v) - 1.) * 0.25 / (n * (n + v + 2.)) * g.k2
                } else {
                    (1. - 2.5 / n) * (1. - 0.5 / (n + v)) * g.invk2
                };
                sum += coeff;
                if coeff.abs() <= tol {
                    converged = true;
                    break;
                }
            }
            require(converged, "J series did not converge")?;
            *output = if low {
                g.k2.powi(v as i32) * g.k * sum
            } else {
                sum
            };
        }
        for v in (0..top - 1).rev() {
            let w = v as f64;
            out[v] = if low {
                (2. * (3. + w + g.k2 * (1. + w)) * out[v + 1] - (2. * w + 7.) * out[v + 2])
                    / (g.k2 * (2. * w + 1.))
            } else {
                (2. * ((3. + w) * g.invk2 + 1. + w) * out[v + 1]
                    - (2. * w + 7.) * g.invk2 * out[v + 2])
                    / (2. * w + 1.)
            };
        }
    } else {
        if low {
            let f = 2. / (3. * g.k);
            out[0] = f * (e + (3. * g.k2 - 2.) * ek);
            out[1] = 0.2 * f * ((4. - 3. * g.k2) * e + (9. * g.k2 - 8.) * ek);
        } else {
            out[0] = 2. / 3. * ((3. - 2. * g.invk2) * e + g.invk2 * ek);
            out[1] = 0.4 / 3. * ((9. - 8. * g.invk2) * e + (4. * g.invk2 - 3.) * ek);
        }
        for v in 2..=top {
            out[v] = (2. * ((v + 1) as f64 + (v - 1) as f64 * g.k2) * out[v - 1]
                - g.k2 * (2 * v - 3) as f64 * out[v - 2])
                / (2 * v + 3) as f64;
        }
    }
    Ok(out)
}

/// Green solution vector for a circular foreground occultor at (0,b).
/// No/full occultation are handled at the public API boundary.
pub fn green(degree: usize, mut b: f64, r: f64) -> Result<Vec<f64>> {
    require(
        degree <= basis::POLYNOMIAL_LIMIT
            && b.is_finite()
            && b >= 0.
            && r.is_finite()
            && r > 0.
            && b < 1. + r
            && r < 1. + b,
        "Green solver requires partial/nonzero occultation",
    )?;
    if b == 0. {
        // All azimuthal odd moments vanish; radial integration is exact.
        let moments = centered_moments(degree, r)?;
        return basis::a2_inverse(degree)?.left_dot(&moments);
    }
    // Upstream's half-radius contact regularization avoids cancellation when
    // a rotated geometry rounds b and r to opposite sides of equality.
    if (b - r).abs() < 5. * f64::EPSILON && (r - 0.5).abs() < 5. * f64::EPSILON {
        b += 5. * f64::EPSILON;
    }
    let g = Geometry::new(b, r);
    // Underflow/roundoff at an external contact is the limiting full disk.
    if g.k2 == 0. {
        return basis::a2_inverse(degree)?.left_dot(&basis::disk_moments(degree)?);
    }
    let mut s = vec![0.; (degree + 1).pow(2)];
    s[0] = g.s0();
    if degree == 0 {
        return Ok(s);
    }
    let (s2, e, ek) = g.s2()?;
    s[2] = s2;
    let delta = 0.5 * (b - r) / r;
    let two_r = 2. * r;
    let mut factor = two_r.powi(3);
    let k11 = if g.k2 >= 1. {
        PI * (2. * delta + 1.) / 16.
    } else {
        let f = 3. + 6. * delta;
        (2. * g.kkc * (2. * g.k2 * (6. * delta + 4. * g.k2 - 1.) - f) + g.kap0 * f) / 48.
    };
    s[3] = -2. / 3. * g.coslam.powi(3) - 2. * factor * k11;
    if degree == 1 {
        return Ok(s);
    }
    let ip = i_integrals(degree, &g)?;
    let jp = j_integrals(degree, &g, e, ek)?;
    let mut h = HIntegral::new(degree, g.coslam, g.sinlam);
    let mut lfactor = (1. - (b - r).powi(2)).max(0.).powf(1.5);
    for l in 2..=degree {
        factor *= two_r;
        lfactor *= two_r;
        for m in -(l as isize)..=l as isize {
            let mu = (l as isize - m) as usize;
            let nu = (l as isize + m) as usize;
            if mu % 4 == 3 || mu % 4 == 2 {
                continue;
            }
            let q = if !mu.is_multiple_of(4) || (g.qcond && !nu.is_multiple_of(4)) {
                0.
            } else {
                h.get((mu + 4) / 2, nu / 2)
            };
            let p = if mu.is_multiple_of(4) {
                2. * factor * integral((mu + 4) / 4, nu / 2, 0, delta, &ip)
            } else if mu == 1 && l % 2 == 0 {
                lfactor
                    * (integral((l - 2) / 2, 0, 0, delta, &jp)
                        - 2. * integral((l - 2) / 2, 0, 1, delta, &jp))
            } else if mu == 1 && l % 2 == 1 {
                lfactor
                    * (integral((l - 3) / 2, 1, 0, delta, &jp)
                        - 2. * integral((l - 3) / 2, 1, 1, delta, &jp))
            } else if mu % 4 == 1 {
                2. * lfactor * integral((mu - 1) / 4, (nu - 1) / 2, 0, delta, &jp)
            } else {
                0.
            };
            s[basis::index(l, m)] = q - p;
        }
    }
    require(
        s.iter().all(|v| v.is_finite()),
        "nonfinite analytic Green solution",
    )?;
    Ok(s)
}

fn centered_moments(degree: usize, r: f64) -> Result<Vec<f64>> {
    let mut values = basis::disk_moments(degree)?;
    // Integral outside radius r, separating radial and angular factors.
    for (i, (a, b, c)) in basis::terms(degree).into_iter().enumerate() {
        if values[i] == 0. {
            continue;
        }
        if c == 0 {
            values[i] *= 1. - r.powi((a + b + 2) as i32);
        } else {
            // ∫_r^1 rho^(a+b+1) sqrt(1-rho²) d rho.
            // Stable positive recurrence in t=1-r² for incomplete beta.
            let k = (a + b) / 2;
            let t = 1. - r * r;
            let mut radial = 2. / 3. * t.powf(1.5);
            let mut total = 2. / 3.;
            for j in 1..=k {
                radial =
                    (j as f64 * radial + r.powi(2 * j as i32) * t.powf(1.5)) / (j as f64 + 1.5);
                total *= j as f64 / (j as f64 + 1.5);
            }
            values[i] *= radial / total;
        }
    }
    Ok(values)
}

/// Analytic flux row, in starry's observer Cartesian frame, for a single occultor.
pub fn flux_design(a1: &crate::matrix::Matrix, degree: usize, occ: Occultor) -> Result<Vec<f64>> {
    let a = basis::a2_inverse(degree)?.solve(a1)?;
    flux_design_cached(a1, &a, degree, occ)
}

pub(crate) fn flux_design_cached(
    a1: &crate::matrix::Matrix,
    a: &crate::matrix::Matrix,
    degree: usize,
    occ: Occultor,
) -> Result<Vec<f64>> {
    Occultor::new(occ.x, occ.y, occ.radius)?;
    let b = occ.x.hypot(occ.y);
    let r = occ.radius;
    if r == 0. || b >= 1. + r {
        return a1.left_dot(&basis::disk_moments(degree)?);
    }
    if r >= 1. + b {
        return Ok(vec![0.; (degree + 1).pow(2)]);
    }
    let row = a.left_dot(&green(degree, b, r)?)?;
    // Position angle aligns the occultor with +y. Harmonics rotate in +/-m pairs.
    let angle = occ.x.atan2(occ.y);
    let mut rotated = row.clone();
    for l in 1..=degree {
        for m in 1..=l {
            let pos = basis::index(l, m as isize);
            let neg = basis::index(l, -(m as isize));
            let (s, c) = (m as f64 * angle).sin_cos();
            rotated[pos] = c * row[pos] + s * row[neg];
            rotated[neg] = c * row[neg] - s * row[pos];
        }
    }
    require(
        rotated.iter().all(|v| v.is_finite()),
        "nonfinite analytic flux",
    )?;
    Ok(rotated)
}

/// Independent adaptive-integration design row for convergence checks.
pub fn numerical_flux_design(
    a1: &crate::matrix::Matrix,
    degree: usize,
    occ: Occultor,
    options: Integration,
) -> Result<Vec<f64>> {
    a1.left_dot(&crate::occultation::visible_moments(degree, occ, options)?.values)
}

/// Analytic flux of a canonical polynomial map, useful for limb-darkening products.
pub fn polynomial_flux(degree: usize, p: &[f64], occ: Option<Occultor>) -> Result<f64> {
    require(
        degree <= basis::POLYNOMIAL_LIMIT
            && p.len() == (degree + 1).pow(2)
            && p.iter().all(|v| v.is_finite()),
        "invalid polynomial map",
    )?;
    let Some(occ) = occ else {
        return Ok(crate::map::dot(p, &basis::disk_moments(degree)?));
    };
    Occultor::new(occ.x, occ.y, occ.radius)?;
    let b = occ.x.hypot(occ.y);
    let r = occ.radius;
    if r == 0. || b >= 1. + r {
        return Ok(crate::map::dot(p, &basis::disk_moments(degree)?));
    }
    if r >= 1. + b {
        return Ok(0.);
    }
    let angle = occ.x.atan2(occ.y);
    let transform = crate::rotation::transpose(crate::rotation::axis_angle([0., 0., 1.], angle)?);
    let aligned = crate::rotation::substitute(degree, p, transform);
    let rhs = crate::matrix::Matrix {
        rows: p.len(),
        cols: 1,
        data: aligned,
    };
    let g = basis::a2_inverse(degree)?.solve(&rhs)?.data;
    Ok(crate::map::dot(&g, &green(degree, b, r)?))
}
