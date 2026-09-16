//! Circular occultation of polynomial maps.
//!
//! The inner (y) integral is analytic. The outer integral uses adaptive Gaussian
//! quadrature split at limb intersections. This is a numerical replacement for
//! upstream's elliptic Green solver, explicitly not the completed analytic port.
use crate::{Error, Result, basis, require};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Occultor {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
}
impl Occultor {
    pub fn new(x: f64, y: f64, radius: f64) -> Result<Self> {
        require(
            [x, y, radius].iter().all(|v| v.is_finite()) && radius >= 0. && x.hypot(y).is_finite(),
            "invalid occultor",
        )?;
        Ok(Self { x, y, radius })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Integration {
    pub absolute_tolerance: f64,
    pub max_depth: usize,
}
impl Default for Integration {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-11,
            max_depth: 18,
        }
    }
}
#[derive(Debug)]
pub struct Moments {
    pub values: Vec<f64>,
    pub estimated_error: f64,
    pub evaluations: usize,
}

pub(crate) fn gauss(n: usize) -> Vec<(f64, f64)> {
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..30 {
            let (p, prev) = legendre(n, z);
            let dp = n as f64 * (z * p - prev) / (z * z - 1.);
            let next = z - p / dp;
            if (next - z).abs() < 2e-16 {
                z = next;
                break;
            }
            z = next;
        }
        let (p, prev) = legendre(n, z);
        let dp = n as f64 * (z * p - prev) / (z * z - 1.);
        nodes.push((z, 2. / ((1. - z * z) * dp * dp)));
    }
    nodes
}
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let mut prev = 0.;
    let mut p = 1.;
    for k in 1..=n {
        let q = ((2 * k - 1) as f64 * x * p - (k - 1) as f64 * prev) / k as f64;
        prev = p;
        p = q;
    }
    (p, prev)
}

// ∫ u^b sqrt(1-u²) du, all b through degree, endpoints in [-1,1].
fn primitives(degree: usize, u: f64) -> Vec<f64> {
    let u = u.clamp(-1., 1.);
    let w = (1. - u * u).max(0.).sqrt();
    let mut j = vec![0.; degree + 1];
    j[0] = 0.5 * (u.asin() + u * w);
    if degree > 0 {
        j[1] = -w.powi(3) / 3.;
    }
    for b in 2..=degree {
        j[b] = ((b - 1) as f64 * j[b - 2] - u.powi(b as i32 - 1) * w.powi(3)) / (b + 2) as f64;
    }
    j
}

struct Integrator {
    degree: usize,
    occultors: Vec<Occultor>,
    ellipses: Vec<crate::conic::Ellipse>,
    terminator: f64,
    axis_ratio: f64,
    terms: Vec<(usize, usize, usize)>,
    g16: Vec<(f64, f64)>,
    g32: Vec<(f64, f64)>,
    evaluations: usize,
}
impl Integrator {
    fn slice(&mut self, x: f64) -> Vec<f64> {
        self.evaluations += 1;
        let mut out = vec![0.; self.terms.len()];
        let h = (1. - x * x).max(0.).sqrt();
        let mut ranges = vec![];
        for occ in &self.occultors {
            let dx = (x - occ.x).abs();
            if dx >= occ.radius {
                continue;
            }
            let q = ((occ.radius - dx) * (occ.radius + dx)).sqrt();
            let lo = (self.terminator * h).max((occ.y - q) / self.axis_ratio);
            let hi = h.min((occ.y + q) / self.axis_ratio);
            if lo < hi {
                ranges.push((lo, hi));
            }
        }
        ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
        for ellipse in &self.ellipses {
            if let Some((lo, hi)) = ellipse.y_bounds(x) {
                let lo = (self.terminator * h).max(lo / self.axis_ratio);
                let hi = h.min(hi / self.axis_ratio);
                if lo < hi {
                    ranges.push((lo, hi));
                }
            }
        }
        ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = vec![];
        for (a, b) in ranges {
            if let Some(last) = merged.last_mut()
                && a <= last.1
            {
                last.1 = last.1.max(b);
                continue;
            }
            merged.push((a, b));
        }
        if h == 0. {
            return out;
        }
        for (lo, hi) in merged {
            let jl = primitives(self.degree, lo / h);
            let jh = primitives(self.degree, hi / h);
            for (i, &(a, b, c)) in self.terms.iter().enumerate() {
                let y = if c == 0 {
                    (hi.powi(b as i32 + 1) - lo.powi(b as i32 + 1)) / (b + 1) as f64
                } else {
                    h.powi(b as i32 + 2) * (jh[b] - jl[b])
                };
                out[i] += self.axis_ratio * x.powi(a as i32) * y;
            }
        }
        out
    }
    fn rule(&mut self, a: f64, b: f64, high: bool) -> Vec<f64> {
        let nodes = if high {
            self.g32.clone()
        } else {
            self.g16.clone()
        };
        let mut out = vec![0.; self.terms.len()];
        // Smooth square-root endpoint behavior with x=mid+half*sin(t).
        let mid = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        for (t, w) in nodes {
            let angle = 0.5 * PI * t;
            let x = mid + half * angle.sin();
            let v = self.slice(x);
            let weight = w * 0.5 * PI * half * angle.cos();
            for (o, v) in out.iter_mut().zip(v) {
                *o += weight * v;
            }
        }
        out
    }
    fn integrate(&mut self, a: f64, b: f64, tol: f64, depth: usize) -> Result<(Vec<f64>, f64)> {
        let low = self.rule(a, b, false);
        let high = self.rule(a, b, true);
        let error = low
            .iter()
            .zip(&high)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if error <= tol {
            return Ok((high, error));
        }
        if depth == 0 {
            return Err(Error(format!(
                "occultation quadrature did not converge; error={error:e}"
            )));
        }
        let mid = 0.5 * (a + b);
        let (mut l, le) = self.integrate(a, mid, tol / 2., depth - 1)?;
        let (r, re) = self.integrate(mid, b, tol / 2., depth - 1)?;
        for (l, r) in l.iter_mut().zip(r) {
            *l += r;
        }
        Ok((l, le + re))
    }
}

/// Visible disk moments; estimated_error is an absolute heuristic per moment.
pub fn visible_moments(degree: usize, occ: Occultor, options: Integration) -> Result<Moments> {
    clipped_moments(degree, occ, options, -1., basis::disk_moments(degree)?)
}

pub(crate) fn clipped_moments(
    degree: usize,
    occ: Occultor,
    options: Integration,
    terminator: f64,
    mut full: Vec<f64>,
) -> Result<Moments> {
    Occultor::new(occ.x, occ.y, occ.radius)?;
    require(
        options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid integration options",
    )?;
    let d = occ.x.hypot(occ.y);
    let r = occ.radius;
    if r == 0. || d >= 1. + r {
        return Ok(Moments {
            values: full,
            estimated_error: 0.,
            evaluations: 0,
        });
    }
    if r >= 1. + d {
        return Ok(Moments {
            values: vec![0.; full.len()],
            estimated_error: 0.,
            evaluations: 0,
        });
    }
    let lo = (-1_f64).max(occ.x - r);
    let hi = 1_f64.min(occ.x + r);
    let mut cuts = vec![lo, hi];
    if terminator.abs() < 1. {
        // [(1-b²)x²-2xo*x+xo²+yo²+b²-r²]² - 4b²yo²(1-x²).
        let a = 1. - terminator * terminator;
        let bb = -2. * occ.x;
        let c = occ.x * occ.x + occ.y * occ.y + terminator * terminator - r * r;
        let t = 4. * terminator * terminator * occ.y * occ.y;
        cuts.extend(crate::polynomial_roots::real_roots(
            &[
                c * c - t,
                2. * bb * c,
                bb * bb + 2. * a * c + t,
                2. * a * bb,
                a * a,
            ],
            lo,
            hi,
        ));
    }
    if d > 0. && d > (1. - r).abs() && d < 1. + r {
        let along = (1. - r * r + d * d) / (2. * d);
        let side = (1. - along * along).max(0.).sqrt();
        let base = along * occ.x / d;
        let offset = side * occ.y / d;
        for x in [base - offset, base + offset] {
            if x > lo && x < hi {
                cuts.push(x);
            }
        }
    }
    // Separate the center of the occultor to aid symmetric/small-radius cases.
    if occ.x > lo && occ.x < hi {
        cuts.push(occ.x);
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    let mut engine = Integrator {
        ellipses: vec![],
        degree,
        occultors: vec![occ],
        terminator,
        axis_ratio: 1.,
        terms: basis::terms(degree),
        g16: gauss(16),
        g32: gauss(32),
        evaluations: 0,
    };
    let mut error = 0.;
    let tol = options.absolute_tolerance / (cuts.len() - 1) as f64;
    for pair in cuts.windows(2) {
        let (blocked, e) = engine.integrate(pair[0], pair[1], tol, options.max_depth)?;
        error += e;
        for (f, v) in full.iter_mut().zip(blocked) {
            *f -= v;
        }
    }
    Ok(Moments {
        values: full,
        estimated_error: error,
        evaluations: engine.evaluations,
    })
}

/// Integrate through the union of any number of opaque foreground circles.
pub fn union_moments(
    degree: usize,
    occultors: &[Occultor],
    options: Integration,
) -> Result<Moments> {
    clipped_union(
        degree,
        occultors,
        options,
        -1.,
        basis::disk_moments(degree)?,
    )
}

pub(crate) fn clipped_union(
    degree: usize,
    occultors: &[Occultor],
    options: Integration,
    terminator: f64,
    mut full: Vec<f64>,
) -> Result<Moments> {
    require(
        degree <= basis::MAX_DEGREE
            && options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid union integration",
    )?;
    let mut active = vec![];
    let mut cuts = vec![-1., 1.];
    for &o in occultors {
        Occultor::new(o.x, o.y, o.radius)?;
        if o.radius == 0. || o.x.hypot(o.y) >= 1. + o.radius {
            continue;
        }
        if o.radius >= 1. + o.x.hypot(o.y) {
            return Ok(Moments {
                values: vec![0.; full.len()],
                estimated_error: 0.,
                evaluations: 0,
            });
        }
        for x in [o.x - o.radius, o.x, o.x + o.radius] {
            if x > -1. && x < 1. {
                cuts.push(x);
            }
        }
        if terminator.abs() < 1. {
            let a = 1. - terminator * terminator;
            let b = -2. * o.x;
            let c = o.x * o.x + o.y * o.y + terminator * terminator - o.radius * o.radius;
            let t = 4. * terminator * terminator * o.y * o.y;
            cuts.extend(crate::polynomial_roots::real_roots(
                &[
                    c * c - t,
                    2. * b * c,
                    b * b + 2. * a * c + t,
                    2. * a * b,
                    a * a,
                ],
                -1.,
                1.,
            ));
        }
        active.push(o);
    }
    if active.is_empty() {
        return Ok(Moments {
            values: full,
            estimated_error: 0.,
            evaluations: 0,
        });
    }
    let mut circles = active.clone();
    circles.push(Occultor::new(0., 0., 1.)?);
    for i in 0..circles.len() {
        for j in 0..i {
            let a = circles[i];
            let b = circles[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let d = dx.hypot(dy);
            if d == 0. || d >= a.radius + b.radius || d <= (a.radius - b.radius).abs() {
                continue;
            }
            let along = (a.radius * a.radius - b.radius * b.radius + d * d) / (2. * d);
            let h = (a.radius * a.radius - along * along).max(0.).sqrt();
            for x in [
                a.x + along * dx / d + h * dy / d,
                a.x + along * dx / d - h * dy / d,
            ] {
                if x > -1. && x < 1. {
                    cuts.push(x);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
    let mut integrator = Integrator {
        ellipses: vec![],
        degree,
        occultors: active,
        terminator,
        axis_ratio: 1.,
        terms: basis::terms(degree),
        g16: gauss(16),
        g32: gauss(32),
        evaluations: 0,
    };
    let mut error = 0.;
    let tol = options.absolute_tolerance / (cuts.len() - 1) as f64;
    for pair in cuts.windows(2) {
        let (v, e) = integrator.integrate(pair[0], pair[1], tol, options.max_depth)?;
        error += e;
        for (a, b) in full.iter_mut().zip(v) {
            *a -= b;
        }
    }
    Ok(Moments {
        values: full,
        estimated_error: error,
        evaluations: integrator.evaluations,
    })
}

/// Ellipse x²+(y/q)²<=1. Moments use unit-sphere coordinates (x,y/q,z).
pub fn ellipse_moments(
    degree: usize,
    axis_ratio: f64,
    occ: Option<Occultor>,
    options: Integration,
) -> Result<Moments> {
    require(
        axis_ratio.is_finite()
            && axis_ratio > 0.
            && axis_ratio <= 1.
            && options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid ellipse integration",
    )?;
    let mut full = basis::disk_moments(degree)?;
    for v in &mut full {
        *v *= axis_ratio;
    }
    let Some(o) = occ else {
        return Ok(Moments {
            values: full,
            estimated_error: 0.,
            evaluations: 0,
        });
    };
    Occultor::new(o.x, o.y, o.radius)?;
    if o.radius == 0. || o.x.hypot(o.y) >= 1. + o.radius {
        return Ok(Moments {
            values: full,
            estimated_error: 0.,
            evaluations: 0,
        });
    }
    if o.radius >= 1. + o.x.hypot(o.y) {
        return Ok(Moments {
            values: vec![0.; full.len()],
            estimated_error: 0.,
            evaluations: 0,
        });
    }
    let lo = (-1_f64).max(o.x - o.radius);
    let hi = 1_f64.min(o.x + o.radius);
    let mut cuts = vec![lo, hi];
    let a = 1. - axis_ratio * axis_ratio;
    let b = -2. * o.x;
    let c = o.x * o.x + o.y * o.y + axis_ratio * axis_ratio - o.radius * o.radius;
    let t = 4. * axis_ratio * axis_ratio * o.y * o.y;
    cuts.extend(crate::polynomial_roots::real_roots(
        &[
            c * c - t,
            2. * b * c,
            b * b + 2. * a * c + t,
            2. * a * b,
            a * a,
        ],
        lo,
        hi,
    ));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
    let mut integrator = Integrator {
        ellipses: vec![],
        degree,
        occultors: vec![o],
        terminator: -1.,
        axis_ratio,
        terms: basis::terms(degree),
        g16: gauss(16),
        g32: gauss(32),
        evaluations: 0,
    };
    let tol = options.absolute_tolerance / (cuts.len() - 1) as f64;
    let mut error = 0.;
    for pair in cuts.windows(2) {
        let (v, e) = integrator.integrate(pair[0], pair[1], tol, options.max_depth)?;
        error += e;
        for (a, b) in full.iter_mut().zip(v) {
            *a -= b;
        }
    }
    Ok(Moments {
        values: full,
        estimated_error: error,
        evaluations: integrator.evaluations,
    })
}

/// Analytic overlap of two disks. Flux of a uniform unit-flux map.
pub fn projected_moments(
    degree: usize,
    axis_ratio: f64,
    ellipses: &[crate::conic::Ellipse],
    options: Integration,
) -> Result<Moments> {
    projected_clipped(
        degree,
        axis_ratio,
        ellipses,
        options,
        -1.,
        basis::disk_moments(degree)?,
    )
}

pub(crate) fn projected_clipped(
    degree: usize,
    axis_ratio: f64,
    ellipses: &[crate::conic::Ellipse],
    options: Integration,
    terminator: f64,
    mut full: Vec<f64>,
) -> Result<Moments> {
    require(
        axis_ratio.is_finite()
            && axis_ratio > 0.
            && axis_ratio <= 1.
            && options.absolute_tolerance.is_finite()
            && options.absolute_tolerance > 0.
            && options.max_depth <= 30,
        "invalid projected integration",
    )?;
    require(
        terminator.is_finite()
            && (-1. ..=1.).contains(&terminator)
            && full.len() == (degree + 1).pow(2),
        "invalid clipped ellipse",
    )?;
    for v in &mut full {
        *v *= axis_ratio;
    }
    let boundary = crate::conic::Ellipse {
        x: 0.,
        y: 0.,
        major: 1.,
        minor: axis_ratio,
        angle: 0.,
    };
    let mut cuts = vec![-1., 1.];
    for (i, e) in ellipses.iter().enumerate() {
        e.validate()?;
        cuts.extend(e.x_extent().into_iter().filter(|v| *v > -1. && *v < 1.));
        cuts.extend(crate::conic::intersections(boundary, *e));
        if terminator.abs() > 1e-14 && terminator.abs() < 1. {
            cuts.extend(crate::conic::intersections(
                crate::conic::Ellipse {
                    minor: axis_ratio * terminator.abs(),
                    ..boundary
                },
                *e,
            ));
        } else if terminator.abs() <= 1e-14 {
            let p = e.conic();
            cuts.extend(crate::polynomial_roots::real_roots(
                &[p[5], p[3], p[0]],
                -1.,
                1.,
            ));
        }
        for other in &ellipses[..i] {
            cuts.extend(crate::conic::intersections(*other, *e));
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
    let mut integrator = Integrator {
        degree,
        occultors: vec![],
        ellipses: ellipses.to_vec(),
        terminator,
        axis_ratio,
        terms: basis::terms(degree),
        g16: gauss(16),
        g32: gauss(32),
        evaluations: 0,
    };
    let mut error = 0.;
    let tolerance = options.absolute_tolerance / (cuts.len() - 1) as f64;
    for pair in cuts.windows(2) {
        let (values, e) = integrator.integrate(pair[0], pair[1], tolerance, options.max_depth)?;
        error += e;
        for (a, b) in full.iter_mut().zip(values) {
            *a -= b;
        }
    }
    Ok(Moments {
        values: full,
        estimated_error: error,
        evaluations: integrator.evaluations,
    })
}

/// Analytic overlap of two disks. Flux of a uniform unit-flux map.
pub fn uniform_flux(separation: f64, radius: f64) -> Result<f64> {
    require(
        separation.is_finite() && separation >= 0. && radius.is_finite() && radius >= 0.,
        "invalid disk geometry",
    )?;
    let b = separation;
    let r = radius;
    if r == 0. || b >= 1. + r {
        return Ok(1.);
    }
    if r >= 1. + b {
        return Ok(0.);
    }
    if b + r <= 1. {
        return Ok(1. - r * r);
    }
    let c1 = ((b * b + 1. - r * r) / (2. * b)).clamp(-1., 1.);
    let c2 = ((b * b + r * r - 1.) / (2. * b * r)).clamp(-1., 1.);
    let area = (-b + 1. + r) * (b + 1. - r) * (b - 1. + r) * (b + 1. + r);
    Ok((1. - (c1.acos() + r * r * c2.acos() - 0.5 * area.max(0.).sqrt()) / PI).clamp(0., 1.))
}
