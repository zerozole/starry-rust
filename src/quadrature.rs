//! Direct surface quadrature in stable harmonic coordinates. Conic intersections
//! split the domain; sine substitutions regularize stellar and occultor limbs.
use crate::{Error, Result, conic::Ellipse, occultation::gauss, require};
use std::f64::consts::PI;

fn rule(f: &impl Fn(f64) -> Result<f64>, a: f64, b: f64, nodes: &[(f64, f64)]) -> Result<f64> {
    let mut value = 0.;
    for &(x, w) in nodes {
        value += w * f((a + b) / 2. + (b - a) / 2. * x)?;
    }
    Ok(value * (b - a) / 2.)
}

fn adaptive(
    f: &impl Fn(f64) -> Result<f64>,
    a: f64,
    b: f64,
    tolerance: f64,
    depth: usize,
    nodes: &[Vec<(f64, f64)>; 2],
) -> Result<f64> {
    let low = rule(f, a, b, &nodes[0])?;
    let high = rule(f, a, b, &nodes[1])?;
    if (high - low).abs() <= tolerance + 2e-12 * high.abs() {
        return Ok(high);
    }
    if depth == 0 {
        return Err(Error("surface quadrature did not converge".into()));
    }
    let mid = (a + b) / 2.;
    Ok(adaptive(f, a, mid, tolerance / 2., depth - 1, nodes)?
        + adaptive(f, mid, b, tolerance / 2., depth - 1, nodes)?)
}

/// Integrate a field over an elliptical disk, excluding a union of silhouettes.
/// The field receives unit-sphere coordinates (x,y/q,z). `terminator` clips to
/// y/q >= terminator*sqrt(1-x*x), in the illumination-aligned frame.
/// Errors are estimated by order doubling; this is not a rigorous error bound.
pub fn surface(
    axis_ratio: f64,
    terminator: f64,
    ellipses: &[Ellipse],
    tolerance: f64,
    field: impl Fn([f64; 3]) -> Result<f64>,
) -> Result<f64> {
    require(
        axis_ratio.is_finite()
            && axis_ratio > 0.
            && axis_ratio <= 1.
            && terminator.is_finite()
            && (-1. ..=1.).contains(&terminator)
            && tolerance.is_finite()
            && tolerance > 0.,
        "invalid surface quadrature",
    )?;
    let boundary = Ellipse {
        x: 0.,
        y: 0.,
        major: 1.,
        minor: axis_ratio,
        angle: 0.,
    };
    let mut cuts = vec![-1., 1.];
    for (i, e) in ellipses.iter().enumerate() {
        e.validate()?;
        cuts.extend(e.x_extent().into_iter().filter(|x| *x > -1. && *x < 1.));
        cuts.extend(crate::conic::intersections(boundary, *e));
        if terminator.abs() > 1e-14 && terminator.abs() < 1. {
            cuts.extend(crate::conic::intersections(
                Ellipse {
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
    let nodes = [gauss(32), gauss(64)];
    let chord = |x: f64| -> Result<f64> {
        let radius = ((1. - x) * (1. + x)).max(0.).sqrt();
        if radius == 0. {
            return Ok(0.);
        }
        let lower = terminator * radius * axis_ratio;
        let upper = radius * axis_ratio;
        let mut hidden: Vec<_> = ellipses
            .iter()
            .filter_map(|e| e.y_bounds(x))
            .map(|(a, b)| (a.max(lower), b.min(upper)))
            .filter(|(a, b)| a < b)
            .collect();
        hidden.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut visible = vec![];
        let mut left = lower;
        for (a, b) in hidden {
            if a > left {
                visible.push((left, a));
            }
            left = left.max(b);
        }
        if left < upper {
            visible.push((left, upper));
        }
        let mut value = 0.;
        for (a, b) in visible {
            let a = (a / (axis_ratio * radius)).clamp(-1., 1.).asin();
            let b = (b / (axis_ratio * radius)).clamp(-1., 1.).asin();
            value += adaptive(
                &|t| {
                    let p = [x, radius * t.sin(), radius * t.cos()];
                    let v = field(p)?;
                    require(v.is_finite(), "nonfinite surface integrand")?;
                    Ok(v * axis_ratio * radius * t.cos())
                },
                a,
                b,
                tolerance * 0.1,
                12,
                &nodes,
            )?;
        }
        Ok(value)
    };
    let mut sum = 0.;
    for pair in cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        sum += adaptive(
            &|t| {
                let x = a + (b - a) * t.sin().powi(2);
                Ok(chord(x)? * (b - a) * (2. * t).sin())
            },
            0.,
            PI / 2.,
            tolerance / (cuts.len() - 1) as f64,
            18,
            &nodes,
        )?;
    }
    Ok(sum)
}

pub fn circles(occultors: &[crate::occultation::Occultor]) -> Result<Vec<Ellipse>> {
    let mut result = vec![];
    for o in occultors {
        crate::occultation::Occultor::new(o.x, o.y, o.radius)?;
        if o.radius > 0. {
            result.push(Ellipse {
                x: o.x,
                y: o.y,
                major: o.radius,
                minor: o.radius,
                angle: 0.,
            });
        }
    }
    Ok(result)
}

/// Integrate along the visible terminator, with dx measure (not arc length).
/// Used for the domain term when differentiating reflected illumination.
pub(crate) fn terminator(
    b: f64,
    ellipses: &[Ellipse],
    field: impl Fn([f64; 3]) -> Result<f64>,
) -> Result<f64> {
    require(
        b.is_finite() && (-1. ..=1.).contains(&b),
        "invalid terminator",
    )?;
    let mut cuts = vec![-1., 1.];
    for e in ellipses {
        e.validate()?;
        if b.abs() > 1e-14 {
            cuts.extend(crate::conic::intersections(
                Ellipse {
                    x: 0.,
                    y: 0.,
                    major: 1.,
                    minor: b.abs(),
                    angle: 0.,
                },
                *e,
            ));
        } else {
            let p = e.conic();
            cuts.extend(crate::polynomial_roots::real_roots(
                &[p[5], p[3], p[0]],
                -1.,
                1.,
            ));
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
    let nodes = [gauss(32), gauss(64)];
    let mut value = 0.;
    let point = |x: f64| {
        let r = ((1. - x) * (1. + x)).max(0.).sqrt();
        [x, b * r, (1. - b * b).max(0.).sqrt() * r]
    };
    for pair in cuts.windows(2) {
        let p = point((pair[0] + pair[1]) / 2.);
        if ellipses.iter().any(|e| e.contains(p[0], p[1])) {
            continue;
        }
        value += adaptive(
            &|t| {
                let x = pair[0] + (pair[1] - pair[0]) * t.sin().powi(2);
                Ok(field(point(x))? * (pair[1] - pair[0]) * (2. * t).sin())
            },
            0.,
            PI / 2.,
            2e-12 / (cuts.len() - 1) as f64,
            18,
            &nodes,
        )?;
    }
    Ok(value)
}

/// Shape derivative with respect to a circular occultor's center and radius.
/// The stellar limb is elliptical; the field receives unit-sphere coordinates.
pub fn circular_boundary(
    axis_ratio: f64,
    occultor: crate::occultation::Occultor,
    field: impl Fn([f64; 3]) -> Result<f64>,
) -> Result<[f64; 3]> {
    require(
        axis_ratio.is_finite() && axis_ratio > 0. && axis_ratio <= 1.,
        "invalid axis ratio",
    )?;
    let silhouettes = circles(&[occultor])?;
    if silhouettes.is_empty() {
        return Ok([0.; 3]);
    }
    let o = occultor;
    require(
        !(axis_ratio == 1. && o.x == 0. && o.y == 0. && o.radius == 1.),
        "gradient undefined for coincident unit disks",
    )?;
    let star = Ellipse {
        x: 0.,
        y: 0.,
        major: 1.,
        minor: axis_ratio,
        angle: 0.,
    };
    let mut cuts = vec![0., 2. * PI];
    for x in crate::conic::intersections(star, silhouettes[0]) {
        let v = (x - o.x) / o.radius;
        if v.abs() <= 1. + 1e-12 {
            let t = v.clamp(-1., 1.).acos();
            cuts.extend([t, 2. * PI - t]);
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
    let nodes = [gauss(32), gauss(64)];
    let mut result = [0.; 3];
    for pair in cuts.windows(2) {
        let mid = (pair[0] + pair[1]) / 2.;
        if !star.contains(o.x + o.radius * mid.cos(), o.y + o.radius * mid.sin()) {
            continue;
        }
        for (k, value) in result.iter_mut().enumerate() {
            *value += adaptive(
                &|t| {
                    let theta = pair[0] + (pair[1] - pair[0]) * t.sin().powi(2);
                    let x = o.x + o.radius * theta.cos();
                    let y = (o.y + o.radius * theta.sin()) / axis_ratio;
                    let p = [x, y, (1. - x * x - y * y).max(0.).sqrt()];
                    let normal = match k {
                        0 => theta.cos(),
                        1 => theta.sin(),
                        _ => 1.,
                    };
                    Ok(-o.radius * normal * field(p)? * (pair[1] - pair[0]) * (2. * t).sin())
                },
                0.,
                PI / 2.,
                2e-11 / (cuts.len() - 1) as f64,
                24,
                &nodes,
            )?;
        }
    }
    Ok(result)
}

/// Shape derivatives of an elliptical occultor clipped by the stellar ellipse
/// and a union of other foreground silhouettes. Order: x,y,major,minor,angle.
pub fn ellipse_boundary(
    axis_ratio: f64,
    target: Ellipse,
    others: &[Ellipse],
    field: impl Fn([f64; 3]) -> Result<f64>,
) -> Result<[f64; 5]> {
    target.validate()?;
    require(
        axis_ratio.is_finite() && axis_ratio > 0. && axis_ratio <= 1.,
        "invalid stellar axis ratio",
    )?;
    let star = Ellipse {
        x: 0.,
        y: 0.,
        major: 1.,
        minor: axis_ratio,
        angle: 0.,
    };
    let (s, c) = target.angle.sin_cos();
    let hx = (target.major * c).hypot(target.minor * s);
    let phi = (-target.minor * s).atan2(target.major * c);
    let point = |t: f64| {
        [
            target.x + target.major * c * t.cos() - target.minor * s * t.sin(),
            target.y + target.major * s * t.cos() + target.minor * c * t.sin(),
        ]
    };
    let mut cuts = vec![0., 2. * PI];
    for e in std::iter::once(&star).chain(others) {
        e.validate()?;
        require(
            !(target.x == e.x
                && target.y == e.y
                && target.major == e.major
                && target.minor == e.minor
                && target.angle == e.angle),
            "coincident silhouette derivative is undefined",
        )?;
        for x in crate::conic::intersections(target, *e) {
            let v = (x - target.x) / hx;
            if v.abs() <= 1. + 1e-12 {
                let a = v.clamp(-1., 1.).acos();
                cuts.extend([(phi + a).rem_euclid(2. * PI), (phi - a).rem_euclid(2. * PI)]);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
    let nodes = [gauss(32), gauss(64)];
    let mut result = [0.; 5];
    for pair in cuts.windows(2) {
        let p = point((pair[0] + pair[1]) / 2.);
        if !star.contains(p[0], p[1]) || others.iter().any(|e| e.contains(p[0], p[1])) {
            continue;
        }
        for (k, result) in result.iter_mut().enumerate() {
            *result += adaptive(
                &|t| {
                    let angle = pair[0] + (pair[1] - pair[0]) * t.sin().powi(2);
                    let p = point(angle);
                    let y = p[1] / axis_ratio;
                    let q = [p[0], y, (1. - p[0] * p[0] - y * y).max(0.).sqrt()];
                    let (sa, ca) = angle.sin_cos();
                    let normal = match k {
                        0 => c * target.minor * ca - s * target.major * sa,
                        1 => s * target.minor * ca + c * target.major * sa,
                        2 => target.minor * ca * ca,
                        3 => target.major * sa * sa,
                        _ => (target.major.powi(2) - target.minor.powi(2)) * sa * ca,
                    };
                    Ok(-field(q)? * normal * (pair[1] - pair[0]) * (2. * t).sin())
                },
                0.,
                PI / 2.,
                2e-11 / (cuts.len() - 1) as f64,
                24,
                &nodes,
            )?;
        }
    }
    Ok(result)
}
