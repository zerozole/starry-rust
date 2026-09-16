//! Polynomial, spherical harmonic, and Green basis transformations.
//! Adapted from upstream `basis.h` (MIT; see LICENSE).
use crate::{Result, matrix::Matrix, require};
use std::f64::consts::PI;

pub const MAX_DEGREE: usize = 32;
/// Limit for specialized kernels that still convert full fields to polynomials.
pub const POLYNOMIAL_LIMIT: usize = 20;

/// Analytic derivative of harmonics in a tangent direction on the unit sphere.
/// Cartesian sector recurrences remain regular at both poles.
pub fn harmonic_directional(degree: usize, xyz: [f64; 3], tangent: [f64; 3]) -> Result<Vec<f64>> {
    require(
        degree <= MAX_DEGREE
            && xyz.iter().chain(&tangent).all(|v| v.is_finite())
            && (xyz.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-10
            && xyz
                .iter()
                .zip(tangent)
                .map(|(x, t)| x * t)
                .sum::<f64>()
                .abs()
                < 1e-10,
        "invalid harmonic tangent",
    )?;
    let [x, y, z] = xyz;
    let [dx, dy, dz] = tangent;
    let mut sector = [1. / PI, 0.];
    let mut derivative = [0.; 2];
    let mut out = vec![0.; (degree + 1).pow(2)];
    for m in 0..=degree {
        if m > 0 {
            let scale = if m == 1 {
                3_f64.sqrt()
            } else {
                ((2 * m + 1) as f64 / (2 * m) as f64).sqrt()
            };
            derivative = [
                scale * (dx * sector[0] + x * derivative[0] - dy * sector[1] - y * derivative[1]),
                scale * (dy * sector[0] + y * derivative[0] + dx * sector[1] + x * derivative[1]),
            ];
            sector = [
                scale * (x * sector[0] - y * sector[1]),
                scale * (y * sector[0] + x * sector[1]),
            ];
        }
        let mut previous = sector;
        let mut previous2 = [0.; 2];
        let mut dprevious = derivative;
        let mut dprevious2 = [0.; 2];
        for l in m..=degree {
            let (value, dvalue) = if l == m {
                (sector, derivative)
            } else {
                let (a, b) = if l == m + 1 {
                    (((2 * m + 3) as f64).sqrt(), 0.)
                } else {
                    let den = (l * l - m * m) as f64;
                    (
                        ((4 * l * l - 1) as f64 / den).sqrt(),
                        (((2 * l + 1) * ((l - 1) * (l - 1) - m * m)) as f64
                            / ((2 * l - 3) as f64 * den))
                            .sqrt(),
                    )
                };
                (
                    std::array::from_fn(|k| a * z * previous[k] - b * previous2[k]),
                    std::array::from_fn(|k| {
                        a * (dz * previous[k] + z * dprevious[k]) - b * dprevious2[k]
                    }),
                )
            };
            out[index(l, m as isize)] = dvalue[0];
            if m > 0 {
                out[index(l, -(m as isize))] = dvalue[1];
            }
            previous2 = previous;
            previous = value;
            dprevious2 = dprevious;
            dprevious = dvalue;
        }
    }
    Ok(out)
}

/// Normalized associated-Legendre recurrence, avoiding monomial cancellation.
pub fn harmonics(degree: usize, xyz: [f64; 3]) -> Result<Vec<f64>> {
    require(
        degree <= MAX_DEGREE
            && xyz.iter().all(|v| v.is_finite())
            && (xyz.iter().map(|v| v * v).sum::<f64>() - 1.).abs() < 1e-10,
        "invalid harmonic point",
    )?;
    let [x, y, z] = xyz;
    let r = x.hypot(y);
    let phi = y.atan2(x);
    let mut out = vec![0.; (degree + 1).pow(2)];
    let mut sector = 1.;
    for m in 0..=degree {
        if m > 0 {
            sector *= r * if m == 1 {
                3_f64.sqrt()
            } else {
                ((2 * m + 1) as f64 / (2 * m) as f64).sqrt()
            };
        }
        let (sin, cos) = (m as f64 * phi).sin_cos();
        let mut prev = sector;
        let mut prev2 = 0.;
        for l in m..=degree {
            let value = if l == m {
                sector
            } else if l == m + 1 {
                ((2 * m + 3) as f64).sqrt() * z * prev
            } else {
                let den = (l * l - m * m) as f64;
                (((4 * l * l - 1) as f64 / den).sqrt() * z) * prev
                    - (((2 * l + 1) * ((l - 1) * (l - 1) - m * m)) as f64
                        / ((2 * l - 3) as f64 * den))
                        .sqrt()
                        * prev2
            };
            out[index(l, m as isize)] = value * cos / PI;
            if m > 0 {
                out[index(l, -(m as isize))] = value * sin / PI;
            }
            prev2 = prev;
            prev = value;
        }
    }
    Ok(out)
}
pub fn index(l: usize, m: isize) -> usize {
    (l * l + l)
        .checked_add_signed(m)
        .expect("invalid harmonic index")
}
pub fn powers(l: usize, m: isize) -> (usize, usize, usize) {
    let mu = (l as isize - m) as usize;
    let nu = (l as isize + m) as usize;
    (mu / 2, nu / 2, nu % 2)
}
pub(crate) fn terms(degree: usize) -> Vec<(usize, usize, usize)> {
    (0..=degree)
        .flat_map(|l| (-(l as isize)..=l as isize).map(move |m| powers(l, m)))
        .collect()
}
pub(crate) fn pindex(a: usize, b: usize, c: usize) -> usize {
    index(a + b + c, b as isize - a as isize)
}

/// Canonical polynomial multiplication on the unit sphere: z² = 1-x²-y².
pub(crate) fn multiply(p: &[f64], pd: usize, q: &[f64], qd: usize) -> Vec<f64> {
    let mut r = vec![0.; (pd + qd + 1).pow(2)];
    let pt = terms(pd);
    let qt = terms(qd);
    for (i, &pv) in p.iter().enumerate() {
        if pv == 0. {
            continue;
        }
        let (a, b, c) = pt[i];
        for (j, &qv) in q.iter().enumerate() {
            if qv == 0. {
                continue;
            }
            let (d, e, f) = qt[j];
            let v = pv * qv;
            if c + f == 2 {
                r[pindex(a + d, b + e, 0)] += v;
                r[pindex(a + d + 2, b + e, 0)] -= v;
                r[pindex(a + d, b + e + 2, 0)] -= v;
            } else {
                r[pindex(a + d, b + e, c + f)] += v;
            }
        }
    }
    r
}

/// Change of basis with the same normalization, phase, and ordering as starry.
pub fn a1(degree: usize) -> Result<Matrix> {
    require(degree <= MAX_DEGREE, "degree exceeds supported limit (32)")?;
    let n = (degree + 1).pow(2);
    let mut out = Matrix::zeros(n, n);
    // Associated Legendre polynomial divided by (1-z²)^(m/2).
    let z = vec![0., 0., 1., 0.];
    let mut fac = 1.;
    for m in 0..=degree {
        let mut prev2 = vec![0.; n];
        let mut prev = vec![0.; n];
        prev[0] = fac;
        // Re/Im of (x+i y)^m, without requiring complex arithmetic.
        let mut re = vec![0.; (m + 1).pow(2)];
        let mut im = re.clone();
        let mut choose = 1.;
        for j in 0..=m {
            let sign = if j % 4 < 2 { 1. } else { -1. };
            if j % 2 == 0 {
                re[pindex(m - j, j, 0)] = sign * choose;
            } else {
                im[pindex(m - j, j, 0)] = sign * choose;
            }
            choose *= (m - j) as f64 / (j + 1) as f64;
        }
        for l in m..=degree {
            let current = if l == m {
                prev.clone()
            } else {
                let zp = multiply(&prev[..l * l], l - 1, &z, 1);
                let mut v = vec![0.; n];
                for k in 0..(l + 1).pow(2) {
                    v[k] = (2 * l - 1) as f64 * zp[k] / (l - m) as f64;
                }
                if l > m + 1 {
                    for k in 0..n {
                        v[k] -= (l + m - 1) as f64 * prev2[k] / (l - m) as f64;
                    }
                }
                v
            };
            let mut amp = ((2 * l + 1) as f64).sqrt() / PI;
            if m > 0 {
                amp *= 2_f64.sqrt();
                for k in 1..=m {
                    amp /= -(((l + k) * (l - k + 1)) as f64).sqrt();
                }
            }
            // current has degree l-m; multiplication fits degree l.
            let d = l - m;
            let pos = multiply(&current[..(d + 1).pow(2)], d, &re, m);
            let neg = multiply(&current[..(d + 1).pow(2)], d, &im, m);
            for k in 0..pos.len() {
                out[(k, index(l, m as isize))] = amp * pos[k];
                if m > 0 {
                    out[(k, index(l, -(m as isize)))] = amp * neg[k];
                }
            }
            prev2 = prev;
            prev = current;
        }
        fac *= -((2 * m + 1) as f64);
    }
    Ok(out)
}

/// Inverse Green change of basis, directly translated from computeA2.
pub fn a2_inverse(degree: usize) -> Result<Matrix> {
    require(degree <= MAX_DEGREE, "degree exceeds supported limit (32)")?;
    let mut a = Matrix::zeros((degree + 1).pow(2), (degree + 1).pow(2));
    for l in 0..=degree {
        for m in -(l as isize)..=l as isize {
            let n = index(l, m);
            let mu = (l as isize - m) as usize;
            let nu = (l as isize + m) as usize;
            if nu.is_multiple_of(2) {
                a[(n, n)] = ((mu + 2) / 2) as f64;
            } else if l == 1 && m == 0 {
                a[(n, n)] = 1.;
            } else if mu == 1 && l % 2 == 0 {
                a[(l * l + 3, n)] = 3.;
            } else if mu == 1 && l % 2 == 1 {
                a[(1 + (l - 2).pow(2), n)] = -1.;
                a[(l * l + 1, n)] = 1.;
                a[(l * l + 5, n)] = 4.;
            } else {
                if mu != 3 {
                    a[(nu + (mu + nu - 4).pow(2) / 4, n)] = ((mu - 3) / 2) as f64;
                    a[(nu + 4 + (mu + nu).pow(2) / 4, n)] = -(((mu - 3) / 2) as f64);
                }
                a[(nu + (mu + nu).pow(2) / 4, n)] = -(((mu + 3) / 2) as f64);
            }
        }
    }
    Ok(a)
}

/// Exact full-disk moments. Recurrences equivalent to upstream computerT.
pub fn disk_moments(degree: usize) -> Result<Vec<f64>> {
    require(degree <= MAX_DEGREE, "degree exceeds supported limit (32)")?;
    Ok(terms(degree)
        .into_iter()
        .map(|(a, b, c)| {
            if a % 2 == 1 || b % 2 == 1 {
                return 0.;
            }
            let mut v = if c == 0 { PI } else { 2. * PI / 3. };
            for j in 0..a / 2 {
                v *= (j as f64 + 0.5) / (j as f64 + 2. + c as f64 / 2.);
            }
            for j in 0..b / 2 {
                v *= (j as f64 + 0.5) / ((a / 2 + j) as f64 + 2. + c as f64 / 2.);
            }
            v
        })
        .collect())
}

pub fn polynomial(degree: usize, xyz: [f64; 3]) -> Result<Vec<f64>> {
    require(
        degree <= MAX_DEGREE && xyz.iter().all(|v| v.is_finite()),
        "invalid polynomial arguments",
    )?;
    Ok(terms(degree)
        .iter()
        .map(|&(a, b, c)| xyz[0].powi(a as i32) * xyz[1].powi(b as i32) * xyz[2].powi(c as i32))
        .collect())
}

/// Cartesian derivatives of the canonical polynomial extension; finite at axes.
pub fn polynomial_gradient(degree: usize, xyz: [f64; 3]) -> Result<[Vec<f64>; 3]> {
    polynomial(degree, xyz)?;
    let ts = terms(degree);
    let mut out = std::array::from_fn(|_| vec![0.; ts.len()]);
    for (i, &(a, b, c)) in ts.iter().enumerate() {
        for k in 0..3 {
            let mut p = [a, b, c];
            let f = p[k];
            if f > 0 {
                p[k] -= 1;
                out[k][i] = f as f64 * (0..3).map(|j| xyz[j].powi(p[j] as i32)).product::<f64>();
            }
        }
    }
    Ok(out)
}
