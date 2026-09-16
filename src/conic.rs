//! Projected elliptical silhouettes and analytic conic intersection cuts.
use crate::{Result, require};
#[derive(Clone, Copy, Debug)]
pub struct Ellipse {
    pub x: f64,
    pub y: f64,
    pub major: f64,
    pub minor: f64,
    pub angle: f64,
}
impl Ellipse {
    pub fn validate(&self) -> Result<()> {
        require(
            [self.x, self.y, self.major, self.minor, self.angle]
                .iter()
                .all(|v| v.is_finite())
                && self.major > 0.
                && self.minor > 0.,
            "invalid projected ellipse",
        )
    }
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let (s, c) = self.angle.sin_cos();
        let dx = x - self.x;
        let dy = y - self.y;
        ((c * dx + s * dy) / self.major).powi(2) + ((-s * dx + c * dy) / self.minor).powi(2) < 1.
    }
    pub fn conic(&self) -> [f64; 6] {
        let (s, c) = self.angle.sin_cos();
        let a = 1. / self.major.powi(2);
        let b = 1. / self.minor.powi(2);
        let xx = c * c * a + s * s * b;
        let xy = 2. * s * c * (a - b);
        let yy = s * s * a + c * c * b;
        [
            xx,
            xy,
            yy,
            -2. * xx * self.x - xy * self.y,
            -xy * self.x - 2. * yy * self.y,
            xx * self.x * self.x + xy * self.x * self.y + yy * self.y * self.y - 1.,
        ]
    }
    pub fn x_extent(&self) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let h = (self.major * c).hypot(self.minor * s);
        [self.x - h, self.x + h]
    }
    pub fn y_bounds(&self, x: f64) -> Option<(f64, f64)> {
        let p = self.conic();
        let b = p[1] * x + p[4];
        let c = p[0] * x * x + p[3] * x + p[5];
        let discriminant = b * b - 4. * p[2] * c;
        if discriminant <= 0. {
            None
        } else {
            let half = discriminant.sqrt() / (2. * p[2]);
            let center = -b / (2. * p[2]);
            Some((center - half, center + half))
        }
    }
    /// Rotate the entire ellipse actively about the origin.
    pub fn rotated(&self, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
            angle: self.angle + angle,
            ..*self
        }
    }
}
fn product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut p = vec![0.; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            p[i + j] += x * y;
        }
    }
    p
}
fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut p = vec![0.; a.len().max(b.len())];
    for (i, x) in a.iter().enumerate() {
        p[i] += x;
    }
    for (i, x) in b.iter().enumerate() {
        p[i] -= x;
    }
    p
}
pub(crate) fn intersections(first: Ellipse, second: Ellipse) -> Vec<f64> {
    let p = first.conic();
    let q = second.conic();
    // Resultant in y: (a*f-d*c)^2 - (a*e-d*b)*(b*f-e*c).
    let a = [p[2]];
    let b = [p[4], p[1]];
    let c = [p[5], p[3], p[0]];
    let d = [q[2]];
    let e = [q[4], q[1]];
    let f = [q[5], q[3], q[0]];
    let afdc = subtract(&product(&a, &f), &product(&d, &c));
    let aedb = subtract(&product(&a, &e), &product(&d, &b));
    let bfec = subtract(&product(&b, &f), &product(&e, &c));
    let polynomial = subtract(&product(&afdc, &afdc), &product(&aedb, &bfec));
    crate::polynomial_roots::real_roots(&polynomial, -1., 1.)
}
