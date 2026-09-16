//! Small second-order forward-mode jets for native parameter derivatives.
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug)]
pub struct Jet<const N: usize> {
    pub value: f64,
    pub gradient: [f64; N],
    pub hessian: [[f64; N]; N],
}
impl<const N: usize> Jet<N> {
    pub fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.; N],
            hessian: [[0.; N]; N],
        }
    }
    pub fn variable(value: f64, index: usize) -> Self {
        let mut x = Self::constant(value);
        x.gradient[index] = 1.;
        x
    }
    fn unary(self, value: f64, first: f64, second: f64) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|g| first * g),
            hessian: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    first * self.hessian[i][j] + second * self.gradient[i] * self.gradient[j]
                })
            }),
        }
    }
    pub fn sin(self) -> Self {
        self.unary(self.value.sin(), self.value.cos(), -self.value.sin())
    }
    pub fn cos(self) -> Self {
        self.unary(self.value.cos(), -self.value.sin(), -self.value.cos())
    }
    pub fn sqrt(self) -> Self {
        let v = self.value.sqrt();
        self.unary(v, 0.5 / v, -0.25 / (v * self.value))
    }
    pub fn powi(self, n: usize) -> Self {
        let mut result = Self::constant(1.);
        for _ in 0..n {
            result = result * self;
        }
        result
    }
    pub fn reciprocal(self) -> Self {
        self.unary(
            1. / self.value,
            -1. / self.value.powi(2),
            2. / self.value.powi(3),
        )
    }
    pub fn ln(self) -> Self {
        self.unary(self.value.ln(), 1. / self.value, -1. / self.value.powi(2))
    }
    pub fn exp(self) -> Self {
        let e = self.value.exp();
        self.unary(e, e, e)
    }
    pub fn exp_m1(self) -> Self {
        let e = self.value.exp();
        self.unary(self.value.exp_m1(), e, e)
    }
    pub fn pow(self, exponent: Self) -> Self {
        (exponent * self.ln()).exp()
    }
    pub fn atan2(self, x: Self) -> Self {
        let y = self;
        let r2 = x.value * x.value + y.value * y.value;
        let fy = x.value / r2;
        let fx = -y.value / r2;
        let fyy = -2. * x.value * y.value / r2.powi(2);
        let fxx = -fyy;
        let fxy = (y.value * y.value - x.value * x.value) / r2.powi(2);
        Self {
            value: y.value.atan2(x.value),
            gradient: std::array::from_fn(|i| fy * y.gradient[i] + fx * x.gradient[i]),
            hessian: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    fy * y.hessian[i][j]
                        + fx * x.hessian[i][j]
                        + fyy * y.gradient[i] * y.gradient[j]
                        + fxx * x.gradient[i] * x.gradient[j]
                        + fxy * (y.gradient[i] * x.gradient[j] + x.gradient[i] * y.gradient[j])
                })
            }),
        }
    }
}
impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            value: self.value + b.value,
            gradient: std::array::from_fn(|i| self.gradient[i] + b.gradient[i]),
            hessian: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.hessian[i][j] + b.hessian[i][j])
            }),
        }
    }
}
impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            gradient: self.gradient.map(|g| -g),
            hessian: self.hessian.map(|row| row.map(|v| -v)),
        }
    }
}
impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}
// Addition is required by the first/second derivative product rules.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            value: self.value * b.value,
            gradient: std::array::from_fn(|i| {
                self.gradient[i] * b.value + self.value * b.gradient[i]
            }),
            hessian: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    self.hessian[i][j] * b.value
                        + self.value * b.hessian[i][j]
                        + self.gradient[i] * b.gradient[j]
                        + b.gradient[i] * self.gradient[j]
                })
            }),
        }
    }
}
impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        self.mul(b.reciprocal())
    }
}
