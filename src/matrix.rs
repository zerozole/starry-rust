//! Small dense matrices. No native-library or registry dependency.
use crate::{Result, require};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl Matrix {
    /// Truncated SVD by one-sided Jacobi rotations. Returns U*S and V^T.
    /// This keeps image-cube decomposition in Rust without a BLAS dependency.
    pub fn low_rank(&self, rank: usize) -> Result<(Self, Self)> {
        require(
            self.rows > 0
                && self.cols > 0
                && rank > 0
                && rank <= self.rows.min(self.cols)
                && self.data.iter().all(|x| x.is_finite()),
            "invalid low-rank matrix",
        )?;
        let transposed = self.rows < self.cols;
        let (m, n) = if transposed {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        let scale = self
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0., f64::max)
            .max(f64::MIN_POSITIVE);
        let mut b = Self::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                b[(i, j)] = if transposed {
                    self[(j, i)]
                } else {
                    self[(i, j)]
                } / scale;
            }
        }
        let mut v = Self::identity(n);
        let mut converged = false;
        for _ in 0..100 {
            let mut rotated = false;
            for p in 0..n {
                for q in p + 1..n {
                    let pp: f64 = (0..m).map(|i| b[(i, p)].powi(2)).sum();
                    let qq: f64 = (0..m).map(|i| b[(i, q)].powi(2)).sum();
                    let pq: f64 = (0..m).map(|i| b[(i, p)] * b[(i, q)]).sum();
                    if pp.min(qq) < 1e-28 || pq.abs() <= 2e-14 * (pp * qq).sqrt() {
                        continue;
                    }
                    let tau = (qq - pp) / (2. * pq);
                    let t = if tau >= 0. { 1. } else { -1. } / (tau.abs() + tau.hypot(1.));
                    let c = 1. / (1. + t * t).sqrt();
                    let s = c * t;
                    for i in 0..m {
                        let a = b[(i, p)];
                        let z = b[(i, q)];
                        b[(i, p)] = c * a - s * z;
                        b[(i, q)] = s * a + c * z;
                    }
                    for i in 0..n {
                        let a = v[(i, p)];
                        let z = v[(i, q)];
                        v[(i, p)] = c * a - s * z;
                        v[(i, q)] = s * a + c * z;
                    }
                    rotated = true;
                }
            }
            if !rotated {
                converged = true;
                break;
            }
        }
        require(converged, "Jacobi SVD did not converge")?;
        let norms: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| b[(i, j)].powi(2)).sum::<f64>().sqrt())
            .collect();
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]));
        let mut scores = Self::zeros(self.rows, rank);
        let mut spectra = Self::zeros(rank, self.cols);
        for (k, &j) in order.iter().take(rank).enumerate() {
            if transposed {
                for i in 0..self.rows {
                    scores[(i, k)] = v[(i, j)] * norms[j] * scale;
                }
                for i in 0..self.cols {
                    spectra[(k, i)] = if norms[j] > 0. {
                        b[(i, j)] / norms[j]
                    } else {
                        0.
                    };
                }
            } else {
                for i in 0..self.rows {
                    scores[(i, k)] = b[(i, j)] * scale;
                }
                for i in 0..self.cols {
                    spectra[(k, i)] = v[(i, j)];
                }
            }
        }
        Ok((scores, spectra))
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.; rows * cols],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut a = Self::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 1.;
        }
        a
    }
    pub fn dot(&self, x: &[f64]) -> Result<Vec<f64>> {
        require(x.len() == self.cols, "matrix/vector shape mismatch")?;
        Ok(self
            .data
            .chunks(self.cols)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect())
    }
    pub fn left_dot(&self, x: &[f64]) -> Result<Vec<f64>> {
        require(x.len() == self.rows, "vector/matrix shape mismatch")?;
        let mut y = vec![0.; self.cols];
        for (i, &v) in x.iter().enumerate() {
            for j in 0..self.cols {
                y[j] += v * self[(i, j)];
            }
        }
        Ok(y)
    }
    /// Solve A X = B using partial-pivot Gaussian elimination.
    pub fn solve(&self, b: &Self) -> Result<Self> {
        require(
            self.rows == self.cols && b.rows == self.rows,
            "invalid solve dimensions",
        )?;
        let n = self.rows;
        let mut a = self.clone();
        let mut x = b.clone();
        for k in 0..n {
            let p = (k..n)
                .max_by(|&i, &j| a[(i, k)].abs().total_cmp(&a[(j, k)].abs()))
                .unwrap();
            require(
                a[(p, k)].is_finite() && a[(p, k)].abs() > f64::MIN_POSITIVE,
                "singular/nonfinite matrix",
            )?;
            for j in 0..n {
                a.data.swap(k * n + j, p * n + j);
            }
            for j in 0..b.cols {
                x.data.swap(k * b.cols + j, p * b.cols + j);
            }
            for i in k + 1..n {
                let f = a[(i, k)] / a[(k, k)];
                a[(i, k)] = 0.;
                for j in k + 1..n {
                    a[(i, j)] -= f * a[(k, j)];
                }
                for j in 0..b.cols {
                    x[(i, j)] -= f * x[(k, j)];
                }
            }
        }
        for i in (0..n).rev() {
            for j in 0..b.cols {
                for k in i + 1..n {
                    x[(i, j)] -= a[(i, k)] * x[(k, j)];
                }
                x[(i, j)] /= a[(i, i)];
            }
        }
        require(
            x.data.iter().all(|v| v.is_finite()),
            "nonfinite solve result",
        )?;
        Ok(x)
    }
    pub fn inverse(&self) -> Result<Self> {
        self.solve(&Self::identity(self.rows))
    }
}
impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}
