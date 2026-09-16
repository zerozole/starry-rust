//! Gaussian linear map inference with full covariance matrices.
use crate::{Result, matrix::Matrix, require};
use std::f64::consts::PI;

fn cholesky(a: &Matrix) -> Result<Matrix> {
    require(
        a.rows == a.cols && a.data.iter().all(|v| v.is_finite()),
        "invalid covariance",
    )?;
    let n = a.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            require(
                (a[(i, j)] - a[(j, i)]).abs() <= 1e-12 * (1. + a[(i, j)].abs()),
                "covariance is not symmetric",
            )?;
            let mut value = a[(i, j)];
            for k in 0..j {
                value -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                require(value > 0., "covariance is not positive definite")?;
                l[(i, j)] = value.sqrt();
            } else {
                l[(i, j)] = value / l[(j, j)];
            }
        }
    }
    Ok(l)
}
fn covariance_solve(l: &Matrix, b: &Matrix) -> Matrix {
    let mut x = b.clone();
    let n = l.rows;
    for j in 0..b.cols {
        for i in 0..n {
            for k in 0..i {
                x[(i, j)] -= l[(i, k)] * x[(k, j)];
            }
            x[(i, j)] /= l[(i, i)];
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                x[(i, j)] -= l[(k, i)] * x[(k, j)];
            }
            x[(i, j)] /= l[(i, i)];
        }
    }
    x
}

#[derive(Clone, Debug)]
pub struct GaussianPrior {
    pub mean: Vec<f64>,
    pub covariance: Matrix,
}
#[derive(Clone, Debug)]
pub struct Solution {
    pub mean: Vec<f64>,
    pub covariance: Matrix,
    pub log_likelihood: f64,
}

#[derive(Clone, Debug)]
pub struct SparseSolution {
    pub weights: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Port of OpsDoppler.L1: iteratively reweighted ridge regression.
/// Minimizes .5*w^T*normal*w - rhs^T*w + lambda*sum(abs(w)), with
/// upstream's epsilon floor and squared-step stopping criterion.
pub fn l1(
    normal: &Matrix,
    rhs: &[f64],
    lambda: f64,
    maximum_iterations: usize,
    epsilon: f64,
    tolerance: f64,
) -> Result<SparseSolution> {
    let n = normal.rows;
    require(
        n > 0
            && normal.cols == n
            && rhs.len() == n
            && normal.data.iter().chain(rhs).all(|v| v.is_finite())
            && lambda.is_finite()
            && lambda >= 0.
            && epsilon.is_finite()
            && epsilon > 0.
            && tolerance.is_finite()
            && tolerance > 0.
            && maximum_iterations > 0,
        "invalid L1 inputs",
    )?;
    let mut weights = vec![1_f64; n];
    let data = Matrix {
        rows: n,
        cols: 1,
        data: rhs.to_vec(),
    };
    for iteration in 1..=maximum_iterations {
        let mut precision = normal.clone();
        for i in 0..n {
            precision[(i, i)] += if lambda == 0. {
                0.
            } else {
                lambda / weights[i].abs().max(lambda * epsilon)
            };
        }
        let next = covariance_solve(&cholesky(&precision)?, &data).data;
        let change: f64 = next
            .iter()
            .zip(&weights)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        weights = next;
        if change < tolerance {
            return Ok(SparseSolution {
                weights,
                iterations: iteration,
                converged: true,
            });
        }
    }
    Ok(SparseSolution {
        weights,
        iterations: maximum_iterations,
        converged: false,
    })
}
impl Solution {
    /// Deterministic transform of caller-supplied independent N(0,1) draws.
    pub fn draw(&self, standard_normals: &[f64]) -> Result<Vec<f64>> {
        require(
            standard_normals.len() == self.mean.len()
                && standard_normals.iter().all(|v| v.is_finite()),
            "invalid normal draw",
        )?;
        let l = cholesky(&self.covariance)?;
        let offset = l.dot(standard_normals)?;
        Ok(self.mean.iter().zip(offset).map(|(a, b)| a + b).collect())
    }
}
pub fn solve(
    design: &Matrix,
    data: &[f64],
    noise: &Matrix,
    prior: Option<&GaussianPrior>,
) -> Result<Solution> {
    let n = design.rows;
    let p = design.cols;
    require(
        n > 0
            && p > 0
            && data.len() == n
            && noise.rows == n
            && noise.cols == n
            && data.iter().all(|v| v.is_finite())
            && design.data.iter().all(|v| v.is_finite()),
        "invalid linear inference inputs",
    )?;
    let chol = cholesky(noise)?;
    let cx = covariance_solve(&chol, design);
    let y = Matrix {
        rows: n,
        cols: 1,
        data: data.to_vec(),
    };
    let cy = covariance_solve(&chol, &y);
    let mut precision = Matrix::zeros(p, p);
    let mut rhs = Matrix::zeros(p, 1);
    for i in 0..p {
        for k in 0..n {
            rhs[(i, 0)] += design[(k, i)] * cy[(k, 0)];
            for j in 0..p {
                precision[(i, j)] += design[(k, i)] * cx[(k, j)];
            }
        }
    }
    if let Some(prior) = prior {
        require(
            prior.mean.len() == p
                && prior.covariance.rows == p
                && prior.covariance.cols == p
                && prior.mean.iter().all(|v| v.is_finite()),
            "invalid Gaussian prior",
        )?;
        let c = cholesky(&prior.covariance)?;
        let inv = covariance_solve(&c, &Matrix::identity(p));
        let pm = inv.dot(&prior.mean)?;
        for i in 0..p {
            rhs[(i, 0)] += pm[i];
            for j in 0..p {
                precision[(i, j)] += inv[(i, j)];
            }
        }
    }
    // This matrix is analytically symmetric. Independent triangular sums can
    // differ at roundoff near a null space; preserve symmetry before factoring.
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (precision[(i, j)] + precision[(j, i)]);
            precision[(i, j)] = v;
            precision[(j, i)] = v;
        }
    }
    let cp = cholesky(&precision)?;
    let mean = covariance_solve(&cp, &rhs).data;
    let covariance = covariance_solve(&cp, &Matrix::identity(p));
    let predicted = design.dot(&mean)?;
    let residual = Matrix {
        rows: n,
        cols: 1,
        data: data.iter().zip(predicted).map(|(a, b)| a - b).collect(),
    };
    let cr = covariance_solve(&chol, &residual);
    let chi = crate::map::dot(&residual.data, &cr.data);
    let logdet = 2. * (0..n).map(|i| chol[(i, i)].ln()).sum::<f64>();
    Ok(Solution {
        mean,
        covariance,
        log_likelihood: -0.5 * (chi + logdet + n as f64 * (2. * PI).ln()),
    })
}

/// Marginalized Gaussian log likelihood after integrating the map prior.
pub fn marginal_log_likelihood(
    design: &Matrix,
    data: &[f64],
    noise: &Matrix,
    prior: &GaussianPrior,
) -> Result<f64> {
    let p = design.cols;
    let solution = solve(design, data, noise, Some(prior))?;
    let prior_chol = cholesky(&prior.covariance)?;
    let posterior_chol = cholesky(&solution.covariance)?;
    let delta = Matrix {
        rows: p,
        cols: 1,
        data: solution
            .mean
            .iter()
            .zip(&prior.mean)
            .map(|(a, b)| a - b)
            .collect(),
    };
    let weighted = covariance_solve(&prior_chol, &delta);
    // Gaussian normalization identity evaluated at the posterior mean. Avoid
    // constructing C+X L X^T, which loses tiny noise terms for diffuse priors.
    Ok(solution.log_likelihood
        - 0.5
            * (crate::map::dot(&delta.data, &weighted.data)
                + 2. * (0..p)
                    .map(|i| prior_chol[(i, i)].ln() - posterior_chol[(i, i)].ln())
                    .sum::<f64>()))
}

/// Diagonal-noise Gaussian regression without constructing an observations by
/// observations matrix. Returns the posterior and optional marginalized lnL.
pub fn solve_diagonal(
    design: &Matrix,
    data: &[f64],
    variance: &[f64],
    prior: Option<&GaussianPrior>,
) -> Result<(Solution, Option<f64>)> {
    let n = design.rows;
    let p = design.cols;
    require(
        n > 0
            && p > 0
            && data.len() == n
            && variance.len() == n
            && design.data.len() == n * p
            && design.data.iter().all(|v| v.is_finite())
            && data.iter().all(|v| v.is_finite())
            && variance.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid diagonal inference inputs",
    )?;
    let mut precision = Matrix::zeros(p, p);
    let mut rhs = Matrix::zeros(p, 1);
    for k in 0..n {
        let weight = 1. / variance[k];
        for i in 0..p {
            rhs[(i, 0)] += design[(k, i)] * weight * data[k];
            for j in 0..=i {
                precision[(i, j)] += design[(k, i)] * weight * design[(k, j)];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            precision[(j, i)] = precision[(i, j)];
        }
    }
    let mut inverse_prior = None;
    let mut prior_logdet = 0.;
    if let Some(prior) = prior {
        require(
            prior.mean.len() == p
                && prior.mean.iter().all(|v| v.is_finite())
                && prior.covariance.rows == p
                && prior.covariance.cols == p,
            "invalid Gaussian prior",
        )?;
        let chol = cholesky(&prior.covariance)?;
        prior_logdet = 2. * (0..p).map(|i| chol[(i, i)].ln()).sum::<f64>();
        let inverse = covariance_solve(&chol, &Matrix::identity(p));
        let mean = inverse.dot(&prior.mean)?;
        for i in 0..p {
            rhs[(i, 0)] += mean[i];
            for j in 0..p {
                precision[(i, j)] += inverse[(i, j)];
            }
        }
        inverse_prior = Some(inverse);
    }
    let chol = cholesky(&precision)?;
    let mean = covariance_solve(&chol, &rhs).data;
    let covariance = covariance_solve(&chol, &Matrix::identity(p));
    let predicted = design.dot(&mean)?;
    let log_likelihood = -0.5
        * (0..n)
            .map(|i| (data[i] - predicted[i]).powi(2) / variance[i] + (2. * PI * variance[i]).ln())
            .sum::<f64>();
    let marginal = if let (Some(prior), Some(inverse)) = (prior, inverse_prior) {
        let delta: Vec<_> = mean.iter().zip(&prior.mean).map(|(a, b)| a - b).collect();
        let penalty = crate::map::dot(&delta, &inverse.dot(&delta)?);
        Some(
            log_likelihood
                - 0.5
                    * (penalty
                        + prior_logdet
                        + 2. * (0..p).map(|i| chol[(i, i)].ln()).sum::<f64>()),
        )
    } else {
        None
    };
    Ok((
        Solution {
            mean,
            covariance,
            log_likelihood,
        },
        marginal,
    ))
}
