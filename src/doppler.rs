//! Native spectral line broadening and Doppler imaging convolution kernels.
//! Chord integral recurrences are ported from core.py OpsDoppler.get_rT.
use crate::{
    Map, Result, basis,
    inference::{self, GaussianPrior, Solution},
    matrix::Matrix,
    require,
};
use std::f64::consts::PI;
pub const SPEED_OF_LIGHT: f64 = 299792458.;
pub fn chord_moments(degree: usize, x: f64) -> Result<Vec<f64>> {
    require(
        degree <= basis::MAX_DEGREE && x.is_finite(),
        "invalid Doppler chord",
    )?;
    let r2 = (1. - x * x).max(0.);
    let r = r2.sqrt();
    let mut even = vec![0.; degree + 1];
    let mut odd = even.clone();
    even[0] = 2. * r;
    odd[0] = PI * r2 / 2.;
    for j in (2..=degree).step_by(2) {
        even[j] = (j - 1) as f64 / (j + 1) as f64 * r2 * even[j - 2];
        odd[j] = (j - 1) as f64 / (j + 2) as f64 * r2 * odd[j - 2];
    }
    Ok(basis::terms(degree)
        .into_iter()
        .map(|(i, j, k)| x.powi(i as i32) * if k == 0 { even[j] } else { odd[j] })
        .collect())
}

pub fn kernel(
    map: &Map,
    inclination: f64,
    phase: f64,
    veq: f64,
    u: &[f64],
    log_spacing: f64,
    half_width: usize,
) -> Result<Vec<f64>> {
    require(
        veq.is_finite()
            && (0. ..SPEED_OF_LIGHT).contains(&veq)
            && log_spacing.is_finite()
            && log_spacing > 0.
            && half_width > 0
            && half_width <= 100_000,
        "invalid Doppler grid",
    )?;
    let map = map.projected(inclination, 0., phase)?;
    let d = map.degree() + u.len();
    require(d <= basis::MAX_DEGREE, "combined Doppler degree exceeds 32")?;
    let vsini = (veq * inclination.sin()).max(1.);
    let (limb, normalization) = crate::map::limb_polynomial(u)?;
    let p = basis::multiply(
        &map.polynomial_coefficients()?,
        map.degree(),
        &limb,
        u.len(),
    );
    let mut values = vec![];
    let mut base = 0.;
    let chord_nodes = if d > 12 {
        crate::occultation::gauss(64)
    } else {
        vec![]
    };
    for j in 0..=2 * half_width {
        let offset = (j as f64 - half_width as f64) * log_spacing;
        let x = -SPEED_OF_LIGHT * offset.tanh() / vsini;
        let moments = chord_moments(d, x)?;
        base += moments[0] / PI;
        if d > 12 {
            let radius = (1. - x * x).max(0.).sqrt();
            let mut integral = 0.;
            if radius > 0. {
                for &(node, weight) in &chord_nodes {
                    let angle = PI / 2. * node;
                    let xyz = [x, radius * angle.sin(), radius * angle.cos()];
                    integral += weight * PI / 2.
                        * radius
                        * angle.cos()
                        * map.intensity_xyz(xyz)?
                        * crate::map::limb_intensity(xyz[2], u)?
                        / normalization;
                }
            }
            values.push(integral);
        } else {
            values.push(map.amplitude() * crate::map::dot(&p, &moments) / normalization);
        }
    }
    require(base > 0., "Doppler kernel has empty support")?;
    for v in &mut values {
        *v /= base;
    }
    Ok(values)
}

#[derive(Clone, Debug)]
pub struct Component {
    pub map: Map,
    pub spectrum: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct DopplerMap {
    pub components: Vec<Component>,
    pub log_spacing: f64,
    pub half_width: usize,
    pub inclination: f64,
    pub veq: f64,
    pub limb_darkening: Vec<f64>,
}
impl DopplerMap {
    /// Apply the unrestricted spectral-map operator or its exact transpose.
    /// Each harmonic has its own padded rest spectrum (harmonic-major order).
    /// Component coefficients and spectra are not used; the first map supplies
    /// the degree. No dense epoch/wavelength by harmonic/wavelength matrix is built.
    pub fn spectral_map_dot(
        &self,
        phases: &[f64],
        values: &[f64],
        transpose: bool,
    ) -> Result<Vec<f64>> {
        let (n, nout) = self.dimensions(phases)?;
        let degree = self.components[0].map.degree();
        let ny = (degree + 1).pow(2);
        let rows = phases
            .len()
            .checked_mul(nout)
            .ok_or(crate::Error("Doppler shape overflow".into()))?;
        require(
            n.checked_mul(ny).is_some_and(|n| n <= 32_000_000) && rows <= 32_000_000,
            "spectral map too large",
        )?;
        let (input, output) = if transpose {
            (rows, n * ny)
        } else {
            (n * ny, rows)
        };
        require(
            values.len() == input && values.iter().all(|v| v.is_finite()),
            "invalid spectral map input",
        )?;
        let mut result = vec![0.; output];
        let mut map = Map::new(degree)?;
        for h in 0..ny {
            let mut y = vec![0.; ny];
            y[h] = 1.;
            map.set_coefficients(&y)?;
            for (t, &phase) in phases.iter().enumerate() {
                let k = kernel(
                    &map,
                    self.inclination,
                    phase,
                    self.veq,
                    &self.limb_darkening,
                    self.log_spacing,
                    self.half_width,
                )?;
                for i in 0..nout {
                    for (j, &weight) in k.iter().enumerate() {
                        let row = t * nout + i;
                        let col = h * n + i + j;
                        if transpose {
                            result[col] += weight * values[row];
                        } else {
                            result[row] += weight * values[col];
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    fn dimensions(&self, phases: &[f64]) -> Result<(usize, usize)> {
        require(
            !self.components.is_empty()
                && !phases.is_empty()
                && phases.iter().all(|v| v.is_finite()),
            "invalid Doppler components/phases",
        )?;
        let n = self.components[0].spectrum.len();
        require(
            self.half_width > 0 && self.half_width < n / 2,
            "invalid Doppler padding",
        )?;
        require(
            self.components
                .iter()
                .all(|c| c.spectrum.len() == n && c.spectrum.iter().all(|v| v.is_finite())),
            "invalid component spectra",
        )?;
        Ok((n, n - 2 * self.half_width))
    }

    /// Absolute map coefficients, concatenated by component. Amplitude is
    /// included in these weights; design matrices exclude amplitude.
    pub fn map_weights(&self) -> Vec<f64> {
        self.components
            .iter()
            .flat_map(|c| c.map.coefficients().iter().map(|y| y * c.map.amplitude()))
            .collect()
    }
    pub fn set_map_weights(&mut self, weights: &[f64]) -> Result<()> {
        require(
            weights.len() == self.map_weights().len() && weights.iter().all(|v| v.is_finite()),
            "invalid Doppler map weights",
        )?;
        let mut start = 0;
        for c in &mut self.components {
            let n = c.map.coefficients().len();
            let y = &weights[start..start + n];
            let amp = if y[0] == 0. { 1. } else { y[0] };
            c.map
                .set_coefficients(&y.iter().map(|v| v / amp).collect::<Vec<_>>())?;
            c.map.set_amplitude(amp)?;
            start += n;
        }
        Ok(())
    }

    /// Exact linear operator for fixed rest spectra, and the corresponding
    /// continuum operator. Rows are epoch-major, wavelength-minor.
    pub fn map_design(&self, phases: &[f64]) -> Result<(Matrix, Matrix)> {
        let (_, nout) = self.dimensions(phases)?;
        let cols = self.map_weights().len();
        require(
            phases
                .len()
                .checked_mul(nout)
                .and_then(|r| r.checked_mul(cols))
                .is_some_and(|n| n <= 32_000_000),
            "Doppler design is too large",
        )?;
        let mut design = Matrix::zeros(phases.len() * nout, cols);
        let mut baseline = Matrix::zeros(phases.len() * nout, cols);
        let mut column = 0;
        for c in &self.components {
            let n = c.map.coefficients().len();
            let mut map = Map::new(c.map.degree())?;
            for j in 0..n {
                let mut y = vec![0.; n];
                y[j] = 1.;
                map.set_coefficients(&y)?;
                for (t, &phase) in phases.iter().enumerate() {
                    let k = kernel(
                        &map,
                        self.inclination,
                        phase,
                        self.veq,
                        &self.limb_darkening,
                        self.log_spacing,
                        self.half_width,
                    )?;
                    let continuum = k.iter().sum::<f64>();
                    for i in 0..nout {
                        design[(t * nout + i, column)] = k
                            .iter()
                            .enumerate()
                            .map(|(j, w)| w * c.spectrum[i + j])
                            .sum();
                        baseline[(t * nout + i, column)] = continuum;
                    }
                }
                column += 1;
            }
        }
        Ok((design, baseline))
    }

    /// Exact linear operator for fixed component maps. Columns concatenate
    /// padded rest spectra by component. Normalization is independent of spectra.
    pub fn spectrum_design(&self, phases: &[f64], normalize: bool) -> Result<Matrix> {
        let (n, nout) = self.dimensions(phases)?;
        let cols = n * self.components.len();
        require(
            phases
                .len()
                .checked_mul(nout)
                .and_then(|r| r.checked_mul(cols))
                .is_some_and(|n| n <= 32_000_000),
            "Doppler design is too large",
        )?;
        let mut design = Matrix::zeros(phases.len() * nout, cols);
        for (t, &phase) in phases.iter().enumerate() {
            let mut baseline = 0.;
            for (c, component) in self.components.iter().enumerate() {
                let k = kernel(
                    &component.map,
                    self.inclination,
                    phase,
                    self.veq,
                    &self.limb_darkening,
                    self.log_spacing,
                    self.half_width,
                )?;
                baseline += k.iter().sum::<f64>();
                for i in 0..nout {
                    for (j, &v) in k.iter().enumerate() {
                        design[(t * nout + i, c * n + i + j)] = v;
                    }
                }
            }
            if normalize {
                require(
                    baseline.is_finite() && baseline != 0.,
                    "zero Doppler continuum",
                )?;
                for i in t * nout..(t + 1) * nout {
                    for j in 0..cols {
                        design[(i, j)] /= baseline;
                    }
                }
            }
        }
        Ok(design)
    }

    /// Flux and exact coefficient Jacobian of the continuum-normalized model.
    /// This accounts for derivatives of both numerator and denominator.
    pub fn map_jacobian(&self, phases: &[f64], normalize: bool) -> Result<(Vec<f64>, Matrix)> {
        let (mut design, baseline) = self.map_design(phases)?;
        let weights = self.map_weights();
        let mut flux = design.dot(&weights)?;
        if normalize {
            let denominator = baseline.dot(&weights)?;
            for i in 0..design.rows {
                require(
                    denominator[i].is_finite() && denominator[i] != 0.,
                    "zero Doppler continuum",
                )?;
                flux[i] /= denominator[i];
                for j in 0..design.cols {
                    design[(i, j)] = (design[(i, j)] - flux[i] * baseline[(i, j)]) / denominator[i];
                }
            }
        }
        Ok((flux, design))
    }

    pub fn solve_map(
        &mut self,
        phases: &[f64],
        data: &[f64],
        noise: &Matrix,
        prior: Option<&GaussianPrior>,
    ) -> Result<Solution> {
        let (design, _) = self.map_design(phases)?;
        let solution = inference::solve(&design, data, noise, prior)?;
        self.set_map_weights(&solution.mean)?;
        Ok(solution)
    }
    pub fn solve_spectrum(
        &mut self,
        phases: &[f64],
        data: &[f64],
        noise: &Matrix,
        prior: Option<&GaussianPrior>,
        normalize: bool,
    ) -> Result<Solution> {
        let design = self.spectrum_design(phases, normalize)?;
        let solution = inference::solve(&design, data, noise, prior)?;
        let n = self.components[0].spectrum.len();
        for (i, c) in self.components.iter_mut().enumerate() {
            c.spectrum
                .copy_from_slice(&solution.mean[i * n..(i + 1) * n]);
        }
        Ok(solution)
    }
    /// Convolve padded rest-frame spectra. Output length = padded_length - 2*half_width.
    pub fn spectrum(&self, phase: f64, normalize: bool) -> Result<Vec<f64>> {
        require(
            !self.components.is_empty(),
            "Doppler map requires spectral components",
        )?;
        let n = self.components[0].spectrum.len();
        require(self.half_width < n / 2, "spectrum is too short for kernel")?;
        let mut output = vec![0.; n - 2 * self.half_width];
        let mut baseline = 0.;
        for c in &self.components {
            require(
                c.spectrum.len() == n && c.spectrum.iter().all(|v| v.is_finite()),
                "invalid component spectrum",
            )?;
            let k = kernel(
                &c.map,
                self.inclination,
                phase,
                self.veq,
                &self.limb_darkening,
                self.log_spacing,
                self.half_width,
            )?;
            baseline += k.iter().sum::<f64>();
            for (i, v) in output.iter_mut().enumerate() {
                for (j, w) in k.iter().enumerate() {
                    *v += w * c.spectrum[i + j];
                }
            }
        }
        if normalize {
            require(baseline != 0., "zero Doppler continuum")?;
            for v in &mut output {
                *v /= baseline;
            }
        }
        Ok(output)
    }
}

/// Linear spline operator, matching the default interp_order=1 in upstream.
/// Outside the input domain the first/last segment is extrapolated, as in the
/// upstream spline. Wavelengths must be strictly increasing and positive.
pub fn interpolation(input: &[f64], output: &[f64]) -> Result<Matrix> {
    require(
        input.len() >= 2
            && input.iter().all(|v| v.is_finite() && *v > 0.)
            && input.windows(2).all(|p| p[1] > p[0])
            && output.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid wavelength grid",
    )?;
    require(
        input
            .len()
            .checked_mul(output.len())
            .is_some_and(|n| n <= 32_000_000),
        "interpolation matrix is too large",
    )?;
    let mut operator = Matrix::zeros(output.len(), input.len());
    for (i, &x) in output.iter().enumerate() {
        let j = input
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(input.len() - 2);
        let f = (x - input[j]) / (input[j + 1] - input[j]);
        operator[(i, j)] = 1. - f;
        operator[(i, j + 1)] = f;
    }
    Ok(operator)
}

#[derive(Clone, Debug)]
pub struct WavelengthGrid {
    pub observed: Vec<f64>,
    pub internal: Vec<f64>,
    pub padded: Vec<f64>,
    pub log_spacing: f64,
    pub half_width: usize,
}
impl WavelengthGrid {
    pub fn new(observed: &[f64], oversample: usize, maximum_velocity: f64) -> Result<Self> {
        require(
            observed.len() >= 2
                && observed.windows(2).all(|p| p[1] > p[0])
                && observed.iter().all(|v| v.is_finite() && *v > 0.)
                && (1..=100).contains(&oversample)
                && maximum_velocity.is_finite()
                && (0. ..SPEED_OF_LIGHT).contains(&maximum_velocity),
            "invalid Doppler wavelength grid",
        )?;
        let n = (observed.len() * oversample) | 1;
        let first = observed[0].ln();
        let last = observed[observed.len() - 1].ln();
        let spacing = (last - first) / (n - 1) as f64;
        let width = ((maximum_velocity / SPEED_OF_LIGHT).atanh() / spacing)
            .ceil()
            .max(1.);
        require(
            width <= 100_000. && n <= 1_000_000,
            "Doppler grid too large",
        )?;
        let half_width = width as usize;
        let padded: Vec<_> = (0..n + 2 * half_width)
            .map(|i| (first + (i as f64 - half_width as f64) * spacing).exp())
            .collect();
        Ok(Self {
            observed: observed.to_vec(),
            internal: padded[half_width..half_width + n].to_vec(),
            padded,
            log_spacing: spacing,
            half_width,
        })
    }
}
