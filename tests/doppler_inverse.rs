use starry_rust::{
    Map,
    doppler::{self, Component, DopplerMap, WavelengthGrid},
    inference::GaussianPrior,
    matrix::Matrix,
};
fn model() -> DopplerMap {
    let mut map = Map::new(1).unwrap();
    map.set(1, 1, 0.12).unwrap();
    map.set(1, -1, -0.08).unwrap();
    let spectrum = (0..25)
        .map(|i| 1. - 0.6 * (-((i as f64 - 12.) / 1.7).powi(2)).exp())
        .collect();
    DopplerMap {
        components: vec![Component { map, spectrum }],
        log_spacing: 2e-5,
        half_width: 4,
        inclination: 1.1,
        veq: 21000.,
        limb_darkening: vec![0.3],
    }
}
fn near(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }
}
#[test]
fn designs_and_normalized_jacobian() {
    let m = model();
    let phases = [0., 0.7, 1.8];
    let weights = m.map_weights();
    let direct: Vec<_> = phases
        .iter()
        .flat_map(|t| m.spectrum(*t, false).unwrap())
        .collect();
    let (d, _) = m.map_design(&phases).unwrap();
    near(&d.dot(&weights).unwrap(), &direct, 1e-13);
    let s = m.spectrum_design(&phases, false).unwrap();
    near(&s.dot(&m.components[0].spectrum).unwrap(), &direct, 1e-13);
    let (f, j) = m.map_jacobian(&phases, true).unwrap();
    let direct: Vec<_> = phases
        .iter()
        .flat_map(|t| m.spectrum(*t, true).unwrap())
        .collect();
    near(&f, &direct, 1e-13);
    for k in 0..weights.len() {
        let mut a = m.clone();
        let mut b = m.clone();
        let mut wa = weights.clone();
        let mut wb = weights.clone();
        wa[k] += 1e-5;
        wb[k] -= 1e-5;
        a.set_map_weights(&wa).unwrap();
        b.set_map_weights(&wb).unwrap();
        let fa = a.map_jacobian(&phases, true).unwrap().0;
        let fb = b.map_jacobian(&phases, true).unwrap().0;
        for i in 0..f.len() {
            assert!(((fa[i] - fb[i]) / 2e-5 - j[(i, k)]).abs() < 1e-9);
        }
    }
}
#[test]
fn recover_surface_and_rest_spectrum() {
    let truth = model();
    let phases = [0., 0.8, 1.7, 2.6];
    let flux: Vec<_> = phases
        .iter()
        .flat_map(|t| truth.spectrum(*t, false).unwrap())
        .collect();
    let mut noise = Matrix::identity(flux.len());
    for i in 0..flux.len() {
        noise[(i, i)] = 1e-8;
    }
    let mut fit = truth.clone();
    fit.components[0].map = Map::new(1).unwrap();
    fit.solve_map(&phases, &flux, &noise, None).unwrap();
    near(&fit.map_weights(), &truth.map_weights(), 1e-10);
    fit.components[0].spectrum.fill(1.);
    let prior = GaussianPrior {
        mean: vec![1.; 25],
        covariance: Matrix::identity(25),
    };
    fit.solve_spectrum(&phases, &flux, &noise, Some(&prior), false)
        .unwrap();
    let recovered: Vec<_> = phases
        .iter()
        .flat_map(|t| fit.spectrum(*t, false).unwrap())
        .collect();
    near(&recovered, &flux, 3e-7);
}
#[test]
fn wavelength_grid_and_spline() {
    let grid = WavelengthGrid::new(&[500., 501., 502., 503.], 2, 20000.).unwrap();
    assert_eq!(grid.internal.len(), 9);
    assert_eq!(grid.padded.len(), 9 + 2 * grid.half_width);
    for pair in grid.padded.windows(2) {
        assert!(((pair[1] / pair[0]).ln() - grid.log_spacing).abs() < 1e-14);
    }
    let input = [1., 2., 4.];
    let output = [0.5, 1.5, 3., 5.];
    let values = [5., 7., 11.];
    near(
        &doppler::interpolation(&input, &output)
            .unwrap()
            .dot(&values)
            .unwrap(),
        &[4., 6., 9., 13.],
        1e-14,
    );
}

#[test]
fn unrestricted_spectral_map_and_adjoint() {
    let model = model();
    let phases = [0., 0.7, 1.3];
    let c = &model.components[0];
    let weights: Vec<_> = c
        .map
        .coefficients()
        .iter()
        .flat_map(|y| c.spectrum.iter().map(move |s| y * s * c.map.amplitude()))
        .collect();
    let flux = model.spectral_map_dot(&phases, &weights, false).unwrap();
    let direct: Vec<_> = phases
        .iter()
        .flat_map(|t| model.spectrum(*t, false).unwrap())
        .collect();
    near(&flux, &direct, 1e-13);
    let arbitrary: Vec<_> = weights
        .iter()
        .enumerate()
        .map(|(i, _)| (i as f64).cos())
        .collect();
    let residual: Vec<_> = flux
        .iter()
        .enumerate()
        .map(|(i, _)| (i as f64).sin())
        .collect();
    let forward = model.spectral_map_dot(&phases, &arbitrary, false).unwrap();
    let adjoint = model.spectral_map_dot(&phases, &residual, true).unwrap();
    let lhs: f64 = forward.iter().zip(&residual).map(|(a, b)| a * b).sum();
    let rhs: f64 = arbitrary.iter().zip(&adjoint).map(|(a, b)| a * b).sum();
    assert!((lhs - rhs).abs() < 1e-12);
}
