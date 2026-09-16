use starry_rust::{Map, oblate, occultation::Occultor, quadrature, reflected};

#[test]
fn degree32_mixed_harmonics_ring_vs_surface() {
    let mut m = Map::new(32).unwrap();
    let mut y = vec![0.; 1089];
    y[0] = 1.;
    for (j, v) in y.iter_mut().enumerate().skip(1) {
        *v = 0.004 * (j as f64).sin() / (1. + (j as f64).sqrt());
    }
    y[1088] = 0.03;
    m.set_coefficients(&y).unwrap();
    for (x, y, r) in [
        (0., 0., 0.0001),
        (0.000001, 0.1, 0.2),
        (0.95, 0.05, 0.3),
        (1.29999, 0., 0.3),
    ] {
        let o = Occultor::new(x, y, r).unwrap();
        let direct =
            quadrature::surface(1., -1., &quadrature::circles(&[o]).unwrap(), 2e-11, |p| {
                m.intensity_xyz(p)
            })
            .unwrap();
        let ring = m.flux(Some(o)).unwrap();
        assert!((direct - ring).abs() < 3e-9, "{x} {r}: {direct} {ring}");
    }
}

#[test]
fn upper_degree_filters_and_extreme_views() {
    let mut low = Map::new(2).unwrap();
    low.set(1, 1, 0.1).unwrap();
    low.set(2, -1, -0.04).unwrap();
    let mut padded = Map::new(27).unwrap();
    let mut y = vec![0.; 784];
    y[..9].copy_from_slice(low.coefficients());
    padded.set_coefficients(&y).unwrap();
    for source in [[0.0001, 0., 2.], [0.001, 0.001, -2.], [0.3, 0.4, 2.]] {
        let occ = Some(Occultor::new(0.7, 0.1, 0.3).unwrap());
        let a = reflected::flux(&low, source, 0.2, occ).unwrap();
        let b = reflected::flux(&padded, source, 0.2, occ).unwrap();
        assert!((a - b).abs() < 3e-9, "{a} {b}");
    }
    for inc in [0.0001, std::f64::consts::FRAC_PI_2 - 0.0001] {
        let gravity = Some(oblate::GravityDarkening {
            degree: 4,
            omega: 0.95,
            flattening: 0.8,
            beta: 0.25,
            polar_temperature: 5000.,
            wavelength_meters: 600e-9,
        });
        let a = oblate::flux(&low, 0.8, inc, 0.3, 0.2, &[0.2], gravity, None, true).unwrap();
        let b = oblate::flux(&padded, 0.8, inc, 0.3, 0.2, &[0.2], gravity, None, true).unwrap();
        assert!((a - b).abs() < 3e-8, "{a} {b}");
    }
}

#[test]
fn degree32_velocity_product_matches_direct_surface() {
    let mut m = Map::new(29).unwrap();
    m.set(29, 29, 0.03).unwrap();
    m.set(1, 1, 0.1).unwrap();
    let o = Occultor::new(0.5, 0.2, 0.3).unwrap();
    let ellipses = quadrature::circles(&[o]).unwrap();
    let denominator =
        quadrature::surface(1., -1., &ellipses, 2e-11, |p| m.intensity_xyz(p)).unwrap();
    let numerator = quadrature::surface(1., -1., &ellipses, 2e-9, |p| {
        Ok(m.intensity_xyz(p)? * 13000. * p[0] * (1. - 0.2 * p[1] * p[1]))
    })
    .unwrap();
    let rv = starry_rust::rv::radial_velocity(
        &m,
        std::f64::consts::FRAC_PI_2,
        0.,
        0.,
        13000.,
        0.2,
        &[],
        Some(o),
    )
    .unwrap();
    assert!(
        (rv - numerator / denominator).abs() < 2e-7,
        "{rv} {}",
        numerator / denominator
    );
}
