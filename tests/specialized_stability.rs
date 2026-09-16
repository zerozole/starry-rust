use starry_rust::{Map, oblate, occultation::Occultor, quadrature, reflected, rv};
use std::f64::consts::PI;

#[test]
fn direct_surface_matches_analytic_overlap_and_phase() {
    for (x, y, r) in [
        (0., 0., 0.3),
        (0.7, 0.2, 0.4),
        (0.2, 0.3, 2.),
        (1.3, 0.1, 0.1),
    ] {
        let o = Occultor::new(x, y, r).unwrap();
        let silhouettes = quadrature::circles(&[o]).unwrap();
        let value = quadrature::surface(1., -1., &silhouettes, 1e-11, |_| Ok(1. / PI)).unwrap();
        let expected = starry_rust::occultation::uniform_flux(x.hypot(y), r).unwrap();
        assert!((value - expected).abs() < 2e-10, "{value} {expected}");
    }
    for b in [-1_f64, -0.8, 0., 0.7, 1.] {
        let value = quadrature::surface(1., b, &[], 1e-11, |p| {
            Ok(((1. - b * b).sqrt() * p[1] - b * p[2]) / PI)
        })
        .unwrap();
        let alpha = (-b).acos();
        let expected = 2. / (3. * PI) * (alpha.sin() + (PI - alpha) * alpha.cos());
        assert!((value - expected).abs() < 2e-11);
    }
}

#[test]
fn high_degree_specialized_paths_reproduce_low_degree_models() {
    let mut low = Map::new(2).unwrap();
    low.set(1, 1, 0.2).unwrap();
    low.set(2, -1, -0.1).unwrap();
    let mut high = Map::new(24).unwrap();
    let mut coefficients = vec![0.; 625];
    coefficients[..9].copy_from_slice(low.coefficients());
    high.set_coefficients(&coefficients).unwrap();
    let o = Some(Occultor::new(0.6, 0.2, 0.3).unwrap());
    for roughness in [0., 0.3] {
        for source in [[0.4, 0.8, 1.3], [0.4, 0.8, -1.3]] {
            let a = reflected::flux(&low, source, roughness, o).unwrap();
            let b = reflected::flux(&high, source, roughness, o).unwrap();
            assert!((a - b).abs() < 2e-9, "reflection {a} {b}");
        }
    }
    for f in [0., 0.3] {
        let a = oblate::flux(&low, f, 1.1, 0.3, 0.2, &[0.3, 0.1], None, o, true).unwrap();
        let b = oblate::flux(&high, f, 1.1, 0.3, 0.2, &[0.3, 0.1], None, o, true).unwrap();
        assert!((a - b).abs() < 2e-9, "oblateness {a} {b}");
    }
    let a = rv::radial_velocity(&low, 1.1, 0.3, 0.2, 20000., 0.2, &[0.3, 0.1], o).unwrap();
    let b = rv::radial_velocity(&high, 1.1, 0.3, 0.2, 20000., 0.2, &[0.3, 0.1], o).unwrap();
    assert!((a - b).abs() < 2e-6, "velocity {a} {b}");
}

#[test]
fn degree_32_oblate_spherical_limit_matches_ring_integral() {
    let mut map = Map::new(32).unwrap();
    map.set(32, 7, 0.2).unwrap();
    map.set(29, -4, -0.1).unwrap();
    let o = Some(Occultor::new(0.3, -0.4, 0.4).unwrap());
    let expected = map.flux(o).unwrap();
    let actual = oblate::flux(&map, 0., PI / 2., 0., 0., &[], None, o, false).unwrap();
    assert!((expected - actual).abs() < 2e-10, "{expected} {actual}");
}
