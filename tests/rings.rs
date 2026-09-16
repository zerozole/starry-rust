use starry_rust::{
    Map,
    occultation::{Integration, Occultor},
    rings,
};
#[test]
fn harmonic_ring_integration_matches_green_solver() {
    for degree in [0, 1, 5, 10, 15] {
        for (b, r) in [
            (0., 0.),
            (0., 0.4),
            (0.3, 0.1),
            (0.7, 0.4),
            (1.2, 0.5),
            (0.5, 0.5),
            (0.01, 1.),
            (2., 0.2),
        ] {
            let occ = Occultor::new(b * 0.6, b * 0.8, r).unwrap();
            let stable =
                rings::flux_design(degree, Some(occ), &[], Integration::default()).unwrap();
            let green = starry_rust::solver::flux_design(
                &starry_rust::basis::a1(degree).unwrap(),
                degree,
                occ,
            )
            .unwrap();
            let error = stable
                .iter()
                .zip(green)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(error < 2e-8, "degree={degree} b={b} r={r} error={error:e}");
            assert!(
                (stable[0] - starry_rust::occultation::uniform_flux(b, r).unwrap()).abs() < 1e-11
            );
        }
    }
}

#[test]
fn stable_high_degree_geometry_derivative() {
    let mut map = Map::new(20).unwrap();
    let mut y = vec![0.; 441];
    y[0] = 1.;
    for (i, v) in y.iter_mut().enumerate().skip(1) {
        *v = (i as f64).sin() * 0.03;
    }
    map.set_coefficients(&y).unwrap();
    for parameters in [[0.3, 0.4, 0.4], [0.0, 0.01, 1.0]] {
        let derivative = map
            .flux_gradient(Occultor::new(parameters[0], parameters[1], parameters[2]).unwrap())
            .unwrap();
        for k in 0..3 {
            let h = 1e-5;
            let mut a = parameters;
            let mut b = parameters;
            a[k] += h;
            b[k] -= h;
            let f = |p: [f64; 3]| {
                map.flux(Some(Occultor::new(p[0], p[1], p[2]).unwrap()))
                    .unwrap()
            };
            let finite = (f(a) - f(b)) / (2. * h);
            assert!(
                (finite - derivative[k]).abs() < 2e-6,
                "{k}: {finite} {}",
                derivative[k]
            );
        }
    }
}

#[test]
fn harmonic_circle_union() {
    use starry_rust::{basis, occultation};
    let overlap = [
        Occultor::new(-0.2, 0., 0.3).unwrap(),
        Occultor::new(0.2, 0., 0.3).unwrap(),
    ];
    for degree in [0, 5, 20] {
        let row = rings::flux_design_many(degree, &overlap, &[], Integration::default()).unwrap();
        let d = 0.4_f64;
        let r = 0.3_f64;
        let area = 2. * r * r * (d / (2. * r)).acos() - d / 2. * (4. * r * r - d * d).sqrt();
        let expected = 1. - 2. * r * r + area / std::f64::consts::PI;
        assert!((row[0] - expected).abs() < 1e-11);
        if degree <= 5 {
            let moments = occultation::union_moments(degree, &overlap, Integration::default())
                .unwrap()
                .values;
            let reference = basis::a1(degree).unwrap().left_dot(&moments).unwrap();
            for (a, b) in row.iter().zip(reference) {
                assert!((a - b).abs() < 2e-10);
            }
        }
        let duplicate = [overlap[0], overlap[1], overlap[0]];
        let repeated =
            rings::flux_design_many(degree, &duplicate, &[], Integration::default()).unwrap();
        for (a, b) in row.iter().zip(repeated) {
            assert!((a - b).abs() < 2e-12);
        }
    }
    let disjoint = [
        Occultor::new(-0.4, 0., 0.1).unwrap(),
        Occultor::new(0.4, 0., 0.1).unwrap(),
    ];
    let union = rings::flux_design_many(20, &disjoint, &[], Integration::default()).unwrap();
    let a = rings::flux_design(20, Some(disjoint[0]), &[], Integration::default()).unwrap();
    let b = rings::flux_design(20, Some(disjoint[1]), &[], Integration::default()).unwrap();
    let full = rings::flux_design(20, None, &[], Integration::default()).unwrap();
    for i in 0..union.len() {
        assert!((union[i] - (a[i] + b[i] - full[i])).abs() < 2e-11);
    }
}

#[test]
fn degree_twenty_convergence_and_azimuthal_symmetry() {
    for (b, r) in [(0.5, 0.5), (0.01, 1.), (0.7, 0.4)] {
        let a = rings::flux_design(
            20,
            Some(Occultor::new(b, 0., r).unwrap()),
            &[],
            Integration::default(),
        )
        .unwrap();
        let b_ring = rings::flux_design(
            20,
            Some(Occultor::new(b * 0.6, b * 0.8, r).unwrap()),
            &[],
            Integration {
                absolute_tolerance: 1e-13,
                max_depth: 24,
            },
        )
        .unwrap();
        for l in 0..=20 {
            for m in 0..=l {
                let (s, c) = (m as f64 * 0.8_f64.atan2(0.6)).sin_cos();
                let index = starry_rust::basis::index(l, m as isize);
                assert!((b_ring[index] - a[index] * c).abs() < 2e-11);
                if m > 0 {
                    assert!(
                        (b_ring[starry_rust::basis::index(l, -(m as isize))] - a[index] * s).abs()
                            < 2e-11
                    );
                }
            }
        }
    }
}
