use starry_rust::{
    Map, basis, elliptic, map,
    occultation::{self, Integration, Occultor},
    rotation,
};
use std::f64::consts::PI;
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() < t,
        "{a:.17e} != {b:.17e}, difference={:.3e}",
        (a - b).abs()
    );
}

#[test]
fn low_degree_basis_and_normalization() {
    let a = basis::a1(1).unwrap();
    close(a[(0, 0)], 1. / PI, 1e-15);
    for (i, j) in [(3, 1), (2, 2), (1, 3)] {
        close(a[(i, j)], 3_f64.sqrt() / PI, 1e-15);
    }
    let m = Map::new(6).unwrap();
    close(m.flux(None).unwrap(), 1., 1e-14);
    close(m.intensity(0., 0.).unwrap(), 1. / PI, 1e-15);
}
#[test]
fn upstream_intensity_cases() {
    let mut m = Map::new(1).unwrap();
    m.set(1, -1, 1.).unwrap();
    close(
        m.intensity(PI / 6., 1.2).unwrap(),
        (1. + 3_f64.sqrt() / 2.) / PI,
        1e-14,
    );
    m.set(1, -1, 0.).unwrap();
    m.set(1, 0, 1.).unwrap();
    close(
        m.intensity(0., 0.).unwrap(),
        (1. + 3_f64.sqrt()) / PI,
        1e-14,
    );
}
#[test]
fn polynomial_derivatives_at_axes() {
    let g = basis::polynomial_gradient(3, [0., 0., 1.]).unwrap();
    close(g[0][1], 1., 1e-15);
    close(g[1][3], 1., 1e-15);
    close(g[2][2], 1., 1e-15);
    assert!(g.iter().flatten().all(|v| v.is_finite()));
}
#[test]
fn intensity_gradients() {
    let mut m = Map::new(6).unwrap();
    let y: Vec<f64> = (0..49).map(|i| (i as f64 * 1.71).sin() / 49.).collect();
    m.set_coefficients(&y).unwrap();
    for (lat, lon) in [(0., 0.), (0.34, -0.78), (-1.2, 2.4)] {
        let g = m.intensity_gradient(lat, lon).unwrap();
        let h = 1e-6;
        close(
            g[0],
            (m.intensity(lat + h, lon).unwrap() - m.intensity(lat - h, lon).unwrap()) / (2. * h),
            2e-9,
        );
        close(
            g[1],
            (m.intensity(lat, lon + h).unwrap() - m.intensity(lat, lon - h).unwrap()) / (2. * h),
            2e-9,
        );
    }
}
#[test]
fn rotations_preserve_field_and_power() {
    let mut m = Map::new(6).unwrap();
    let y: Vec<f64> = (0..49).map(|i| (i as f64 * 0.73).cos() / 49.).collect();
    m.set_coefficients(&y).unwrap();
    for axis in [[1., 0., 0.], [0., 0., 1.], [1., 2., 3.]] {
        for angle in [0., 0.7, PI, -PI / 2.] {
            let mut r = m.clone();
            r.rotate(axis, angle).unwrap();
            let p = [0.3, 0.4, 0.75_f64.sqrt()];
            let q = rotation::apply(
                rotation::transpose(rotation::axis_angle(axis, angle).unwrap()),
                p,
            );
            close(
                r.intensity_xyz(p).unwrap(),
                m.intensity_xyz(q).unwrap(),
                2e-13,
            );
            for l in 0_usize..=6 {
                close(
                    r.coefficients()[l * l..(l + 1).pow(2)]
                        .iter()
                        .map(|v| v * v)
                        .sum(),
                    y[l * l..(l + 1).pow(2)].iter().map(|v| v * v).sum(),
                    2e-13,
                );
            }
            r.rotate(axis, -angle).unwrap();
            for (a, b) in r.coefficients().iter().zip(&y) {
                close(*a, *b, 2e-13);
            }
        }
    }
}
#[test]
fn uniform_occultations_and_contact_limits() {
    for r in [0., 1e-4, 0.1, 0.5, 1., 1.5, 3.] {
        for b in [0., 1e-6, 0.3, 0.5, 0.9, 1., 1.1, 1.5, 3., 4.] {
            for phi in [0_f64, 0.7, PI / 2.] {
                let o = Occultor::new(b * phi.cos(), b * phi.sin(), r).unwrap();
                let v = occultation::visible_moments(0, o, Integration::default())
                    .unwrap()
                    .values[0]
                    / PI;
                close(v, occultation::uniform_flux(b, r).unwrap(), 2e-10);
            }
        }
    }
}
#[test]
fn centered_dipole_occultation() {
    for r in [0.1, 0.5, 0.9] {
        let row = Map::new(1)
            .unwrap()
            .flux_design(Some(Occultor::new(0., 0., r).unwrap()))
            .unwrap();
        close(row[0], 1. - r * r, 1e-12);
        close(row[1], 0., 1e-12);
        close(row[3], 0., 1e-12);
        close(row[2], 2. / 3_f64.sqrt() * (1. - r * r).powf(1.5), 1e-12);
    }
}
#[test]
fn limb_darkening_normalization() {
    for u in [
        vec![],
        vec![0.5],
        vec![0.4, 0.2],
        vec![0.3, -0.2, 0.1, 0.02],
    ] {
        close(map::limb_flux(&u, None).unwrap(), 1., 2e-14);
        close(map::limb_intensity(1., &u).unwrap(), 1., 1e-15);
        close(
            map::limb_intensity(0., &u).unwrap(),
            1. - u.iter().sum::<f64>(),
            1e-15,
        );
        close(
            map::limb_flux(&u, Some(Occultor::new(0., 0., 2.).unwrap())).unwrap(),
            0.,
            1e-15,
        );
    }
    let u = 0.5;
    let r = 0.3_f64;
    let expected = ((1. - u) * (1. - r * r) + 2. * u / 3. * (1. - r * r).powf(1.5)) / (1. - u / 3.);
    close(
        map::limb_flux(&[u], Some(Occultor::new(0., 0., r).unwrap())).unwrap(),
        expected,
        1e-12,
    );
}
#[test]
fn elliptic_limits() {
    close(elliptic::cel(0., 1., 1., 1., 1.).unwrap(), PI / 2., 1e-14);
    close(
        elliptic::cel(0.5, 0.5_f64.sqrt(), 1., 1., 0.5).unwrap(),
        1.3506438810476755,
        1e-14,
    );
}
#[test]
fn invalid_inputs_are_errors() {
    assert!(Map::new(33).is_err());
    assert!(Occultor::new(0., 0., -1.).is_err());
    assert!(Occultor::new(f64::NAN, 0., 1.).is_err());
    assert!(Map::new(1).unwrap().intensity_xyz([0., 0., 0.]).is_err());
    assert!(Map::new(1).unwrap().rotate([0., 0., 0.], 0.).is_err());
    assert!(map::limb_flux(&[4.], None).is_err());
    assert!(elliptic::cel(1.1, 0., 1., 1., 1.).is_err());
}

#[test]
fn analytic_matches_independent_integrator() {
    for degree in [0, 1, 2, 5, 10] {
        let a = basis::a1(degree).unwrap();
        for (b, r) in [
            (0., 0.3),
            (0., 0.9),
            (0.5, 0.5),
            (0.7, 0.2),
            (1., 0.1),
            (0.7, 1.2),
            (0.01, 1.),
            (1.2, 0.20001),
        ] {
            for phi in [0_f64, 0.73, 2.3] {
                let occ = Occultor::new(b * phi.cos(), b * phi.sin(), r).unwrap();
                let analytic = starry_rust::solver::flux_design(&a, degree, occ).unwrap();
                let numerical =
                    starry_rust::solver::numerical_flux_design(&a, degree, occ, Default::default())
                        .unwrap();
                for (i, (x, y)) in analytic.iter().zip(numerical).enumerate() {
                    assert!(
                        (*x - y).abs() < 3e-9,
                        "degree={degree}, b={b}, r={r}, phi={phi}, index={i}: {x} != {y}"
                    );
                }
            }
        }
    }
}

#[test]
fn upstream_rotation_convention() {
    let mut m = Map::new(1).unwrap();
    m.set(1, 1, 1.).unwrap();
    m.rotate([0., 0., 1.], PI / 2.).unwrap();
    for (&x, y) in m.coefficients().iter().zip([1., 1., 0., 0.]) {
        close(x, y, 1e-14);
    }
}

#[test]
fn geometric_gradients_match_flux_differences() {
    let mut map = Map::new(5).unwrap();
    map.set(2, 1, 0.3).unwrap();
    map.set(3, -2, -0.2).unwrap();
    for xyz in [
        [0., 0., 0.3],
        [0.2, 0.3, 0.1],
        [0.7, -0.5, 0.4],
        [-0.3, 0.7, 1.1],
    ] {
        let occ = Occultor::new(xyz[0], xyz[1], xyz[2]).unwrap();
        let grad = map.flux_gradient(occ).unwrap();
        let h = 1e-5;
        for j in 0..3 {
            let mut a = xyz;
            let mut b = xyz;
            a[j] += h;
            b[j] -= h;
            let fa = map
                .flux(Some(Occultor::new(a[0], a[1], a[2]).unwrap()))
                .unwrap();
            let fb = map
                .flux(Some(Occultor::new(b[0], b[1], b[2]).unwrap()))
                .unwrap();
            close(grad[j], (fa - fb) / (2. * h), 3e-8);
        }
    }
}
