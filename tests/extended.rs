use starry_rust::{Map, occultation::Occultor, reflected, rv};
use std::f64::consts::PI;
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() < t, "{a} != {b}");
}
#[test]
fn lambert_phase_curve_and_inverse_square() {
    let map = Map::new(3).unwrap();
    for alpha in [0., 0.01, 0.3, PI / 2., 2.8, PI] {
        let source = [0., alpha.sin(), alpha.cos()];
        let expected = 2. / (3. * PI) * (alpha.sin() + (PI - alpha) * alpha.cos());
        close(
            reflected::flux(&map, source, 0., None).unwrap(),
            expected,
            1e-13,
        );
        close(
            reflected::flux(&map, source.map(|v| 2. * v), 0., None).unwrap(),
            expected / 4.,
            1e-13,
        );
    }
    close(
        reflected::intensity(&map, [0., 0., 1.], [0., 0., 1.], 0.).unwrap(),
        1. / PI,
        1e-14,
    );
}
#[test]
fn reflected_occultation_limits_and_symmetry() {
    let map = Map::new(2).unwrap();
    for rough in [0., 0.3] {
        for b in [-1_f64, -0.7, 0., 0.7, 1.] {
            let source = [0., (1. - b * b).sqrt(), -b];
            let full = reflected::flux(&map, source, rough, None).unwrap();
            close(
                reflected::flux(
                    &map,
                    source,
                    rough,
                    Some(Occultor::new(0., 0., 2.).unwrap()),
                )
                .unwrap(),
                0.,
                1e-14,
            );
            close(
                reflected::flux(
                    &map,
                    source,
                    rough,
                    Some(Occultor::new(2., 0., 0.1).unwrap()),
                )
                .unwrap(),
                full,
                1e-14,
            );
            let a = reflected::flux(
                &map,
                source,
                rough,
                Some(Occultor::new(0.3, 0.2, 0.2).unwrap()),
            )
            .unwrap();
            let c = reflected::flux(
                &map,
                source,
                rough,
                Some(Occultor::new(-0.3, 0.2, 0.2).unwrap()),
            )
            .unwrap();
            close(a, c, 2e-11);
        }
    }
    let r = 0.3_f64;
    close(
        reflected::flux(
            &map,
            [0., 0., 1.],
            0.,
            Some(Occultor::new(0., 0., r).unwrap()),
        )
        .unwrap(),
        2. / 3. * (1. - r * r).powf(1.5),
        1e-12,
    );
}
#[test]
fn projection_and_rv() {
    let mut map = Map::new(2).unwrap();
    map.set(1, -1, 0.2).unwrap();
    close(
        map.projected(PI / 2., 0., 0.).unwrap().flux(None).unwrap(),
        map.flux(None).unwrap(),
        1e-14,
    );
    let pole = map.projected(0., 0., 0.).unwrap();
    close(pole.coefficients()[2], 0.2, 1e-14);
    let uniform = Map::new(0).unwrap();
    for inc in [0., 0.4, PI / 2.] {
        for alpha in [0., 0.3] {
            close(
                rv::radial_velocity(&uniform, inc, 0.3, 0., 1000., alpha, &[0.4, 0.2], None)
                    .unwrap(),
                0.,
                1e-10,
            );
        }
    }
    let a = rv::radial_velocity(
        &uniform,
        PI / 2.,
        0.,
        0.,
        1000.,
        0.,
        &[],
        Some(Occultor::new(0.4, 0.1, 0.1).unwrap()),
    )
    .unwrap();
    let b = rv::radial_velocity(
        &uniform,
        PI / 2.,
        0.,
        0.,
        1000.,
        0.,
        &[],
        Some(Occultor::new(-0.4, 0.1, 0.1).unwrap()),
    )
    .unwrap();
    close(a, -1000. * 0.4 * 0.01 / 0.99, 1e-10);
    close(a, -b, 1e-10);
}

#[test]
fn orbit_equation_velocity_and_mass_conversion() {
    use starry_rust::orbit::{Orbit, solve_kepler};
    for e in [0., 0.5, 0.99, 0.999999] {
        for m in [-3., -0.1, 0., 0.001, 2.9] {
            let x = solve_kepler(m, e).unwrap();
            close(x - e * x.sin(), m, 1e-13);
        }
    }
    let mut orbit = Orbit::from_period(3., 1.2).unwrap();
    let other = Orbit::from_semimajor_axis(orbit.semimajor_axis, 1.2).unwrap();
    close(other.period, 3., 1e-13);
    orbit.eccentricity = 0.4;
    orbit.omega = 0.7;
    orbit.ascending_node = 0.3;
    orbit.inclination = 1.2;
    for t in [-0.7, 0., 0.4] {
        let (p, v) = orbit.state(t).unwrap();
        let h = 1e-6;
        let (a, _) = orbit.state(t + h).unwrap();
        let (b, _) = orbit.state(t - h).unwrap();
        for k in 0..3 {
            close(v[k], (a[k] - b[k]) / (2. * h), 1e-7);
        }
        let (q, _) = orbit.state(t + orbit.period).unwrap();
        for k in 0..3 {
            close(p[k], q[k], 1e-12);
        }
    }
}

#[test]
fn system_transit_eclipse_and_barycenter() {
    use starry_rust::{
        orbit::Orbit,
        system::{Body, Secondary, System},
    };
    let primary = Body::new(Map::new(0).unwrap(), 1., 1.).unwrap();
    let mut dark = Map::new(0).unwrap();
    dark.set_amplitude(0.).unwrap();
    let secondary = Secondary {
        body: Body::new(dark, 0.1, 0.01).unwrap(),
        orbit: Orbit::circular(5., 3.).unwrap(),
    };
    let system = System::new(primary, vec![secondary]).unwrap();
    close(system.light_curve(&[0.]).unwrap()[0], 0.99, 1e-13);
    close(system.light_curve(&[0.75]).unwrap()[0], 1., 1e-13);
    let states = system.states(0.3).unwrap();
    for k in 0..3 {
        close(states[0].0[k] + 0.01 * states[1].0[k], 0., 1e-14);
    }
    let mut planet = system.clone();
    planet.secondaries[0].body.map.set_amplitude(0.2).unwrap();
    close(planet.light_curve(&[0.]).unwrap()[0], 1.19, 1e-12);
    close(planet.light_curve(&[1.5]).unwrap()[0], 1., 1e-12);
    let mut reflection = system.clone();
    reflection.primary.radius = 0.;
    reflection.secondaries[0].body.reflected = true;
    reflection.secondaries[0]
        .body
        .map
        .set_amplitude(1.)
        .unwrap();
    close(
        reflection.flux(1.5).unwrap()[1],
        2. / 3. * 0.1_f64.powi(2) / 25.,
        1e-12,
    );
    let mut exposed = system.clone();
    exposed.exposure = 0.001;
    exposed.exposure_samples = 101;
    close(exposed.light_curve(&[0.]).unwrap()[0], 0.99, 1e-12);
}

#[test]
fn union_occultation_does_not_double_count() {
    use starry_rust::occultation::{uniform_flux, union_moments};
    let o = Occultor::new(0.3, 0.2, 0.2).unwrap();
    let v = union_moments(0, &[o, o], Default::default())
        .unwrap()
        .values[0]
        / PI;
    close(v, uniform_flux(o.x.hypot(o.y), o.radius).unwrap(), 1e-12);
    let p = Occultor::new(-0.3, 0.2, 0.2).unwrap();
    close(
        union_moments(0, &[o, p], Default::default())
            .unwrap()
            .values[0]
            / PI,
        0.92,
        1e-12,
    );
}

#[test]
fn gaussian_inference_closed_form() {
    use starry_rust::{
        inference::{GaussianPrior, marginal_log_likelihood, solve},
        matrix::Matrix,
    };
    let design = Matrix {
        rows: 2,
        cols: 1,
        data: vec![1., 1.],
    };
    let noise = Matrix::identity(2);
    let prior = GaussianPrior {
        mean: vec![0.],
        covariance: Matrix::identity(1),
    };
    let posterior = solve(&design, &[2., 4.], &noise, Some(&prior)).unwrap();
    close(posterior.mean[0], 2., 1e-14);
    close(posterior.covariance[(0, 0)], 1. / 3., 1e-14);
    let expected = -0.5 * (8. + 3_f64.ln() + 2. * (2. * PI).ln());
    close(
        marginal_log_likelihood(&design, &[2., 4.], &noise, &prior).unwrap(),
        expected,
        1e-13,
    );
    close(posterior.draw(&[0.]).unwrap()[0], 2., 1e-14);
    let fit = solve(&design, &[2., 4.], &noise, None).unwrap();
    close(fit.mean[0], 3., 1e-14);
}

#[test]
fn spots_images_and_doppler() {
    use starry_rust::{
        doppler::{Component, DopplerMap},
        surface::{self, Projection},
    };
    let mut map = Map::new(8).unwrap();
    surface::add_spot(&mut map, 0.5, 0.4, 0., 0., None).unwrap();
    assert!(map.intensity(0., 0.).unwrap() < 1. / PI);
    let image = surface::render(&map, 32, 32, Projection::Orthographic).unwrap();
    assert!(image[0].is_nan());
    assert!(image[16 * 32 + 16].is_finite());
    let spectrum = vec![1.; 101];
    let d = DopplerMap {
        components: vec![Component {
            map: Map::new(0).unwrap(),
            spectrum,
        }],
        log_spacing: 1e-5,
        half_width: 10,
        inclination: PI / 2.,
        veq: 20000.,
        limb_darkening: vec![],
    };
    for v in d.spectrum(0., true).unwrap() {
        close(v, 1., 1e-13);
    }
    let k = starry_rust::doppler::kernel(&Map::new(0).unwrap(), PI / 2., 0., 20000., &[], 1e-5, 10)
        .unwrap();
    close(k.iter().sum(), 1., 1e-14);
    for i in 0..k.len() {
        close(k[i], k[k.len() - 1 - i], 1e-14);
    }
}

#[test]
fn oblate_flux_normalization_and_gravity_profile() {
    use starry_rust::oblate::{self, GravityDarkening};
    let map = Map::new(0).unwrap();
    for f in [0., 0.1, 0.4] {
        close(
            oblate::flux(&map, f, PI / 2., 0., 0., &[], None, None, false).unwrap(),
            1. - f,
            1e-14,
        );
        close(
            oblate::flux(&map, f, PI / 2., 0., 0., &[], None, None, true).unwrap(),
            1.,
            1e-14,
        );
    }
    let g = GravityDarkening {
        degree: 4,
        omega: 0.3,
        flattening: 0.1,
        beta: 0.23,
        polar_temperature: 6000.,
        wavelength_meters: 550e-9,
    };
    close(
        g.profile(1.).unwrap(),
        1. / (1.43877735e-2_f64 / (550e-9 * 6000.)).exp_m1(),
        1e-14,
    );
    assert!(g.profile(0.).unwrap() < g.profile(1.).unwrap());
    close(
        oblate::flux(&map, 0.1, PI / 2., 0., 0., &[], Some(g), None, true).unwrap(),
        1.,
        1e-12,
    );
}
