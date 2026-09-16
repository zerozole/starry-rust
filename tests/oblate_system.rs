use starry_rust::{
    Map,
    conic::Ellipse,
    oblate,
    occultation::{self, Integration, Occultor},
    orbit::Orbit,
    system::{Body, Secondary, System},
};
use std::f64::consts::PI;
fn circle(x: f64, y: f64, r: f64) -> Ellipse {
    Ellipse {
        x,
        y,
        major: r,
        minor: r,
        angle: 0.,
    }
}
#[test]
fn conic_moments_match_circles_and_ellipses() {
    for degree in [0, 1, 3] {
        for q in [1., 0.8] {
            for (x, y, r) in [(0.2, 0.3, 0.1), (0., 0.7, 0.4), (0., 0., 2.)] {
                let conic = occultation::projected_moments(
                    degree,
                    q,
                    &[circle(x, y, r)],
                    Integration::default(),
                )
                .unwrap();
                let reference = occultation::ellipse_moments(
                    degree,
                    q,
                    Some(Occultor::new(x, y, r).unwrap()),
                    Integration::default(),
                )
                .unwrap();
                for (a, b) in conic.values.iter().zip(reference.values) {
                    assert!((a - b).abs() < 1e-10, "{a} != {b}");
                }
            }
        }
    }
    let ellipse = Ellipse {
        x: 0.1,
        y: -0.2,
        major: 0.2,
        minor: 0.1,
        angle: 0.73,
    };
    let m =
        occultation::projected_moments(0, 1., &[ellipse, ellipse], Integration::default()).unwrap();
    assert!((m.values[0] - PI * (1. - 0.02)).abs() < 1e-12);
}
#[test]
fn oblate_primary_transit_matches_map() {
    let mut body = Body::new(Map::new(2).unwrap(), 1., 1.).unwrap();
    body.map.set(1, 1, 0.1).unwrap();
    body.flattening = Some(0.2);
    body.obliquity = 0.4;
    let mut planet = Body::new(Map::new(0).unwrap(), 0.1, 0.).unwrap();
    planet.map.set_amplitude(0.).unwrap();
    let orbit = Orbit::from_semimajor_axis(8., 1.).unwrap();
    let system = System::new(
        body.clone(),
        vec![Secondary {
            body: planet,
            orbit,
        }],
    )
    .unwrap();
    for t in [-0.01, 0., 0.01] {
        let positions = system.states(t).unwrap();
        let p = positions[1].0;
        let s = positions[0].0;
        let reference = oblate::flux(
            &body.map,
            0.2,
            body.inclination,
            body.obliquity,
            2. * PI * t,
            &[],
            None,
            Some(Occultor::new(p[0] - s[0], p[1] - s[1], 0.1).unwrap()),
            true,
        )
        .unwrap();
        assert!((system.flux(t).unwrap()[0] - reference).abs() < 1e-10);
    }
}
#[test]
fn eclipse_uses_elliptical_primary_silhouette() {
    let ellipse = Ellipse {
        x: 0.,
        y: 0.,
        major: 1.3,
        minor: 0.7,
        angle: 0.4,
    };
    let visible = occultation::projected_moments(0, 1., &[ellipse], Integration::default())
        .unwrap()
        .values[0]
        / PI;
    assert!(visible > 0. && visible < 0.3);
    let rotated =
        occultation::projected_moments(0, 1., &[ellipse.rotated(0.8)], Integration::default())
            .unwrap()
            .values[0]
            / PI;
    assert!((rotated - visible).abs() < 1e-10);
}
