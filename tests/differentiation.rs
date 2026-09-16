use starry_rust::{Map, differentiation, occultation::Occultor};
#[test]
fn all_emitted_derivatives() {
    let mut map = Map::new(3).unwrap();
    map.set(1, 1, 0.14).unwrap();
    map.set(2, -1, -0.08).unwrap();
    map.set(3, 2, 0.05).unwrap();
    map.set_amplitude(1.3).unwrap();
    let u = [0.3, 0.1];
    let inc = 0.8;
    let obl = 0.4;
    let phase = 0.7;
    let occ = Some(Occultor::new(0.2, 0.6, 0.15).unwrap());
    let d = differentiation::emitted(&map, inc, obl, phase, &u, occ).unwrap();
    let flux = |inc, obl, phase| {
        map.projected(inc, obl, phase)
            .unwrap()
            .flux_limb_darkened(&u, occ)
            .unwrap()
    };
    let h = 1e-5;
    for (actual, expected) in [
        (
            d.inclination,
            (flux(inc + h, obl, phase) - flux(inc - h, obl, phase)) / (2. * h),
        ),
        (
            d.obliquity,
            (flux(inc, obl + h, phase) - flux(inc, obl - h, phase)) / (2. * h),
        ),
        (
            d.phase,
            (flux(inc, obl, phase + h) - flux(inc, obl, phase - h)) / (2. * h),
        ),
    ] {
        assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
    }
    for i in 0..2 {
        let mut a = u;
        let mut b = u;
        a[i] += h;
        b[i] -= h;
        let view = map.projected(inc, obl, phase).unwrap();
        let fd = (view.flux_limb_darkened(&a, occ).unwrap()
            - view.flux_limb_darkened(&b, occ).unwrap())
            / (2. * h);
        assert!((d.limb[i] - fd).abs() < 1e-9);
    }
    let reconstructed: f64 = d
        .coefficients
        .iter()
        .zip(map.coefficients())
        .map(|(a, b)| a * b)
        .sum();
    assert!((reconstructed - d.value).abs() < 1e-12);
    assert!((d.amplitude * map.amplitude() - d.value).abs() < 1e-12);
    map.set_amplitude(0.).unwrap();
    let zero = differentiation::emitted(&map, inc, obl, phase, &u, occ).unwrap();
    assert_eq!(zero.value, 0.);
    assert!(zero.amplitude > 0.);
    assert!(zero.coefficients.iter().all(|v| *v == 0.));
}
