use starry_rust::{Map, doppler, map, occultation::Occultor};
use std::f64::consts::PI;
#[test]
fn projected_limb_product_and_chord_quadrature() {
    for degree in [18, 30] {
        let mut m = Map::new(degree).unwrap();
        m.set(degree, 4, 0.04).unwrap();
        m.set(degree - 1, -3, -0.02).unwrap();
        let u = [0.3, 0.1];
        let norm = 1. - u[0] / 3. - u[1] / 6.;
        let filtered = m.limb_filtered(&u).unwrap();
        for i in 0..30 {
            let lat = -1.4 + 2.8 * i as f64 / 29.;
            let lon = 0.7;
            let expected = m.intensity(lat, lon).unwrap()
                * map::limb_intensity(lat.cos() * lon.cos(), &u).unwrap()
                / norm;
            assert!((filtered.intensity(lat, lon).unwrap() - expected).abs() < 2e-12);
        }
        let occ = Some(Occultor::new(0.2, 0.3, 0.4).unwrap());
        assert!(
            (filtered.flux(occ).unwrap() - m.flux_limb_darkened(&u, occ).unwrap()).abs() < 2e-11
        );
        let k = doppler::kernel(&m, PI / 2., 0., 20000., &u, 2e-5, 3).unwrap();
        let x: Vec<_> = (0..7)
            .map(|j| -doppler::SPEED_OF_LIGHT * ((j as f64 - 3.) * 2e-5).tanh() / 20000.)
            .collect();
        let base: f64 = x
            .iter()
            .map(|x| 2. * (1. - x * x).max(0.).sqrt() / PI)
            .sum();
        for (index, &x) in x.iter().enumerate() {
            let r = (1. - x * x).max(0.).sqrt();
            let n = 2048;
            let mut value = 0.;
            for j in 0..=n {
                let theta = -PI / 2. + PI * j as f64 / n as f64;
                let point = [x, r * theta.sin(), r * theta.cos()];
                let weight = if j == 0 || j == n {
                    1.
                } else if j % 2 == 1 {
                    4.
                } else {
                    2.
                };
                value += weight
                    * m.intensity_xyz(point).unwrap()
                    * map::limb_intensity(point[2], &u).unwrap()
                    * r
                    * theta.cos();
            }
            value *= PI / (3. * n as f64 * base * norm);
            assert!((k[index] - value).abs() < 2e-9, "{} {value}", k[index]);
        }
    }
}
