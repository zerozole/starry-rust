use starry_rust::{Map, pixels};
#[test]
fn surface_reconstruction_and_derivatives() {
    for degree in [0, 1, 3, 6] {
        let grid = pixels::grid(degree, 3).unwrap();
        let t = pixels::transforms(degree, &grid, 0.).unwrap();
        let y: Vec<_> = (0..(degree + 1).pow(2))
            .map(|i| (i as f64 + 0.2).sin())
            .collect();
        let intensity = t.forward.dot(&y).unwrap();
        let recovered = t.inverse.dot(&intensity).unwrap();
        for (a, b) in y.iter().zip(recovered) {
            assert!((a - b).abs() < 1e-9);
        }
        let dlat = t.latitude.dot(&y).unwrap();
        let dlon = t.longitude.dot(&y).unwrap();
        let mut map = Map::new(degree).unwrap();
        map.set_coefficients(&y).unwrap();
        for (i, &[lat, lon]) in grid.iter().enumerate() {
            assert!((intensity[i] - map.intensity(lat, lon).unwrap()).abs() < 1e-10);
            let h = 1e-6;
            if lat.abs() < 1.56 {
                let finite = (map.intensity(lat + h, lon).unwrap()
                    - map.intensity(lat - h, lon).unwrap())
                    / (2. * h);
                assert!((dlat[i] - finite).abs() < 1e-7);
            }
            let finite = (map.intensity(lat, lon + h).unwrap()
                - map.intensity(lat, lon - h).unwrap())
                / (2. * h);
            assert!((dlon[i] - finite).abs() < 1e-7);
        }
    }
}
