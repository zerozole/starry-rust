use starry_rust::{basis, rotation};
#[test]
fn recurrence_and_rotation_projection() {
    let point = [0.3, 0.4, (0.75_f64).sqrt()];
    for degree in [0, 1, 2, 5, 8, 12] {
        let polynomial = basis::a1(degree)
            .unwrap()
            .left_dot(&basis::polynomial(degree, point).unwrap())
            .unwrap();
        let stable = basis::harmonics(degree, point).unwrap();
        for (a, b) in polynomial.iter().zip(stable) {
            assert!((a - b).abs() < 2e-12, "{degree}: {a} {b}");
        }
    }
    for degree in [1, 5, 10, 15, 20] {
        let y: Vec<_> = (0..(degree + 1) * (degree + 1))
            .map(|i| (i as f64).sin())
            .collect();
        let axis = [0.2, 0.5, -0.7];
        let angle = 0.9;
        let rotated = rotation::projected_coefficients(degree, &y, axis, angle).unwrap();
        let restored = rotation::projected_coefficients(degree, &rotated, axis, -angle).unwrap();
        for (a, b) in restored.iter().zip(&y) {
            assert!((a - b).abs() < 2e-13, "degree {degree}: {a} {b}");
        }
        for l in 0..=degree {
            let power = |v: &[f64]| {
                v[l * l..(l + 1) * (l + 1)]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
            };
            assert!((power(&y) - power(&rotated)).abs() < 2e-12);
        }
        let inverse = rotation::transpose(rotation::axis_angle(axis, angle).unwrap());
        let a = basis::harmonics(degree, rotation::apply(inverse, point)).unwrap();
        let b = basis::harmonics(degree, point).unwrap();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        assert!((dot(&a, &y) - dot(&b, &rotated)).abs() < 2e-12);
    }
}

#[test]
fn tangent_derivatives_including_poles() {
    for degree in [1, 5, 12, 20] {
        for point in [[0., 0., 1.], [0., 0., -1.], [0.3, 0.4, (0.75_f64).sqrt()]] {
            let axis = [0., 1., 0.];
            let tangent = [point[2], 0., -point[0]];
            let derivative = basis::harmonic_directional(degree, point, tangent).unwrap();
            let h = 1e-6;
            let a = basis::harmonics(
                degree,
                rotation::apply(rotation::axis_angle(axis, h).unwrap(), point),
            )
            .unwrap();
            let b = basis::harmonics(
                degree,
                rotation::apply(rotation::axis_angle(axis, -h).unwrap(), point),
            )
            .unwrap();
            for ((a, b), d) in a.iter().zip(b).zip(derivative) {
                assert!(
                    ((a - b) / (2. * h) - d).abs() < 2e-8,
                    "degree {degree}: {} {d}",
                    (a - b) / (2. * h)
                );
            }
        }
    }
}
