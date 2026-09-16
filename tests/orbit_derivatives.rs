use starry_rust::orbit::Orbit;
fn orbit(p: [f64; 8]) -> Orbit {
    Orbit {
        semimajor_axis: p[0],
        period: p[1],
        eccentricity: p[2],
        inclination: p[3],
        omega: p[4],
        ascending_node: p[5],
        transit_epoch: p[6],
    }
}
#[test]
fn orbital_jacobian_hessian_and_time_identities() {
    for e in [0.2, 0.8, 0.97] {
        let p = [5., 3.2, e, 1.1, 0.7, 0.3, 0.2, 1.4];
        let base = orbit(p).state_derivatives(p[7]).unwrap();
        let (position, velocity) = orbit(p).state(p[7]).unwrap();
        for k in 0..3 {
            assert!((base[k].value - position[k]).abs() < 1e-13);
            assert!((base[k + 3].value - velocity[k]).abs() < 1e-13);
            assert!((base[k].gradient[7] - velocity[k]).abs() < 1e-12);
            for j in 0..8 {
                assert!((base[k].hessian[7][j] - base[k + 3].gradient[j]).abs() < 1e-9);
            }
        }
        for i in 0..8 {
            let h = 1e-6;
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            let plus = orbit(a).state_derivatives(a[7]).unwrap();
            let minus = orbit(b).state_derivatives(b[7]).unwrap();
            for k in 0..6 {
                let fd = (plus[k].value - minus[k].value) / (2. * h);
                assert!(
                    (fd - base[k].gradient[i]).abs() < 1e-6 * (1. + fd.abs()),
                    "gradient e={e} k={k} i={i}"
                );
                for j in 0..8 {
                    let fd = (plus[k].gradient[j] - minus[k].gradient[j]) / (2. * h);
                    assert!(
                        (fd - base[k].hessian[i][j]).abs() < 2e-5 * (1. + fd.abs()),
                        "Hessian e={e} k={k} i={i} j={j}: {fd} vs {}",
                        base[k].hessian[i][j]
                    );
                    assert!((base[k].hessian[i][j] - base[k].hessian[j][i]).abs() < 1e-10);
                }
            }
        }
    }
}
