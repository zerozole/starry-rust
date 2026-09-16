use starry_rust::{
    Map,
    system::{Body, System},
};
#[test]
fn exposure_quadrature_moments() {
    let mut s = System::new(Body::new(Map::new(0).unwrap(), 1., 1.).unwrap(), vec![]).unwrap();
    s.exposure = 2.;
    s.exposure_samples = 4;
    for order in 0..3 {
        s.exposure_order = order;
        let stencil = s.exposure_stencil().unwrap();
        let moment = |n: i32| stencil.iter().map(|(t, w)| w * t.powi(n)).sum::<f64>();
        assert!((moment(0) - 1.).abs() < 1e-15);
        assert!(moment(1).abs() < 1e-15);
        if order == 2 {
            assert!((moment(2) - 1. / 3.).abs() < 1e-15);
            assert!(moment(3).abs() < 1e-15);
        }
    }
    s.exposure_samples = 1;
    assert_eq!(s.exposure_stencil().unwrap(), vec![(0., 1.)]);
    s.exposure_order = 3;
    assert!(s.exposure_stencil().is_err());
}
