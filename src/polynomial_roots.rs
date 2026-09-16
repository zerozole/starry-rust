//! Real polynomial roots on a bounded interval, isolated by derivative roots.
//! Used for circle/terminator intersection cuts, including tangent roots.
pub(crate) fn evaluate(p: &[f64], x: f64) -> f64 {
    p.iter().rev().fold(0., |a, b| a * x + b)
}
pub(crate) fn real_roots(p: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let mut p = p.to_vec();
    while p.len() > 1 && p.last() == Some(&0.) {
        p.pop();
    }
    if p.len() < 2 {
        return vec![];
    }
    if p.len() == 2 {
        let x = -p[0] / p[1];
        return if x >= lo && x <= hi { vec![x] } else { vec![] };
    }
    let dp: Vec<_> = p
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| i as f64 * c)
        .collect();
    let mut cuts = vec![lo];
    cuts.extend(real_roots(&dp, lo, hi));
    cuts.push(hi);
    let tol = 64. * f64::EPSILON * p.iter().map(|v| v.abs()).sum::<f64>();
    let mut roots = vec![];
    for &x in &cuts {
        if evaluate(&p, x).abs() <= tol {
            roots.push(x);
        }
    }
    for pair in cuts.windows(2) {
        let (mut a, mut b) = (pair[0], pair[1]);
        let mut fa = evaluate(&p, a);
        let fb = evaluate(&p, b);
        if fa * fb >= 0. {
            continue;
        }
        for _ in 0..70 {
            let m = (a + b) / 2.;
            let fm = evaluate(&p, m);
            if fm == 0. {
                a = m;
                b = m;
                break;
            }
            if fa.signum() == fm.signum() {
                a = m;
                fa = fm;
            } else {
                b = m;
            }
        }
        roots.push((a + b) / 2.);
    }
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
    roots
}
