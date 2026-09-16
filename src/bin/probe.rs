//! Text protocol for independent numerical validation, not a public CLI API.
use starry_rust::{Map, Result, basis, elliptic, occultation::Occultor, rotation};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = &args[1];
    let d: usize = args[2].parse().unwrap();
    let number = |i: usize| args[i].parse::<f64>().unwrap();
    let values = match mode.as_str() {
        "rv-filter-jacobian" => {
            starry_rust::rv_derivatives::filters(number(3), number(4), number(5), number(6))?
                .iter()
                .flat_map(|m| m.coefficients().iter().copied())
                .collect()
        }
        "oblate" => {
            let b = number(3);
            let r = number(4);
            let f = number(5);
            let theta = number(6);
            let n = (d + 1).pow(2);
            let occ = Some(Occultor::new(b * theta.sin(), b * theta.cos(), r)?);
            let mut map = Map::new(d)?;
            let mut out = vec![];
            for i in 0..n {
                let mut y = vec![0.; n];
                y[i] = 1.;
                map.set_coefficients(&y)?;
                out.push(starry_rust::oblate::flux(
                    &map,
                    f,
                    std::f64::consts::FRAC_PI_2,
                    0.,
                    0.,
                    &[],
                    None,
                    occ,
                    false,
                )?);
            }
            out
        }
        "reflected" => {
            let b = number(3);
            let theta = number(4);
            let bo = number(5);
            let ro = number(6);
            let rough = number(7);
            let source = [
                -(1. - b * b).sqrt() * theta.sin(),
                (1. - b * b).sqrt() * theta.cos(),
                -b,
            ];
            let occ = if ro == 0. {
                None
            } else {
                Some(Occultor::new(0., bo, ro)?)
            };
            let mut map = Map::new(d)?;
            let n = (d + 1).pow(2);
            let mut out = vec![];
            for i in 0..n {
                let mut y = vec![0.; n];
                y[i] = 1.;
                map.set_coefficients(&y)?;
                out.push(starry_rust::reflected::flux(&map, source, rough, occ)?);
            }
            out
        }
        "basis" => {
            let mut v = basis::a1(d)?.data;
            v.extend(basis::a2_inverse(d)?.data);
            v.extend(basis::disk_moments(d)?);
            v
        }
        "flux" => Map::new(d)?.flux_design_numerical(
            Some(Occultor::new(0., number(3), number(4))?),
            Default::default(),
        )?,
        "rings" => starry_rust::rings::flux_design(
            d,
            Some(Occultor::new(0., number(3), number(4))?),
            &[],
            Default::default(),
        )?,
        "analytic" => starry_rust::solver::flux_design(
            &basis::a1(d)?,
            d,
            Occultor::new(0., number(3), number(4))?,
        )?,
        "gradient" => {
            let rows = starry_rust::gradients::flux_design(
                d,
                &basis::a1(d)?,
                Occultor::new(0., number(3), number(4))?,
                1e-12,
            )?;
            rows[1].iter().chain(&rows[2]).copied().collect()
        }
        "limb" => {
            let n = (d + 1).pow(2);
            let mut map = Map::new(d)?;
            let mut row = vec![0.; n];
            for j in 0..n {
                let mut y = vec![0.; n];
                y[j] = 1.;
                map.set_coefficients(&y)?;
                row[j] = map.flux_limb_darkened(
                    &[0.4, 0.2],
                    Some(Occultor::new(0., number(3), number(4))?),
                )?;
            }
            row
        }
        "rotation" => {
            let n = (d + 1).pow(2);
            let mut v = vec![0.; n * n];
            for j in 0..n {
                let mut y = vec![0.; n];
                y[j] = 1.;
                let r =
                    rotation::coefficients(d, &y, [number(3), number(4), number(5)], number(6))?;
                for i in 0..n {
                    v[i * n + j] = r[i];
                }
            }
            v
        }
        "cel" => vec![elliptic::cel(
            number(3),
            (1. - number(3)).sqrt(),
            number(4),
            number(5),
            number(6),
        )?],
        _ => panic!("unknown probe mode"),
    };
    for v in values {
        print!("{v:.17e} ");
    }
    println!();
    Ok(())
}
