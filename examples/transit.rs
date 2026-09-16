use starry_rust::{Map, Result, occultation::Occultor};
fn main() -> Result<()> {
    let mut map = Map::new(3)?;
    map.set(1, 0, 0.2)?;
    map.set(2, 2, -0.05)?;
    println!("x,flux");
    for i in 0..101 {
        let x = -1.3 + 2.6 * i as f64 / 100.;
        println!(
            "{x:.8},{:.12}",
            map.flux(Some(Occultor::new(x, 0.3, 0.12)?))?
        );
    }
    Ok(())
}
