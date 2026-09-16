//! Native Rust port of selected starry numerical kernels.
//!
//! Coefficients are ordered by `n = l*l + l + m`; `Y00 = 1/pi`.
//! See README.md for the compatibility boundary of this in-progress port.
pub mod autodiff;
pub mod basis;
pub mod conic;
pub mod differentiation;
pub mod doppler;
pub mod doppler_derivatives;
pub mod elliptic;
mod ffi;
pub mod gradients;
pub mod imaging;
pub mod inference;
pub mod map;
pub mod matrix;
pub mod oblate;
pub mod oblate_derivatives;
pub mod occultation;
pub mod orbit;
mod oren_coefficients;
pub mod pixels;
mod polynomial_roots;
pub mod quadrature;
pub mod reflected;
pub mod reflected_derivatives;
pub mod rings;
pub mod rotation;
pub mod rv;
pub mod rv_derivatives;
mod session;
pub mod solver;
mod source_orbits;
pub mod specialized_jacobian;
pub mod surface;
pub mod system;
pub mod system_derivatives;
pub mod system_surface;

pub use map::Map;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub(crate) fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
