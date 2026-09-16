//! C ABI for the small dependency-free Python ctypes wrapper.
use crate::{Map, occultation::Occultor};

/// Evaluate a flux design row. Returns 0 on success, -1 on invalid input/error.
/// # Safety
/// `out` must point to at least `len` writable doubles, with no concurrent access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_flux_design(
    degree: u32,
    x: f64,
    y: f64,
    r: f64,
    out: *mut f64,
    len: usize,
) -> i32 {
    if out.is_null()
        || degree > crate::basis::MAX_DEGREE as u32
        || len != (degree as usize + 1).pow(2)
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let map = Map::new(degree as usize)?;
        let occ = Occultor::new(x, y, r)?;
        map.flux_design(Some(occ))
    });
    match result {
        Ok(Ok(v)) => {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
            }
            0
        }
        _ => -1,
    }
}

/// # Safety
/// `out` must point to `len` writable doubles, with no concurrent access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_intensity_design(
    degree: u32,
    lat: f64,
    lon: f64,
    out: *mut f64,
    len: usize,
) -> i32 {
    if out.is_null()
        || degree > crate::basis::MAX_DEGREE as u32
        || len != (degree as usize + 1).pow(2)
        || !lat.is_finite()
        || lat.abs() > std::f64::consts::FRAC_PI_2
        || !lon.is_finite()
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        Map::new(degree as usize)?.intensity_design([
            lat.cos() * lon.sin(),
            lat.sin(),
            lat.cos() * lon.cos(),
        ])
    });
    match result {
        Ok(Ok(v)) => {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
            }
            0
        }
        _ => -1,
    }
}

/// # Safety
/// `coefficients` must point to `len` readable doubles, and `out` to `len`
/// writable doubles. Input and output may alias. No concurrent access permitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_rotate(
    degree: u32,
    coefficients: *const f64,
    len: usize,
    x: f64,
    y: f64,
    z: f64,
    angle: f64,
    out: *mut f64,
) -> i32 {
    if coefficients.is_null()
        || out.is_null()
        || degree > crate::basis::MAX_DEGREE as u32
        || len != (degree as usize + 1).pow(2)
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let coeffs = unsafe { std::slice::from_raw_parts(coefficients, len) }.to_vec();
        crate::rotation::coefficients(degree as usize, &coeffs, [x, y, z], angle)
    });
    match result {
        Ok(Ok(v)) => {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), out, len);
            }
            0
        }
        _ => -1,
    }
}

/// # Safety
/// `coefficients` must point to `len` readable doubles and `out` to three
/// writable doubles. No concurrent access permitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_flux_gradient(
    degree: u32,
    coefficients: *const f64,
    len: usize,
    x: f64,
    y: f64,
    r: f64,
    out: *mut f64,
) -> i32 {
    if coefficients.is_null()
        || out.is_null()
        || degree > crate::basis::MAX_DEGREE as u32
        || len != (degree as usize + 1).pow(2)
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let coeffs = unsafe { std::slice::from_raw_parts(coefficients, len) };
        let mut map = Map::new(degree as usize)?;
        map.set_coefficients(coeffs)?;
        map.flux_gradient(Occultor::new(x, y, r)?)
    });
    match result {
        Ok(Ok(v)) => {
            unsafe {
                std::ptr::copy_nonoverlapping(v.as_ptr(), out, 3);
            }
            0
        }
        _ => -1,
    }
}

/// # Safety
/// `coefficients` and `u` must point to `len` and `udeg` readable doubles,
/// respectively, and `out` to one writable double. `u` may be null if udeg=0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_limb_flux(
    degree: u32,
    coefficients: *const f64,
    len: usize,
    u: *const f64,
    udeg: usize,
    x: f64,
    y: f64,
    r: f64,
    out: *mut f64,
) -> i32 {
    if coefficients.is_null()
        || out.is_null()
        || (u.is_null() && udeg > 0)
        || degree > crate::basis::MAX_DEGREE as u32
        || udeg > crate::basis::MAX_DEGREE - degree as usize
        || len != (degree as usize + 1).pow(2)
    {
        return -1;
    }
    let result = std::panic::catch_unwind(|| {
        let coeffs = unsafe { std::slice::from_raw_parts(coefficients, len) };
        let limb = if udeg == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(u, udeg) }
        };
        let mut map = Map::new(degree as usize)?;
        map.set_coefficients(coeffs)?;
        map.flux_limb_darkened(limb, Some(Occultor::new(x, y, r)?))
    });
    match result {
        Ok(Ok(v)) => {
            unsafe {
                *out = v;
            }
            0
        }
        _ => -1,
    }
}
