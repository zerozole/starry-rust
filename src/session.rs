//! Persistent C ABI map handles for Python. Handles are integer IDs, not pointers.
use crate::{Error, Map, Result, occultation::Occultor, require};
use std::{
    collections::BTreeMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
static MAPS: OnceLock<Mutex<BTreeMap<u64, Map>>> = OnceLock::new();
static NEXT: AtomicU64 = AtomicU64::new(1);

/// # Safety
/// parameters holds rows*8 values [a,period,e,inc,omega,node,epoch,time].
/// output holds rows*6*73 doubles, each state component followed by 8 first
/// derivatives and 64 second derivatives. No overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_orbit_derivatives(
    parameters: *const f64,
    rows: usize,
    out: *mut f64,
) -> i32 {
    if parameters.is_null() || out.is_null() || rows == 0 || rows > 10000 {
        return -1;
    }
    status(|| {
        let parameters = unsafe { std::slice::from_raw_parts(parameters, rows * 8) };
        let mut values = Vec::with_capacity(rows * 6 * 73);
        for p in parameters.chunks_exact(8) {
            let orbit = crate::orbit::Orbit {
                semimajor_axis: p[0],
                period: p[1],
                eccentricity: p[2],
                inclination: p[3],
                omega: p[4],
                ascending_node: p[5],
                transit_epoch: p[6],
            };
            for jet in orbit.state_derivatives(p[7])? {
                values.push(jet.value);
                values.extend(jet.gradient);
                values.extend(jet.hessian.into_iter().flatten());
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}
fn registry() -> &'static Mutex<BTreeMap<u64, Map>> {
    MAPS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// # Safety
/// p contains np map-batch doubles; out has len writable doubles matching
/// specialized_jacobian's documented layout. Buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_specialized_jacobian(
    id: u64,
    mode: u32,
    p: *const f64,
    np: usize,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null() || out.is_null() || !(22..=42).contains(&np) {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, np) };
        let map = get(id)?;
        let view = image_view(mode, p, 1)?;
        let occ = Occultor::new(p[3], p[4], p[6])?;
        let occ = if p[5] > 0. { Some(occ) } else { None };
        let result = match mode {
            1 => crate::specialized_jacobian::reflection(&map, &view, occ)?,
            2 => crate::rv_derivatives::jacobian(&map, &view, p[11], p[12], occ)?,
            3 => crate::specialized_jacobian::oblate(&map, &view, occ)?,
            _ => return Err(Error("invalid specialized Jacobian mode".into())),
        };
        require(result.len() == len, "invalid specialized Jacobian length")?;
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// p has 4+nu doubles [inc,veq,spacing,half_width,u...], rest has ns;
/// output has (5+nu)*(ns-2*half_width+1) doubles. Buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_doppler_jacobian(
    id: u64,
    p: *const f64,
    nu: usize,
    phase: f64,
    rest: *const f64,
    ns: usize,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null() || rest.is_null() || out.is_null() || nu > 20 || ns > 1_000_000 {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, 4 + nu) };
        require(
            p[3].is_finite() && p[3] >= 1. && p[3] <= 100_000. && p[3].fract() == 0.,
            "invalid kernel width",
        )?;
        let result = crate::doppler_derivatives::spectrum(
            &get(id)?,
            p[0],
            phase,
            p[1],
            &p[4..],
            p[2],
            p[3] as usize,
            unsafe { std::slice::from_raw_parts(rest, ns) },
        )?;
        require(
            result.data.len() == len,
            "invalid Doppler derivative output length",
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(result.data.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// p has np doubles in map-batch layout. Output has ten doubles matching
/// oblate_derivatives::flux. Input/output buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_oblate_jacobian(
    id: u64,
    p: *const f64,
    np: usize,
    out: *mut f64,
) -> i32 {
    if p.is_null() || out.is_null() || !(22..=42).contains(&np) {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, np) };
        let map = get(id)?;
        let view = image_view(3, p, 1)?;
        let (f, gravity, normalized) = view
            .oblateness
            .ok_or(Error("missing oblate parameters".into()))?;
        let occ = Occultor::new(p[3], p[4], p[6])?;
        let result = crate::oblate_derivatives::flux(
            &map,
            f,
            p[1],
            p[2],
            p[0],
            &p[22..],
            gravity,
            if p[5] > 0. { Some(occ) } else { None },
            normalized,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), out, result.len());
        }
        Ok(())
    })
}

/// # Safety
/// p holds np doubles in map-batch layout. Output has six doubles:
/// flux, source x/y/z, roughness and source-radius derivatives. No overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_reflection_jacobian(
    id: u64,
    p: *const f64,
    np: usize,
    out: *mut f64,
) -> i32 {
    if p.is_null() || out.is_null() || !(22..=42).contains(&np) {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, np) };
        let map = get(id)?;
        let value = evaluate(&map, 1, p)?;
        let projected = map.projected(p[1], p[2], p[0])?.limb_filtered(&p[22..])?;
        let occultors = if p[5] > 0. {
            vec![Occultor::new(p[3], p[4], p[6])?]
        } else {
            vec![]
        };
        let gradient = crate::reflected_derivatives::extended(
            &projected,
            [p[7], p[8], p[9]],
            p[20],
            p[21] as usize,
            p[10],
            &occultors,
        )?;
        unsafe {
            *out = value;
            std::ptr::copy_nonoverlapping(gradient.as_ptr(), out.add(1), 5);
        }
        Ok(())
    })
}

/// # Safety
/// data has rows*cols doubles, output has rank*(rows+cols); no overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_low_rank(
    data: *const f64,
    rows: usize,
    cols: usize,
    rank: usize,
    out: *mut f64,
) -> i32 {
    if data.is_null()
        || out.is_null()
        || rows.checked_mul(cols).is_none_or(|n| n > 16_000_000)
        || rows.min(cols) > 2048
        || rank == 0
        || rank > rows.min(cols)
    {
        return -1;
    }
    status(|| {
        let matrix = crate::matrix::Matrix {
            rows,
            cols,
            data: unsafe { std::slice::from_raw_parts(data, rows * cols) }.to_vec(),
        };
        let (a, b) = matrix.low_rank(rank)?;
        unsafe {
            std::ptr::copy_nonoverlapping(a.data.as_ptr(), out, a.data.len());
            std::ptr::copy_nonoverlapping(b.data.as_ptr(), out.add(a.data.len()), b.data.len());
        }
        Ok(())
    })
}

/// # Safety
/// normal holds n*n doubles, rhs n, output n+2. Buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_l1(
    normal: *const f64,
    rhs: *const f64,
    n: usize,
    lambda: f64,
    iterations: usize,
    epsilon: f64,
    tolerance: f64,
    out: *mut f64,
) -> i32 {
    if normal.is_null()
        || rhs.is_null()
        || out.is_null()
        || n == 0
        || n > 4096
        || iterations > 100000
    {
        return -1;
    }
    status(|| {
        let normal = crate::matrix::Matrix {
            rows: n,
            cols: n,
            data: unsafe { std::slice::from_raw_parts(normal, n * n) }.to_vec(),
        };
        let result = crate::inference::l1(
            &normal,
            unsafe { std::slice::from_raw_parts(rhs, n) },
            lambda,
            iterations,
            epsilon,
            tolerance,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(result.weights.as_ptr(), out, n);
            *out.add(n) = result.iterations as f64;
            *out.add(n + 1) = if result.converged { 1. } else { 0. };
        }
        Ok(())
    })
}

/// Specialized flux derivatives with respect to foreground x, y and radius.
/// # Safety
/// parameters contains rows*width doubles in map-batch layout; output contains
/// rows*3 writable doubles. Input and output buffers must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_specialized_gradients(
    id: u64,
    mode: u32,
    parameters: *const f64,
    rows: usize,
    width: usize,
    out: *mut f64,
) -> i32 {
    if parameters.is_null()
        || out.is_null()
        || !(22..=42).contains(&width)
        || rows == 0
        || rows > 1_000_000
    {
        return -1;
    }
    status(|| {
        require(mode == 1 || mode == 3, "invalid specialized gradient mode")?;
        let map = get(id)?;
        let parameters = unsafe { std::slice::from_raw_parts(parameters, rows * width) };
        let mut values = Vec::with_capacity(rows * 3);
        for p in parameters.chunks_exact(width) {
            let view = image_view(mode, p, 1)?;
            let occ = Occultor::new(p[3], p[4], p[6])?;
            let g = if p[5] > 0. {
                crate::imaging::occultor_gradient(&map, &view, occ)?
            } else {
                [0.; 3]
            };
            values.extend(g);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}
fn get(id: u64) -> Result<Map> {
    registry()
        .lock()
        .map_err(|_| Error("map registry poisoned".into()))?
        .get(&id)
        .cloned()
        .ok_or(Error("unknown map handle".into()))
}

/// Native Doppler linear operator for one component. mode=0 map design,
/// mode=1 map continuum design, mode=2 fixed-map rest-spectrum design.
/// # Safety
/// p holds 4+nu doubles [inc,veq,spacing,half_width,u...], phases holds nt,
/// spectrum holds ns, and out holds len writable doubles. Buffers do not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_doppler_design(
    id: u64,
    p: *const f64,
    nu: usize,
    phases: *const f64,
    nt: usize,
    spectrum: *const f64,
    ns: usize,
    mode: u32,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null()
        || phases.is_null()
        || spectrum.is_null()
        || out.is_null()
        || nu > 20
        || nt == 0
        || nt > 10000
        || ns > 1_000_000
    {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, 4 + nu) };
        require(
            p[3].is_finite() && p[3] >= 1. && p[3] <= 100_000. && p[3].fract() == 0.,
            "invalid kernel width",
        )?;
        let model = crate::doppler::DopplerMap {
            components: vec![crate::doppler::Component {
                map: get(id)?,
                spectrum: unsafe { std::slice::from_raw_parts(spectrum, ns) }.to_vec(),
            }],
            inclination: p[0],
            veq: p[1],
            log_spacing: p[2],
            half_width: p[3] as usize,
            limb_darkening: p[4..].to_vec(),
        };
        let phases = unsafe { std::slice::from_raw_parts(phases, nt) };
        let design = match mode {
            0 => model.map_design(phases)?.0,
            1 => model.map_design(phases)?.1,
            2 => model.spectrum_design(phases, false)?,
            _ => return Err(Error("invalid Doppler design mode".into())),
        };
        require(design.data.len() == len, "invalid Doppler output length")?;
        unsafe {
            std::ptr::copy_nonoverlapping(design.data.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// p holds 4+nu doubles [inc,veq,spacing,half_width,u...]; phases has nt;
/// values has nv; output has len. ns is the padded rest spectrum length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_spectral_map_dot(
    id: u64,
    p: *const f64,
    nu: usize,
    phases: *const f64,
    nt: usize,
    ns: usize,
    values: *const f64,
    nv: usize,
    transpose: u32,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null()
        || phases.is_null()
        || values.is_null()
        || out.is_null()
        || nu > 20
        || nt == 0
        || nt > 10000
        || ns > 1_000_000
        || nv > 32_000_000
        || transpose > 1
    {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, 4 + nu) };
        require(
            p[3].is_finite() && p[3] >= 1. && p[3] <= 100_000. && p[3].fract() == 0.,
            "invalid kernel width",
        )?;
        let model = crate::doppler::DopplerMap {
            components: vec![crate::doppler::Component {
                map: get(id)?,
                spectrum: vec![0.; ns],
            }],
            inclination: p[0],
            veq: p[1],
            log_spacing: p[2],
            half_width: p[3] as usize,
            limb_darkening: p[4..].to_vec(),
        };
        let result = model.spectral_map_dot(
            unsafe { std::slice::from_raw_parts(phases, nt) },
            unsafe { std::slice::from_raw_parts(values, nv) },
            transpose != 0,
        )?;
        require(result.len() == len, "invalid spectral map output length")?;
        unsafe {
            std::ptr::copy_nonoverlapping(result.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// input holds ni wavelengths, output holds no wavelengths, out holds ni*no
/// writable doubles for a row-major interpolation operator, without overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_interpolation(
    input: *const f64,
    ni: usize,
    output: *const f64,
    no: usize,
    out: *mut f64,
) -> i32 {
    if input.is_null()
        || output.is_null()
        || out.is_null()
        || ni.checked_mul(no).is_none_or(|n| n > 32_000_000)
    {
        return -1;
    }
    status(|| {
        let operator = crate::doppler::interpolation(
            unsafe { std::slice::from_raw_parts(input, ni) },
            unsafe { std::slice::from_raw_parts(output, no) },
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(operator.data.as_ptr(), out, operator.data.len());
        }
        Ok(())
    })
}
thread_local! {static LAST_ERROR:std::cell::RefCell<String>=const{std::cell::RefCell::new(String::new())};}

/// # Safety
/// design: rows*cols; data and variance: rows; optional prior mean/covariance:
/// cols and cols*cols. Output: cols+cols*cols+2 doubles. No overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_linear_solve_diagonal(
    design: *const f64,
    rows: usize,
    cols: usize,
    data: *const f64,
    variance: *const f64,
    prior_mean: *const f64,
    prior_cov: *const f64,
    out: *mut f64,
) -> i32 {
    if design.is_null()
        || data.is_null()
        || variance.is_null()
        || out.is_null()
        || rows == 0
        || cols == 0
        || cols > 4096
        || rows.checked_mul(cols).is_none_or(|n| n > 32_000_000)
        || prior_mean.is_null() != prior_cov.is_null()
    {
        return -1;
    }
    status(|| {
        let matrix = crate::matrix::Matrix {
            rows,
            cols,
            data: unsafe { std::slice::from_raw_parts(design, rows * cols) }.to_vec(),
        };
        let prior = if prior_mean.is_null() {
            None
        } else {
            Some(crate::inference::GaussianPrior {
                mean: unsafe { std::slice::from_raw_parts(prior_mean, cols) }.to_vec(),
                covariance: crate::matrix::Matrix {
                    rows: cols,
                    cols,
                    data: unsafe { std::slice::from_raw_parts(prior_cov, cols * cols) }.to_vec(),
                },
            })
        };
        let (solution, marginal) = crate::inference::solve_diagonal(
            &matrix,
            unsafe { std::slice::from_raw_parts(data, rows) },
            unsafe { std::slice::from_raw_parts(variance, rows) },
            prior.as_ref(),
        )?;
        let mut values = solution.mean;
        values.extend(solution.covariance.data);
        values.extend([solution.log_likelihood, marginal.unwrap_or(f64::NAN)]);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}
fn status(f: impl FnOnce() -> Result<()> + std::panic::UnwindSafe) -> i32 {
    LAST_ERROR.with(|s| s.borrow_mut().clear());
    match std::panic::catch_unwind(f) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            LAST_ERROR.with(|s| *s.borrow_mut() = e.0);
            -1
        }
        Err(_) => {
            LAST_ERROR.with(|s| *s.borrow_mut() = "native operation panicked".into());
            -1
        }
    }
}

/// Render an oriented emitted map. Projection: 0 orthographic, 1 rectangular,
/// 2 Mollweide. Angles are radians.
/// # Safety
/// out must hold width*height writable doubles, with no overlapping inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_render(
    id: u64,
    width: usize,
    height: usize,
    projection: u32,
    inclination: f64,
    obliquity: f64,
    phase: f64,
    out: *mut f64,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    status(|| {
        let projection = match projection {
            0 => crate::surface::Projection::Orthographic,
            1 => crate::surface::Projection::Rectangular,
            2 => crate::surface::Projection::Mollweide,
            _ => return Err(Error("invalid image projection".into())),
        };
        let map = get(id)?.projected(inclination, obliquity, phase)?;
        let values = crate::surface::render(&map, width, height, projection)?;
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// Fit a sampled surface and replace the handle's map on success.
/// # Safety
/// samples holds n*4 doubles in latitude, longitude, intensity, weight order;
/// out holds (degree+1)^2 writable doubles. Arrays must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_fit(
    id: u64,
    samples: *const f64,
    n: usize,
    ridge: f64,
    out: *mut f64,
) -> i32 {
    if samples.is_null() || out.is_null() || n == 0 || n > 1_000_000 {
        return -1;
    }
    status(|| {
        let degree = get(id)?.degree();
        let samples = unsafe { std::slice::from_raw_parts(samples, n * 4) };
        let columns: Vec<Vec<f64>> = (0..4)
            .map(|j| samples.chunks_exact(4).map(|r| r[j]).collect())
            .collect();
        let map = crate::surface::fit_samples(
            degree,
            &columns[0],
            &columns[1],
            &columns[2],
            &columns[3],
            ridge,
        )?;
        let values = map.coefficients().to_vec();
        registry()
            .lock()
            .map_err(|_| Error("map registry poisoned".into()))?
            .insert(id, map);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// Convolve a padded rest spectrum or return a broadening kernel (nspectrum=0).
/// # Safety
/// u holds nu doubles (may be null when nu=0), spectrum holds nspectrum doubles
/// (may be null when nspectrum=0), out holds nspectrum-2*half_width doubles for
/// convolution or 2*half_width+1 doubles for a kernel. No overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_doppler(
    id: u64,
    inclination: f64,
    phase: f64,
    veq: f64,
    u: *const f64,
    nu: usize,
    spacing: f64,
    half_width: usize,
    spectrum: *const f64,
    nspectrum: usize,
    normalize: u32,
    out: *mut f64,
) -> i32 {
    if out.is_null()
        || nu > 20
        || (nu > 0 && u.is_null())
        || nspectrum > 16_000_000
        || (nspectrum > 0 && spectrum.is_null())
    {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        let u = if nu == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(u, nu) }
        };
        let values = if nspectrum == 0 {
            crate::doppler::kernel(&map, inclination, phase, veq, u, spacing, half_width)?
        } else {
            let spectrum = unsafe { std::slice::from_raw_parts(spectrum, nspectrum) }.to_vec();
            crate::doppler::DopplerMap {
                components: vec![crate::doppler::Component { map, spectrum }],
                log_spacing: spacing,
                half_width,
                inclination,
                veq,
                limb_darkening: u.to_vec(),
            }
            .spectrum(phase, normalize != 0)?
        };
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn starry_map_create(degree: u32) -> u64 {
    let mut id = 0;
    let result = std::panic::catch_unwind(|| -> Result<u64> {
        let map = Map::new(degree as usize)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        registry()
            .lock()
            .map_err(|_| Error("map registry poisoned".into()))?
            .insert(id, map);
        Ok(id)
    });
    if let Ok(Ok(value)) = result {
        id = value;
    }
    id
}
#[unsafe(no_mangle)]
pub extern "C" fn starry_map_destroy(id: u64) {
    if let Ok(mut maps) = registry().lock() {
        maps.remove(&id);
    }
}

/// # Safety
/// `out` must have `capacity` writable bytes. Returns required length including NUL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_last_error(out: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|s| {
        let s = s.borrow();
        if !out.is_null() && capacity > 0 {
            let n = s.len().min(capacity - 1);
            unsafe {
                std::ptr::copy_nonoverlapping(s.as_ptr(), out, n);
                *out.add(n) = 0;
            }
        }
        s.len() + 1
    })
}
/// # Safety
/// `coefficients` must point to `len` readable doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_update(
    id: u64,
    coefficients: *const f64,
    len: usize,
    amplitude: f64,
) -> i32 {
    if coefficients.is_null() {
        return -1;
    }
    status(|| {
        let mut map = get(id)?;
        require(
            len == map.coefficients().len(),
            "coefficient shape mismatch",
        )?;
        let y = unsafe { std::slice::from_raw_parts(coefficients, len) };
        map.set_coefficients(y)?;
        map.set_amplitude(amplitude)?;
        registry()
            .lock()
            .map_err(|_| Error("map registry poisoned".into()))?
            .insert(id, map);
        Ok(())
    })
}

/// Parameter layout: theta, inc, obl, xo, yo, zo, ro, xs, ys, zs, roughness,
/// veq, alpha, flattening, omega, beta, tpole, wavelength_meters, gravity_degree,
/// normalized, source_radius, source_npts, then limb coefficients. Angles in radians.
fn evaluate(map: &Map, mode: u32, p: &[f64]) -> Result<f64> {
    require(
        p.len() >= 22 && p.iter().all(|v| v.is_finite()),
        "invalid map parameter row",
    )?;
    let occ = if p[5] > 0. {
        Some(Occultor::new(p[3], p[4], p[6])?)
    } else {
        Occultor::new(p[3], p[4], p[6])?;
        None
    };
    let u = &p[22..];
    match mode {
        0 => map.projected(p[1], p[2], p[0])?.flux_limb_darkened(u, occ),
        1 => {
            require(
                p[21] >= 1. && p[21] <= 100_000. && p[21].fract() == 0.,
                "invalid source sample count",
            )?;
            crate::reflected::flux_extended(
                &map.projected(p[1], p[2], p[0])?.limb_filtered(u)?,
                [p[7], p[8], p[9]],
                p[20],
                p[21] as usize,
                p[10],
                &occ.into_iter().collect::<Vec<_>>(),
            )
        }
        2 => crate::rv::radial_velocity(map, p[1], p[2], p[0], p[11], p[12], u, occ),
        3 => {
            require(
                p[18] >= 0. && p[18] <= 20. && p[18].fract() == 0.,
                "invalid gravity degree",
            )?;
            let gravity = if p[18] > 0. {
                Some(crate::oblate::GravityDarkening {
                    degree: p[18] as usize,
                    omega: p[14],
                    flattening: p[13],
                    beta: p[15],
                    polar_temperature: p[16],
                    wavelength_meters: p[17],
                })
            } else {
                None
            };
            crate::oblate::flux(map, p[13], p[1], p[2], p[0], u, gravity, occ, p[19] != 0.)
        }
        _ => Err(Error("unknown map evaluation mode".into())),
    }
}

fn image_view(mode: u32, p: &[f64], flags: u32) -> Result<crate::imaging::View> {
    require(
        (22..=42).contains(&p.len()) && p.iter().all(|v| v.is_finite()),
        "invalid image parameters",
    )?;
    let mut view = crate::imaging::View {
        upstream_grid: flags & 16 != 0,
        inclination: p[1],
        obliquity: p[2],
        phase: p[0],
        limb: p[22..].to_vec(),
        ..Default::default()
    };
    match mode {
        0 => {}
        1 => {
            require(
                p[21] >= 1. && p[21] <= 100_000. && p[21].fract() == 0.,
                "invalid source sample count",
            )?;
            view.reflection = Some(crate::imaging::Reflection {
                source: [p[7], p[8], p[9]],
                radius: p[20],
                samples: p[21] as usize,
                roughness: p[10],
                illuminate: flags & 1 != 0,
                exact: flags & 2 != 0,
            });
        }
        2 => {
            if flags & 8 != 0 {
                view.velocity = Some((p[11], p[12]));
            }
        }
        3 => {
            require(
                p[18] >= 0. && p[18] <= 20. && p[18].fract() == 0.,
                "invalid gravity degree",
            )?;
            let gravity = if p[18] > 0. {
                Some(crate::oblate::GravityDarkening {
                    degree: p[18] as usize,
                    omega: p[14],
                    flattening: p[13],
                    beta: p[15],
                    polar_temperature: p[16],
                    wavelength_meters: p[17],
                })
            } else {
                None
            };
            view.oblateness = Some((p[13], gravity, p[19] != 0.));
        }
        _ => return Err(Error("invalid imaging mode".into())),
    }
    Ok(view)
}

/// # Safety
/// p holds np input doubles; output has len doubles. op0: bounded minimum,
/// p=[latmin,latmax,lonmin,lonmax,oversample,ntries], output=5 values.
/// op1: limb coefficients, output=physical flag. op2: lo,hi,ascending polynomial
/// coefficients, output=root count. No overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_surface_operation(
    id: u64,
    op: u32,
    p: *const f64,
    np: usize,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null() || out.is_null() || np > 102 {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(p, np) };
        require(p.iter().all(|v| v.is_finite()), "invalid surface arguments")?;
        let values = match op {
            0 => {
                require(
                    np == 6
                        && p[4] >= 1.
                        && p[4] <= 20.
                        && p[4].fract() == 0.
                        && p[5] >= 1.
                        && p[5] <= 100.
                        && p[5].fract() == 0.,
                    "invalid minimizer parameters",
                )?;
                let minimum = crate::surface::minimize(
                    &get(id)?,
                    [[p[0], p[1]], [p[2], p[3]]],
                    p[4] as usize,
                    p[5] as usize,
                )?;
                vec![
                    minimum.latitude,
                    minimum.longitude,
                    minimum.value,
                    minimum.evaluations as f64,
                    if minimum.converged { 1. } else { 0. },
                ]
            }
            1 => vec![if crate::surface::limb_is_physical(p)? {
                1.
            } else {
                0.
            }],
            2 => {
                require(np >= 3 && p[0] < p[1], "invalid polynomial interval")?;
                vec![crate::polynomial_roots::real_roots(&p[2..], p[0], p[1]).len() as f64]
            }
            3 => {
                require(np == 1, "smoothing needs one argument")?;
                crate::surface::smoothing_weights(get(id)?.degree(), p[0])?
            }
            _ => return Err(Error("invalid surface operation".into())),
        };
        require(values.len() == len, "invalid surface output length")?;
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// count points contains interleaved latitude/longitude radians; output holds
/// 4*count*(degree+1)^2 values, forward, inverse, latitude and longitude matrices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_pixel_transforms(
    id: u64,
    points: *const f64,
    count: usize,
    ridge: f64,
    out: *mut f64,
    len: usize,
) -> i32 {
    if points.is_null() || out.is_null() || count == 0 || count > 10000 {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        require(
            len == 4 * count * (map.degree() + 1).pow(2),
            "invalid pixel output length",
        )?;
        let points: Vec<_> = unsafe { std::slice::from_raw_parts(points, count * 2) }
            .chunks_exact(2)
            .map(|p| [p[0], p[1]])
            .collect();
        let t = crate::pixels::transforms(map.degree(), &points, ridge)?;
        let mut offset = 0;
        for matrix in [t.forward, t.inverse, t.latitude, t.longitude] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    matrix.data.as_ptr(),
                    out.add(offset),
                    matrix.data.len(),
                );
            }
            offset += matrix.data.len();
        }
        Ok(())
    })
}

/// # Safety
/// If out is non-null it holds len doubles. A null output queries point count.
/// Returns zero on invalid arguments, otherwise the point count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_pixel_grid(
    degree: u32,
    oversample: usize,
    out: *mut f64,
    len: usize,
) -> usize {
    match crate::pixels::grid(degree as usize, oversample) {
        Ok(points) => {
            if !out.is_null() {
                if len != 2 * points.len() {
                    return 0;
                }
                for (i, p) in points.iter().enumerate() {
                    unsafe {
                        *out.add(2 * i) = p[0];
                        *out.add(2 * i + 1) = p[1];
                    }
                }
            }
            points.len()
        }
        Err(_) => 0,
    }
}

/// # Safety
/// p holds np parameter doubles; out holds width*height doubles. No overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_image(
    id: u64,
    mode: u32,
    p: *const f64,
    np: usize,
    width: usize,
    height: usize,
    projection: u32,
    flags: u32,
    out: *mut f64,
) -> i32 {
    if p.is_null() || out.is_null() || !(22..=42).contains(&np) {
        return -1;
    }
    status(|| {
        let view = image_view(mode, unsafe { std::slice::from_raw_parts(p, np) }, flags)?;
        let projection = match projection {
            0 => crate::surface::Projection::Orthographic,
            1 => crate::surface::Projection::Rectangular,
            2 => crate::surface::Projection::Mollweide,
            _ => return Err(Error("invalid image projection".into())),
        };
        let image = crate::imaging::render(&get(id)?, &view, width, height, projection)?;
        unsafe {
            std::ptr::copy_nonoverlapping(image.as_ptr(), out, image.len());
        }
        Ok(())
    })
}

/// # Safety
/// parameters holds [inclination,obliquity,phase,flattening]; output holds
/// 2*width*height doubles, interleaved latitude/longitude in radians.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_render_grid(
    parameters: *const f64,
    width: usize,
    height: usize,
    projection: u32,
    upstream: u32,
    out: *mut f64,
) -> i32 {
    if parameters.is_null() || out.is_null() {
        return -1;
    }
    status(|| {
        let p = unsafe { std::slice::from_raw_parts(parameters, 4) };
        let projection = match projection {
            0 => crate::surface::Projection::Orthographic,
            1 => crate::surface::Projection::Rectangular,
            2 => crate::surface::Projection::Mollweide,
            _ => return Err(Error("invalid projection".into())),
        };
        let view = crate::imaging::View {
            inclination: p[0],
            obliquity: p[1],
            phase: p[2],
            oblateness: Some((p[3], None, true)),
            upstream_grid: upstream != 0,
            ..Default::default()
        };
        let points = crate::imaging::coordinate_grid(&view, width, height, projection)?;
        for (i, p) in points.iter().enumerate() {
            unsafe {
                *out.add(2 * i) = p[0];
                *out.add(2 * i + 1) = p[1];
            }
        }
        Ok(())
    })
}

/// # Safety
/// p holds np doubles; xyz holds rows*3 doubles; out holds rows doubles. No overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_surface(
    id: u64,
    mode: u32,
    p: *const f64,
    np: usize,
    xyz: *const f64,
    rows: usize,
    flags: u32,
    out: *mut f64,
) -> i32 {
    if p.is_null()
        || xyz.is_null()
        || out.is_null()
        || !(22..=42).contains(&np)
        || rows > 16_000_000
    {
        return -1;
    }
    status(|| {
        let view = image_view(mode, unsafe { std::slice::from_raw_parts(p, np) }, flags)?;
        let points: Vec<_> = unsafe { std::slice::from_raw_parts(xyz, rows * 3) }
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let values = crate::imaging::intensity(&get(id)?, &view, &points, flags & 4 != 0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}
/// # Safety
/// `parameters` has `rows*width` readable doubles, `out` has `rows` writable
/// doubles; buffers must not overlap. Results are committed only on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_batch(
    id: u64,
    mode: u32,
    parameters: *const f64,
    rows: usize,
    width: usize,
    out: *mut f64,
) -> i32 {
    if parameters.is_null()
        || out.is_null()
        || rows == 0
        || !(22..=42).contains(&width)
        || rows.checked_mul(width).is_none()
    {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        let parameters = unsafe { std::slice::from_raw_parts(parameters, rows * width) };
        let values = parameters
            .chunks_exact(width)
            .map(|p| evaluate(&map, mode, p))
            .collect::<Result<Vec<_>>>()?;
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, rows);
        }
        Ok(())
    })
}

/// Observer-frame emitted flux geometry gradients, including orientation and limb darkening.
/// # Safety
/// p holds np parameters in the batch layout; out holds 8+Ny+udeg doubles:
/// value,phase,inc,obl,xo,yo,ro,amp,coefficient derivatives,limb derivatives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_jacobian(
    id: u64,
    p: *const f64,
    np: usize,
    out: *mut f64,
    len: usize,
) -> i32 {
    if p.is_null() || out.is_null() || !(22..=42).contains(&np) {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        require(
            len == 8 + map.coefficients().len() + np - 22,
            "invalid Jacobian shape",
        )?;
        let p = unsafe { std::slice::from_raw_parts(p, np) };
        require(
            p.iter().all(|v| v.is_finite()),
            "invalid Jacobian parameters",
        )?;
        let occ = Occultor::new(p[3], p[4], p[6])?;
        let d = crate::differentiation::emitted(
            &map,
            p[1],
            p[2],
            p[0],
            &p[22..],
            if p[5] > 0. { Some(occ) } else { None },
        )?;
        let mut values = vec![
            d.value,
            d.phase,
            d.inclination,
            d.obliquity,
            d.occultor[0],
            d.occultor[1],
            d.occultor[2],
            d.amplitude,
        ];
        values.extend(d.coefficients);
        values.extend(d.limb);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// Observer-frame emitted flux geometry gradients, including orientation and limb darkening.
/// # Safety
/// parameters holds rows*width doubles using the map-batch layout; out holds
/// rows*3 doubles (x, y, radius derivatives), with no overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_gradients(
    id: u64,
    parameters: *const f64,
    rows: usize,
    width: usize,
    out: *mut f64,
) -> i32 {
    if parameters.is_null()
        || out.is_null()
        || rows == 0
        || rows > 1_000_000
        || !(22..=42).contains(&width)
    {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        let parameters = unsafe { std::slice::from_raw_parts(parameters, rows * width) };
        let mut values = Vec::with_capacity(rows * 3);
        for p in parameters.chunks_exact(width) {
            require(
                p.iter().all(|v| v.is_finite()),
                "invalid gradient parameters",
            )?;
            let occ = Occultor::new(p[3], p[4], p[6])?;
            let g = if p[5] > 0. {
                map.projected(p[1], p[2], p[0])?
                    .limb_filtered(&p[22..])?
                    .flux_gradient(occ)?
            } else {
                [0.; 3]
            };
            values.extend(g);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// # Safety
/// `args` has `nargs` readable doubles and `out` has `len` writable doubles.
/// Output is amplitude followed by coefficients, so len=1+(degree+1)^2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_transform(
    id: u64,
    operation: u32,
    args: *const f64,
    nargs: usize,
    out: *mut f64,
    len: usize,
) -> i32 {
    if args.is_null() || out.is_null() {
        return -1;
    }
    status(|| {
        let mut map = get(id)?;
        require(
            len == map.coefficients().len() + 1,
            "invalid transformed map shape",
        )?;
        let p = unsafe { std::slice::from_raw_parts(args, nargs) };
        match operation {
            0 => {
                require(p.len() == 4, "rotation needs axis and angle")?;
                map.rotate([p[0], p[1], p[2]], p[3])?;
            }
            1 => {
                require(
                    p.len() == 5,
                    "spot needs contrast, radius, latitude, longitude, smoothing",
                )?;
                crate::surface::add_spot(
                    &mut map,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    if p[4] < 0. { None } else { Some(p[4]) },
                )?;
            }
            _ => return Err(Error("unknown map transformation".into())),
        }
        let mut data = vec![map.amplitude()];
        data.extend_from_slice(map.coefficients());
        registry()
            .lock()
            .map_err(|_| Error("map registry poisoned".into()))?
            .insert(id, map);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out, len);
        }
        Ok(())
    })
}

/// # Safety
/// ids: nbodies u64s; bodies: nbodies*42 doubles; orbits: (nbodies-1)*7 doubles;
/// times: ntimes doubles. Modes 1/14 return three coordinates per body;
/// modes 0/2/3/4 return one scalar. Modes 5..10 return value plus Jacobian:
/// 3+7*(nbodies-1)+7*nbodies columns, plus sum(9+Ny+udeg) for modes 8..10.
/// light_delay bits: bit 0 enables delay; bit 1 selects starry orbital conventions.
/// Arrays must be nonoverlapping. Body layout: r,m,prot,t0,theta0,inc,obl,
/// reflected,roughness,udeg,u[0..20],rv,veq,alpha,source_npts. Orbit: a,period,ecc,inc,w,Omega,t0.
/// Body entries 34..42: oblate,f,omega,beta,tpole,wavelength,fdeg,normalized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_system_batch(
    ids: *const u64,
    nbodies: usize,
    bodies: *const f64,
    orbits: *const f64,
    times: *const f64,
    ntimes: usize,
    light_delay: u32,
    exposure: f64,
    samples: usize,
    mode: u32,
    out: *mut f64,
) -> i32 {
    if ntimes == 0 || ntimes > 1_000_000 {
        return -1;
    }
    let exposures = vec![exposure; ntimes];
    unsafe {
        starry_system_batch_v2(
            ids,
            nbodies,
            bodies,
            orbits,
            times,
            ntimes,
            light_delay,
            exposures.as_ptr(),
            samples,
            0,
            mode,
            out,
        )
    }
}

/// # Safety
/// Same buffers as starry_system_batch; exposures holds ntimes readable doubles.
/// order selects midpoint (0), trapezoid (1) or Simpson (2) exposure integration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_system_batch_v2(
    ids: *const u64,
    nbodies: usize,
    bodies: *const f64,
    orbits: *const f64,
    times: *const f64,
    ntimes: usize,
    light_delay: u32,
    exposures: *const f64,
    samples: usize,
    order: u32,
    mode: u32,
    out: *mut f64,
) -> i32 {
    if ids.is_null()
        || bodies.is_null()
        || times.is_null()
        || out.is_null()
        || exposures.is_null()
        || order > 2
        || nbodies == 0
        || nbodies > 100
        || ntimes == 0
        || ntimes > 1_000_000
        || (nbodies > 1 && orbits.is_null())
    {
        return -1;
    }
    status(|| {
        let ids = unsafe { std::slice::from_raw_parts(ids, nbodies) };
        let b = unsafe { std::slice::from_raw_parts(bodies, nbodies * 42) };
        let orbital = if nbodies > 1 {
            unsafe { std::slice::from_raw_parts(orbits, (nbodies - 1) * 7) }
        } else {
            &[]
        };
        let times = unsafe { std::slice::from_raw_parts(times, ntimes) };
        let mut body_list = vec![];
        for i in 0..nbodies {
            let p = &b[i * 42..(i + 1) * 42];
            require(
                p.iter().all(|v| v.is_finite()) && p[9] >= 0. && p[9] <= 20. && p[9].fract() == 0.,
                "invalid serialized body",
            )?;
            let mut body = crate::system::Body::new(get(ids[i])?, p[0], p[1])?;
            body.rotation_period = p[2];
            body.reference_epoch = p[3];
            body.phase0 = p[4];
            body.inclination = p[5];
            body.obliquity = p[6];
            body.reflected = p[7] != 0.;
            body.roughness = p[8];
            body.limb_darkening = p[10..10 + p[9] as usize].to_vec();
            body.radial_velocity = p[30] != 0.;
            body.equatorial_velocity = p[31];
            body.differential_rotation = p[32];
            require(
                p[33] >= 1. && p[33] <= 100_000. && p[33].fract() == 0.,
                "invalid source sample count",
            )?;
            body.source_samples = p[33] as usize;
            if p[34] != 0. {
                require(
                    p[40] >= 0. && p[40] <= 20. && p[40].fract() == 0.,
                    "invalid gravity degree",
                )?;
                body.flattening = Some(p[35]);
                body.normalized = p[41] != 0.;
                if p[40] > 0. {
                    body.gravity = Some(crate::oblate::GravityDarkening {
                        degree: p[40] as usize,
                        omega: p[36],
                        flattening: p[35],
                        beta: p[37],
                        polar_temperature: p[38],
                        wavelength_meters: p[39],
                    });
                }
            }
            body_list.push(body);
        }
        let primary = body_list.remove(0);
        let mut secondaries = vec![];
        for (i, body) in body_list.into_iter().enumerate() {
            let p = &orbital[i * 7..i * 7 + 7];
            let orbit = crate::orbit::Orbit {
                semimajor_axis: p[0],
                period: p[1],
                eccentricity: p[2],
                inclination: p[3],
                omega: p[4],
                ascending_node: p[5],
                transit_epoch: p[6],
            };
            secondaries.push(crate::system::Secondary { body, orbit });
        }
        let mut system = crate::system::System::new(primary, secondaries)?;
        system.light_delay = light_delay & 1 != 0;
        system.source_convention = light_delay & 2 != 0;
        system.exposure_samples = samples;
        system.exposure_order = order;
        let exposures = unsafe { std::slice::from_raw_parts(exposures, ntimes) };
        require(
            exposures.iter().all(|e| e.is_finite() && *e >= 0.),
            "invalid exposure durations",
        )?;
        let mut values = vec![];
        for (&t, &exposure) in times.iter().zip(exposures) {
            system.exposure = exposure;
            match mode {
                0 => values.extend(system.flux(t)?),
                14 => {
                    for p in system.apparent(t)?.0 {
                        values.extend(p);
                    }
                }
                1 => {
                    if system.source_convention {
                        for p in crate::source_orbits::positions(&system, t)? {
                            values.extend(p);
                        }
                        continue;
                    }
                    for (p, _) in system.states(t)? {
                        values.extend(p)
                    }
                }
                2 => {
                    for (_, v) in system.states(t)? {
                        values.push(
                            -v[2] * crate::orbit::SOLAR_RADIUS_METERS / crate::orbit::DAY_SECONDS,
                        )
                    }
                }
                3 | 4 => values.extend(system.radial_velocities(t, mode == 3)?),
                5..=10 => {
                    let jac = match mode {
                        5 => crate::system_derivatives::flux(&system, t)?,
                        8 => crate::system_derivatives::flux_surface(&system, t)?,
                        6 | 7 => crate::system_derivatives::radial_velocity(&system, t, mode == 6)?,
                        _ => crate::system_derivatives::radial_velocity_surface(
                            &system,
                            t,
                            mode == 9,
                            true,
                        )?,
                    };
                    for (i, value) in jac.values.iter().enumerate() {
                        values.push(*value);
                        for k in 0..jac.derivatives.cols {
                            values.push(jac.derivatives[(i, k)]);
                        }
                    }
                }
                _ => return Err(Error("invalid system mode".into())),
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// # Safety
/// design: rows*cols, data: rows, noise: rows*rows. Prior mean/covariance may
/// both be null, or have cols and cols*cols readable doubles. Output has
/// cols+cols*cols+2 writable doubles: mean, covariance, conditional and marginal lnL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_linear_solve(
    design: *const f64,
    rows: usize,
    cols: usize,
    data: *const f64,
    noise: *const f64,
    prior_mean: *const f64,
    prior_cov: *const f64,
    out: *mut f64,
) -> i32 {
    if design.is_null()
        || data.is_null()
        || noise.is_null()
        || out.is_null()
        || rows == 0
        || cols == 0
        || rows > 10000
        || cols > 441
        || prior_mean.is_null() != prior_cov.is_null()
    {
        return -1;
    }
    status(|| {
        let x = crate::matrix::Matrix {
            rows,
            cols,
            data: unsafe { std::slice::from_raw_parts(design, rows * cols) }.to_vec(),
        };
        let data = unsafe { std::slice::from_raw_parts(data, rows) };
        let noise = crate::matrix::Matrix {
            rows,
            cols: rows,
            data: unsafe { std::slice::from_raw_parts(noise, rows * rows) }.to_vec(),
        };
        let prior = if prior_mean.is_null() {
            None
        } else {
            Some(crate::inference::GaussianPrior {
                mean: unsafe { std::slice::from_raw_parts(prior_mean, cols) }.to_vec(),
                covariance: crate::matrix::Matrix {
                    rows: cols,
                    cols,
                    data: unsafe { std::slice::from_raw_parts(prior_cov, cols * cols) }.to_vec(),
                },
            })
        };
        let solution = crate::inference::solve(&x, data, &noise, prior.as_ref())?;
        let marginal = if let Some(p) = &prior {
            crate::inference::marginal_log_likelihood(&x, data, &noise, p)?
        } else {
            f64::NAN
        };
        let mut values = solution.mean;
        values.extend(solution.covariance.data);
        values.extend([solution.log_likelihood, marginal]);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
        }
        Ok(())
    })
}

/// # Safety
/// xyz: rows*3 readable doubles; source: three readable doubles; out: rows
/// writable doubles. Mode 0 emitted, 1 reflected, 2 albedo. No overlapping buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn starry_map_intensity_batch(
    id: u64,
    xyz: *const f64,
    rows: usize,
    mode: u32,
    source: *const f64,
    roughness: f64,
    out: *mut f64,
) -> i32 {
    if xyz.is_null() || source.is_null() || out.is_null() || rows == 0 || rows > 16_000_000 {
        return -1;
    }
    status(|| {
        let map = get(id)?;
        let xyz = unsafe { std::slice::from_raw_parts(xyz, rows * 3) };
        let s = unsafe { std::slice::from_raw_parts(source, 3) };
        let source = [s[0], s[1], s[2]];
        let mut values = vec![];
        for p in xyz.chunks_exact(3) {
            let point = [p[0], p[1], p[2]];
            values.push(match mode {
                0 => map.intensity_xyz(point)?,
                1 => crate::reflected::intensity(&map, point, source, roughness)?,
                2 => std::f64::consts::PI * map.intensity_xyz(point)?,
                _ => return Err(Error("invalid intensity mode".into())),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, rows);
        }
        Ok(())
    })
}
