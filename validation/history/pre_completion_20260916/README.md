# starry-rust

An **in-progress Rust port of starry**, covering emitted and reflected light,
oblate and gravity-darkened maps, Doppler convolution, radial velocities,
Keplerian systems, surface tools, and Gaussian map inference.
The target is complete numerical functionality in Rust with a new Python API.
Drop-in Python/Theano compatibility is not required. Numerical coverage is
still being expanded; see [PORT_STATUS.md](PORT_STATUS.md).

The reference is [rodluger/starry](https://github.com/rodluger/starry/tree/b72dff08588532f96bd072f2f1005e227d8e4ed8),
commit `b72dff08588532f96bd072f2f1005e227d8e4ed8`. Upstream's MIT license is
retained in [LICENSE](LICENSE). The upstream checkout is separate at
`../starry-upstream` and is used only for validation. No upstream C++ code is
compiled into the Rust library.

## Working features

- Real spherical harmonics, polynomial and Green basis transformations.
- Analytic full-disk flux and single circular occultations of emitted-light maps.
- Bulirsch elliptic integrals and the upstream I/J integral recurrences.
- Arbitrary axis rotations, with stable spherical projection above degree 8.
- Polynomial limb darkening, alone or multiplied into a harmonic map.
- Flux design rows (exact derivatives with respect to harmonic coefficients).
- Analytic intensity derivatives with respect to latitude and longitude.
- Occultor position/radius derivatives via integration of the boundary shape derivative.
- Independent adaptive surface integration to cross-check the analytic flux path.
- Inclination, obliquity, and rotational phase projections.
- Lambert and polynomial Oren-Nayar reflection, finite-source sampling, and occultations.
- Oblate projected disks and wavelength-dependent gravity-darkening filters.
- Differential-rotation velocity fields and Rossiter-McLaughlin anomalies.
- Native Doppler chord kernels and multicomponent rest-spectrum convolution.
- Keplerian systems, overlapping occultor unions, light delays, and exposure averaging.
- Spectral maps and systems (independent wavelength channels).
- Circular spots, three image projections, and sampled image-to-harmonic fitting.
- Gaussian priors, posterior means/covariances, and marginalized likelihoods.
- A persistent, array-oriented Python `ctypes` interface and a native Rust API.
- First emitted-light parameter Jacobians, bounded surface minimization, and limb validity.
- Filtered surface images, oblate primary systems, and exposure-integrated RV.
- Doppler wavelength resampling, conditional and alternating inverse fits,
  and unrestricted spectral-map forward/transpose operators without dense matrices.
- Fixed-geometry scalar/spectral Map and System inference and pixel/harmonic transforms.

The crate has **no Rust package dependencies**. The default `Map::flux` path is
analytic for emitted light; reflected occultations, overlapping occultors, and
oblate disks use adaptive numerical integration. Basis transforms are cached in
each Rust `Map` and retained by integer handles in Python. No performance
comparison with upstream has been established.

## Build and run

```powershell
cd D:\Games\starry-rust
cargo build --release --offline
cargo test --offline
cargo run --release --offline --example transit
python -m unittest discover -s python -v
```

Tested on Windows x64 with Rust 1.96.1 (MSVC) and Python 3.13. The reference
harness uses MinGW `g++`. Linux and macOS have not been tested.

Build and install a wheel containing the Rust runtime:

```powershell
python -m pip wheel --no-cache-dir --no-build-isolation --no-deps --wheel-dir dist .
python -m pip install dist/starry_rust-0.1.0-py3-none-win_amd64.whl
```

Source builds require Rust, setuptools and wheel. Installed wheels require
NumPy but do not require Rust or upstream starry. See [NUMERICAL_API.md](NUMERICAL_API.md).

### Rust

```rust
use starry_rust::{Map, Result, occultation::Occultor};

fn main() -> Result<()> {
    let mut map = Map::new(5)?;
    map.set(1, 0, 0.2)?;
    map.rotate([0.0, 1.0, 0.0], 0.3)?;
    let occ = Occultor::new(0.2, 0.4, 0.1)?;
    println!("flux = {}", map.flux(Some(occ))?);
    println!("gradient = {:?}", map.flux_gradient(occ)?);
    println!("limb-darkened = {}", map.flux_limb_darkened(&[0.4, 0.2], Some(occ))?);
    Ok(())
}
```

### Python

Add the repository's `python` directory to `sys.path`, or run from that directory:

```python
from starry_rust import Map

m = Map(ydeg=5)
m.y[2] = 0.2
m.rotate([0, 1, 0], 0.3)
print(m.flux(xo=0.2, yo=0.4, ro=0.1))
print(m.flux_gradient(xo=0.2, yo=0.4, ro=0.1))
print(m.flux_limb_darkened([0.4, 0.2], xo=0.2, yo=0.4, ro=0.1))
```

## Conventions and compatibility

- Rust angles are **radians**. Python defaults to radians and accepts
  `angle_unit='deg'`; upstream Python defaults to degrees.
- Indexing is `n = l*l + l + m`. `Y00 = 1/pi`, and `[Y1,-1, Y1,0, Y1,1]`
  equals `sqrt(3)/pi * [y, z, x]`.
- The observer is on +z. Map intensity uses
  `(x,y,z) = (cos(lat) sin(lon), sin(lat), cos(lat) cos(lon))`.
- `y00=1` gives unit unocculted flux for a uniform map. Raw coefficient setters
  do not rescale amplitude or enforce `y00=1`. Amplitude is a separate multiplier.
- Map rotations are active, right handed, and normalize the axis. They match
  upstream `Map.rotate`, including its RHS-operator sign reversal.
- An `Occultor` is a **foreground**, opaque circle, in units of the map radius.
  Low-level Rust calls require foreground geometry. Python `Map.flux` accepts
  `zo`, and `System` computes foreground/background ordering from its orbits.
- Limb darkening is applied in the observer frame, after rotating the map.
  The profile is `1 - sum(u_n * (1-mu)^n)`, normalized by its disk average.
  A nonpositive disk average is rejected; positivity everywhere is not enforced.
- The admitted degree is 0 through 20, including combined map/limb degree.
  Accuracy is tested on a finite set through degree 10 for flux/geometry
  gradients, 12 for basis transformations, and 6 for rotation field identities.
  Higher degrees are experimental; polynomial cancellation can amplify errors.
- Coefficient design rows exclude amplitude. Flux and `Map` gradients include it.
- Geometry gradients use a numerical boundary integral with convergence checks,
  not upstream's automatic-differentiation graph. At coincident unit disks the
  radius derivative is undefined and returns an error.

## Validation

Run the full reproducible suite:

```powershell
python validation/run.py
```

It checks formatting, Clippy, Rust tests, the release build, the Python ABI, and
direct comparisons against the pinned, unmodified upstream C++ kernels.
`rustfmt` and `clippy` components must be installed. The runner does not install
anything or use the network. The reference harness needs the sibling checkout.

See [VALIDATION.md](VALIDATION.md) for measured errors and limitations. The latest
machine-readable report is `validation/report.json`, with source and binary hashes.
No speedup over starry has been established.

## Remaining port work

See [PORT_STATUS.md](PORT_STATUS.md) for the precise compatibility boundary.
High-degree support beyond 20, broader derivative coverage,
and wider numerical regression coverage remain open. Theano integration and
drop-in upstream API signatures are outside the user-confirmed scope.

### Extended Python examples

```python
import numpy as np
from starry_rust import Map, Primary, Secondary, System, DopplerMap

planet = Map(3, reflected=True, angle_unit='deg')
planet.spot(contrast=0.2, radius=20, lat=30, lon=45)
phase_curve = planet.flux(theta=np.arange(360), xs=0, ys=3, zs=4)

star = Map(2, udeg=2, rv=True, veq=10000)
star[1], star[2] = 0.3, 0.1
system = System(Primary(star), Secondary(Map(amp=0), porb=5, r=0.1, m=0.001))
times = np.linspace(-0.1, 0.1, 101)
flux, velocity = system.flux(times), system.rv(times)

surface = Map(5)
surface.spot(0.2, radius=0.3)
image = surface.render(res=100, projection='rect')
rest_spectrum = np.ones(101)  # includes ten padding samples at each end
doppler = DopplerMap(surface, rest_spectrum, log_spacing=1e-5,
                     half_width=10, veq=30000)
broadened = doppler.flux(theta=[0, 0.5])
```
