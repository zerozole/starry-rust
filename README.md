# starry-rust

A Rust numerical port of [starry](https://github.com/rodluger/starry),
with a new NumPy-based Python interface. The numerical kernels run in Rust;
Python handles arrays and optimization orchestration. Upstream C++ and Theano
are not runtime dependencies.

The supported domain and intentional numerical differences are specified in
[NUMERICAL_LIMITS.md](NUMERICAL_LIMITS.md). This is a numerical port, not a
Python/Theano drop-in replacement.

Upstream starry depends on Theano and a C++ extension that are difficult to
build against current Python. This port exists to generate spotted-star light
curve templates without that toolchain, and runs anywhere Rust and NumPy do.

## Features

- Emitted, reflected, finite-source, oblate and gravity-darkened maps.
- Occultations, overlapping silhouettes, rotation and radial velocities.
- Keplerian multi-body systems, light delays and exposure integration.
- Native first Jacobians for Map flux/RV, System flux/RV and Doppler spectra;
  orbital state Jacobians and Hessians.
- Doppler convolution, map/spectrum inversion, L1 and tempered solvers.
- Scalar/spectral Gaussian map and System inference with full covariances.
- Surface rendering, spots, numeric-image loading, harmonic/pixel transforms,
  smoothing, minimization and spectral cube factorization.
- Harmonic maps and combined products through degree 32, with the filter limits
  and singular boundaries documented in the numerical contract.

Use `System(..., orbit_convention='starry')` for the original orbital/RV
conventions. The default barycentric mode is a documented extension. Use
`DopplerMap(..., continuum_index=...)` for source-style sampled-continuum
normalization; its default is integrated-continuum normalization.

## Build and use

```bash
cargo build --release --offline
python -m unittest discover -s python -v
python setup.py bdist_wheel
python -m pip install dist/starry_rust-0.1.0-*.whl
```

Source builds require Rust, setuptools and wheel. Installed wheels require
NumPy. Built and tested on Ubuntu Linux and Windows x64; the wheel is named for
whichever platform builds it. macOS is not tested.

```python
from starry_rust import Map
m = Map(3, udeg=2)
m.y[3] = .2
m.u[1:] = [.3, .1]
flux = m.flux(xo=.2, yo=.4, ro=.1)
jacobian = m.flux_jacobian(xo=.2, yo=.4, ro=.1)
```

For source-tree use, add `python/` to `PYTHONPATH`; the loader finds the release
library. `STARRY_RUST_LIBRARY` selects an explicit native library.

## Documentation and verification

- [Numerical API and executable examples](NUMERICAL_API.md)
- [Numerical limits and source differences](NUMERICAL_LIMITS.md)
- [Validation evidence and commands](VALIDATION.md)

Release verification passes 47 Rust tests, 77 Python tests and 200 selected
upstream tests, together with the reference suites and installed-wheel examples
on Windows and Linux.

`python validation/record_run.py` runs formatting, strict Clippy, Rust/Python
tests, reference comparisons, source audits and selected original upstream
tests. Reference compilation needs the separate `../starry-upstream` checkout
and a C++ compiler; prebuilt local reference executables can be reused.

Upstream's MIT license is retained in [LICENSE](LICENSE). The separately
vendored exoplanet validation reference retains its own license under
`validation/`. No source-reference Python is used by the installed runtime.

## Acknowledgements

This is a port of Rodrigo Luger's starry package and its C++/Theano core. The
spherical harmonic basis, rotation and occultation solvers, reflected and
oblate map models, Doppler machinery and Keplerian system layer are all ports
of the original code. Numerical results agree with the original across roughly
a thousand comparison cases: 2.37e-9 maximum absolute discrepancy on reflected
flux over 120 cases, 1.10e-9 on oblate flux over 54, 7.28e-12 on RV filters and
their derivatives over 60, and 2.71e-14 on analytic circular system flux over
88. [VALIDATION.md](VALIDATION.md) gives the full table, and
[NUMERICAL_LIMITS.md](NUMERICAL_LIMITS.md) records where this port departs from
the original on purpose.
