# starry-rust

A Rust numerical port of [starry](https://github.com/rodluger/starry/tree/b72dff08588532f96bd072f2f1005e227d8e4ed8),
with a new NumPy-based Python interface. The numerical kernels run in Rust;
Python handles arrays and optimization orchestration. Upstream C++ and Theano
are not runtime dependencies.

Reference commit: `b72dff08588532f96bd072f2f1005e227d8e4ed8`.
The supported domain and intentional numerical differences are specified in
[NUMERICAL_LIMITS.md](NUMERICAL_LIMITS.md). This is a numerical port, not a
Python/Theano drop-in replacement.

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
- Harmonic maps and combined products through degree32, with the filter limits
  and singular boundaries documented in the numerical contract.

Use `System(..., orbit_convention='starry')` for the original orbital/RV
conventions. The default barycentric mode is a documented extension. Use
`DopplerMap(..., continuum_index=...)` for source-style sampled-continuum
normalization; its default is integrated-continuum normalization.

## Build and use

```powershell
cargo build --release --offline
python -m unittest discover -s python -v
python setup.py bdist_wheel
python -m pip install dist/starry_rust-0.1.0-py3-none-win_amd64.whl
```

Source builds require Rust, setuptools and wheel. Installed wheels require
NumPy. The packaged wheel targets Windows x64; native Rust/Python execution is
also tested on Ubuntu Linux. macOS is not tested.

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

