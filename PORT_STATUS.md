# Numerical port status

Updated 2026-09-16. Scope: native numerical functionality, a new Python API,
and evidence against pinned upstream within the documented supported domain.
Final release verification passed: 47 Rust tests, 77 Python tests, 200 selected
upstream tests, all reference suites, Linux checks and installed-wheel examples.

## Implemented

Map emitted/reflected/finite-source/oblate/gravity/RV models; Doppler convolution
and inverse modes; Keplerian systems with overlapping silhouettes and exposure
integration; scalar/spectral Gaussian inference; surface/image/pixel tools.
Native first observable Jacobians include System surface parameters and RV.
Orbital state Jacobians and Hessians are available.

The source audit includes public methods, properties and explicit core-operator
implementation/test mappings. Evidence includes original C++/AD kernels,
selected original Python tests, source-extracted Doppler solvers and exoplanet
0.4.5 orbital methods, independent analytic/quadrature references, and native
forward perturbations. An inventory entry is not itself a parity test.

## Important choices

- Select `orbit_convention='starry'` for source orbital/RV semantics.
- Select Doppler `continuum_index` for source sampled normalization.
- New Python API; no Theano graph or plotting-widget compatibility.
- Combined degree32; individual Python limb/gravity filters capped at20.
- First observable derivatives and orbital Hessians; singular boundaries and
  source derivative-order restrictions are explicit in NUMERICAL_LIMITS.md.
- Windows x64 wheel and Linux native validation; macOS untested.

Read NUMERICAL_API.md for usage and VALIDATION.md for measured evidence.
Historical status sections are preserved under
validation/history/pre_completion_20260916/ and are superseded by this file.
