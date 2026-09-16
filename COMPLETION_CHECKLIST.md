# Numerical port completion checklist

Completed 2026-09-16 for the numerical profile in NUMERICAL_LIMITS.md.
The three requested areas are closed: native numerical functionality, the new
Python interface, and validation against pinned source/independent references.
This does not claim Theano drop-in compatibility or uniform accuracy at every
possible input. Evidence and limits are recorded in VALIDATION.md.

## 1. Native numerical functionality

- [x] Emitted/reflected/finite-source/oblate/gravity/RV forward models.
- [x] Keplerian geometry, overlapping silhouettes, light delays and exposures.
- [x] Source-compatible independent-pair orbital positions and stellar-reflex
  RV contributions, selected explicitly with orbit_convention='starry'.
- [x] Map, Doppler and full System first observable Jacobians, including
  surface coefficients/filters, masses/radii, source coupling and RV quotients.
- [x] Orbital Hessians; source derivative-order and singular-boundary audit.
- [x] Scalar/spectral Map and System Gaussian inference; Doppler L1/L2,
  nonlinear, normalized, alternating/tempered and initialization modes.
- [x] Source sampled-continuum normalization and exact spectrum Jacobian.
- [x] Rendering, coordinates, pixels, smoothing, spots, numeric-image loading,
  spectral cube factorization and surface minimization.
- [x] Degree32 paths and combined-product stress tests; independent Decimal
  evidence for unstable high-degree source polynomial cases.

## 2. New Python interface

- [x] Scalar/spectral shapes, broadcasting, angle units and explicit boundaries.
- [x] Source numerical modes integrated into the public interface.
- [x] System/RV parameter-update and continuum examples, executed from wheel.
- [x] README/API/status/validation documents consolidated; earlier handoffs
  preserved in validation/history/pre_completion_20260916/.
- [x] Windows wheel rebuilt and installed in isolation with its own DLL.

## 3. Validation

- [x] Public methods, properties and 85 core operators mapped to counterparts
  and evidence; each reference type is labeled rather than treated as universal parity.
- [x] Original C++/AD comparisons and 200 selected original Python tests.
- [x] Independent System/Doppler/RV derivative checks and source-extracted
  Doppler solver/exoplanet orbital comparisons.
- [x] Singular branches, contact neighborhoods, high degree, extreme views,
  overlap, finite-source and gravity limits tested/documented.
- [x] Windows integrated run: formatting, strict Clippy, 47 Rust tests,
  77 Python tests, all reference comparisons and selected upstream tests.
- [x] Linux: 47 Rust tests and 77 Python tests with separate native build.
- [x] Packaged wheel smoke and all numerical guide Python examples pass.

No identified required task remains open within this supported profile.
MacOS testing, performance work, real-data science validation and extensions
outside this profile are separate follow-up work, not claims of this release.
