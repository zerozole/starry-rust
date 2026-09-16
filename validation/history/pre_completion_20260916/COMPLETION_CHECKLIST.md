# Numerical port completion checklist

Scope agreed with the user: numerical functionality in Rust, a new Python
interface, and validation against upstream over supported parameter ranges.
There is no requirement to reproduce Python/Theano symbolic graphs or GUI APIs.

This checklist records completion evidence, not just the existence of methods.
See REMAINING_WORK.md for the current detailed list of open work.
`validation/feature_audit.py` inventories every public non-property method in
upstream maps.py, doppler.py and kepler.py. Its JSON output identifies code
counterparts and incomplete mappings. Internal operators and degree-domain
coverage must also be audited before claiming completion.

## 1. Numerical functionality

- [x] Emitted/reflected/oblate forward models, velocity maps, finite sources.
- [x] Native orbital geometry, foreground ordering, overlapping silhouettes.
- [x] Scalar/per-observation exposure durations; midpoint/trapezoid/Simpson
  stencils for flux and flux-weighted rotational RV.
- [x] Map/System scalar and spectral Gaussian inference.
- [x] Doppler map/spectrum fits for normalized and unnormalized data, with
  conditional covariances and a clearly labeled joint local approximation.
- [x] Render coordinates, pixel transforms and Gaussian harmonic smoothing.
- [x] Stable harmonic values, tangents, rotations and circular emitted flux
  through the current maximum degree20. Independent high-precision checks
  resolve observed upstream floating-point cancellation cases.
- [x] Degree32 emitted flux and stable high-degree specialized paths; broader stress validation remains.
- [ ] Complete native parameter-derivative coverage and audit singular limits.
- [x] Reflected illumination/scattering, oblate physical and Doppler first derivatives.
- [x] System orbit/spin/time/exposure/mass/radius Jacobian with overlaps and light delay.
- [x] Map radial-velocity parameter, coefficient, limb and shape Jacobian.
- [x] Upstream combined Doppler temperature schedule with supplied spectrum guess.
- [x] Numeric image preprocessing, smoothing, extents, cube SVD and spectrum resampling; new grid convention documented.
- [ ] Finish reference validation of the expanded internal-operator inventory.

General unrestricted spectral-map posterior inference and joint primary/reflected
inference appeared on earlier lists. Their precise upstream equivalents must be
checked before treating proposed extensions as missing upstream features.

## 2. New Python interface

- [x] Map, System and Doppler interfaces; broadcasting and spectral shapes.
- [x] Native library packaged in a platform wheel.
- [x] Numerical guide with executable examples.
- [ ] Integrate every remaining numerical mode into the public interface.
- [ ] Final documentation review against the completed feature inventory.

## 3. Validation

- [x] Rust tests and Python ABI tests.
- [x] Selected unmodified upstream Python tests via an eager test adapter.
- [x] Pinned unmodified C++ references and independent analytic identities.
- [x] High-precision reference checks for discovered degree15/20 cancellation.
- [ ] Finish source-wide derivative reference coverage; reflected and oblate AD comparisons pass, Doppler/System perturbation checks pass.
- [ ] Validate newly admitted degrees and extreme supported geometries.
- [x] Windows and Linux Rust/Python validation; macOS remains untested.
- [ ] Final integrated run and installable artifact verification after all changes.

The three requested objectives are not complete while required boxes remain open.
