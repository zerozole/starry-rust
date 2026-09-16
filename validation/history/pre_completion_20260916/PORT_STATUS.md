# Latest verified additions: RV, system masses and radii

Map.rv_jacobian now supplies native physical, harmonic, limb and occultor
first derivatives. System.flux_jacobian now includes masses and radii, with
barycentric/Kepler/light-delay and finite-source illumination chain rules.

Windows integrated validation passes: 43 Rust integration tests, 66 Python
tests, 200 selected upstream tests, all existing reference comparisons and
60 new upstream RV-filter value/derivative cases (max error 7.28e-12).
Linux's native build and 66 Python tests pass.

The current actionable inventory is REMAINING_WORK.md. Full system surface
parameter composition and System RV derivatives remain. Higher derivative
requirements must be tied to source support: several source GradientOp classes
lack another grad method, and the specialized limb solver rejects differentiation
of its derivative outputs. Do not invent a blanket arbitrary-Hessian requirement.

The following earlier sections are historical; this section supersedes their
mass/radius and Map RV derivative limitations.

# Latest continuation: native derivatives and system composition

This section supersedes older statements below about unavailable Jacobians.
Upstream remains clean at b72dff08588532f96bd072f2f1005e227d8e4ed8.

Implemented native reflected source/scattering/finite-radius derivatives;
oblate gravity, flattening and viewing derivatives; Doppler physical/limb
Jacobian; complete common Map first-Jacobian interface for specialized maps;
and System orbital/spin/time/exposure derivatives with overlap and light delay.
System derivatives hold mass, radii and surface parameters fixed. New Python
methods and singular boundaries are documented in NUMERICAL_API.md.

Reflected shape and illumination derivatives each match 63 upstream AD cases
(max errors 1.22e-15 and 6.04e-16); oblate derivatives match 40 upstream AD
cases (max 1.06e-10). Specialized high-degree comparisons retain 6 passing
cases. These supplement the previous 581 C++ cases and 200 selected upstream
Python tests. System/Doppler physical derivatives have numerical-perturbation
and identity tests, not direct source-wide AD parity.

Added the upstream alternating Doppler temperature sequence with explicit
initial spectra via solve_bilinear_tempered, including L1/L2 spectrum steps.
Optional continuum-based uniform-kernel L1 deconvolution initialization is implemented and checked against independent matrix equations.

Remaining: source-wide internal-operator reference coverage; full System
mass/radius/surface parameter composition; RV parameter Jacobians; higher
flux derivatives and singular-limit audit; wider degree32/extreme-domain
validation. macOS is untested. Full numerical-port completion is not claimed.

Current validation logs and reports are under validation/. Reproduce using
validation/record_run.py and validation/linux_python.sh. Windows packaging:
python setup.py bdist_wheel, install into .wheel-smoke, then run
python -I validation/wheel_smoke.py.


## Final verification for this continuation

- Windows integrated runner: exit 0; 43 Rust integration tests, all C++
  reference comparisons, 200 selected unmodified upstream Python tests.
- Latest Python suite: 62 passing on Windows and Ubuntu Linux (including
  final deconvolution initialization and spectral System Jacobian additions).
- cargo fmt --check and clippy --all-targets -D warnings pass.
- Windows wheel rebuilt, installed into .wheel-smoke, and loaded its packaged
  DLL under isolated Python. Numerical guide examples and new derivative ABI
  smoke checks pass.
- feature_audit.json maps 49 public numerical methods, 7 API-scope exclusions
  and 6 state operations. Mapping is not a source-wide equivalence proof.
- Latest logs: validation/current_run.log (combined), final_python.log,
  linux_python.log and wheel_build.log. The combined log predates the last
  three Python-only tests; final_python.log and linux_python.log supersede its
  Python test count.

## Previous continuation record

# Current status - September 16, 2026

The numerical port is still incomplete: full physical-parameter derivatives
and broader internal-operator reference validation remain. The table/history
below predates this continuation; the following updates and RESTART_HANDOFF.md
supersede conflicting degree/solver/platform statements.

- Maps and emitted/Doppler calculations admit combined degree32.
- High-degree reflected/oblate fields and elliptical systems use stable direct
  harmonic integration. Gravity/RV products use stable harmonic projection.
  RV requires map+limb+3<=32; gravity requires map+gravity+limb<=32.
  Python limb/gravity filter degrees individually remain capped at20.
- Reflected/oblate occultor geometry derivatives are available through
  Map.flux_gradient; full specialized parameter Jacobians remain unavailable.
- Doppler L1 spectra, joint nonlinear fits, fixed baselines and tempered-map
  baseline handling are implemented. Covariances are conditional/local.
- Native cube SVD, numeric image/spectrum loading, smoothing/extents and
  System.render are implemented. File decoding is caller-side.
- Windows and Ubuntu Rust/Python tests pass. macOS is untested.
- Upstream AD and mixed high-degree specialized comparison reports live in
  validation/specialized_gradients_report.json and specialized_high_report.json.
- The complete native library does not call C++ or Theano at runtime.

## Earlier detailed implementation map (historical boundaries)

# Port status and source map

Status: broad native numerical implementation; **complete numerical coverage not yet established**.
Started 2026-09-15 in response to the user's request to port starry to Rust.

The user clarified completion means **complete numerical functionality through
Rust with a new Python interface**. Drop-in Python/Theano compatibility, lazy
symbolic execution, and reproducing upstream convenience signatures are not
completion requirements. The optional eager compatibility adapter is a testing
aid, not the primary API.

Reference: `b72dff08588532f96bd072f2f1005e227d8e4ed8` in the sibling
`starry-upstream` checkout. Do not advance it without regenerating comparisons.

| Upstream area | Rust implementation | Status |
|---|---|---|
| `basis.h`: A1, polynomial products | `src/basis.rs` | Equivalent Legendre/azimuthal recurrences; matches source fixtures |
| `basis.h`: A2 inverse, rT | `src/basis.rs` | A2 inverse translated; equivalent disk moment recurrence |
| `ellip.h`: scalar CEL | `src/elliptic.rs` | Translated; tested |
| `solver.h`: emitted-light values | `src/solver.rs` | Analytic I/J/K/L/H solution ported, degree capped at 20 |
| `wigner.h`: map rotation | `src/rotation.rs` | Polynomial substitution through degree 8; stable spherical quadrature projection above; Wigner kernel not ported |
| `filter.h`: limb/map product | `src/basis.rs`, `src/map.rs` | Canonical polynomial product and limb normalization |
| `limbdark.h`: specialized solver | General analytic emitted solver | Functional limb flux; specialized speed/derivative recurrences not ported |
| Intensity derivatives | `src/basis.rs`, `src/map.rs` | Analytic coordinate derivatives, handles zero coordinates |
| Geometry derivatives | `src/gradients.rs` | Shape derivative + converged boundary quadrature; AD kernel unported |
| Numerical flux cross-check | `src/occultation.rs` | Independent inner analytic/outer adaptive integration |
| Python Map | `python/_starry_api.py`, `src/session.rs` | Persistent handles, broadcasting, spectral channels, coefficient/limb indexing, designs |
| Reflected light, Oren-Nayar | `src/reflected.rs`, `src/oren_coefficients.rs` | Analytic phase moments, source polynomial scattering; adaptive occultations and finite-source sampling |
| Oblate and gravity-darkened maps | `src/oblate.rs`, `src/occultation.rs` | Source gravity profile/SHT, numerical ellipse occultation, spectral filters |
| Doppler maps and radial velocity | `src/doppler.rs`, `src/rv.rs` | Source chord/velocity recurrences, native convolution and differential-rotation RV |
| Keplerian multi-body systems, delays, exposures | `src/orbit.rs`, `src/system.rs` | Kepler solver, barycentric states, apparent positions, midpoint exposures, overlapping occultors, emission/reflection/RV |
| Orientation/projection | `Map::projected`, Python inc/obl/theta | Observer projection implemented |
| Multi-wavelength arrays, image/spot tools | `src/surface.rs`, Python interface | Channel batches, spots, orthographic/rectangular/Mollweide render, sampled/raster fitting |
| Linear solves, priors, posterior inference | `src/inference.rs`, Python Map methods | Full Gaussian covariance/prior solves, marginal likelihood, posterior draws |
| First emitted parameter Jacobian | `src/differentiation.rs` | Phase, inclination, obliquity, amplitude, harmonics, limb and occultor geometry |
| Doppler grids, inverse problems, unrestricted operator | `src/doppler.rs`, `python/_inverse_api.py` | Linear spline grids, map/spectrum conditional Gaussian fits, normalized-map and alternating fits, matrix-free spectral-map forward/transpose |
| Oblate primary systems | `src/conic.rs`, `src/system.rs` | Elliptical transits/eclipses, including reflected background bodies; oblate secondaries unsupported upstream too |
| Map and System inference | `python/_inverse_api.py`, `src/inference.rs` | Fixed-geometry scalar/spectral Gaussian solves, wavelength-correlated map priors, marginal likelihood and draws |
| Specialized surface images | `src/imaging.rs` | Limb, RV, oblate gravity, finite-source reflection, exact/approximate Oren-Nayar intensities |
| Pixel transforms and surface constraints | `src/pixels.rs`, `src/surface.rs` | Mollweide samples, regularized inverse, analytic surface derivatives, bounded multistart minimum, limb validity |
| Packaging | `setup.py`, `pyproject.toml` | Platform wheel bundles native library; Windows smoke-tested |
| Symbolic/automatic differentiation interface | None | Out of requested API scope; native derivative coverage remains explicitly limited |

## Numerical differences from the source

- Uses `f64` only, with dense partial-pivot solves in place of Eigen sparse LU.
- Computes Vieta polynomial coefficients by multiplication of `(1-t)^u
  (delta+t)^v`, rather than the source's double-binomial recurrence.
- Uses separate scalar CEL calls instead of the vectorized CEL kernel.
- Handles no occultation, complete occultation, and centered occultation at the
  API boundary. Centered polynomial moments use a positive incomplete-beta
  recurrence. This avoids infinite intermediate quantities.
- Retains upstream's regularization near `b=r=0.5` after independent integration
  exposed the same instability with rounded position angles.
- Omits the high-degree J-series refresh beyond degree 25 because the public
  degree cap is 20. Does not claim arbitrary-degree accuracy.
- The numerical integrator's error estimate is heuristic, per polynomial
  moment. It is not a rigorous bound on a weighted harmonic map or its flux.
- No finite-difference derivatives are used in the library. Finite differences
  are used only as one independent check of the boundary derivative.

## Remaining numerical work and boundaries

1. Degree above 20, stability at the upper supported degrees, and a broader
   numerical reference campaign. Intensity/derivative recurrences and rotations
   are now stable through tested degree 20; flux/filter polynomial cancellation
   still needs investigation. Do not just raise the cap.
2. Native derivatives beyond the first emitted-light Jacobian: reflected/oblate
   geometry and scattering, orbital parameters, and higher derivatives remain
   unimplemented. No symbolic backend is required by the clarified scope.
3. Broader independent Doppler inverse and multi-body delay/exposure validation.
   The unrestricted spectral-map forward/transpose exists; a general joint
   spectral-map posterior solver is not supplied. Alternating component fits
   return conditional covariances, not a joint posterior covariance. Normalized
   map fits return a local linearized covariance and convergence status.
4. System inference fits scalar and spectral maps at fixed geometry. Simultaneously
   inferring a luminous primary and reflected secondary maps is bilinear and is
   rejected.
5. Cross-platform build/runtime tests and performance characterization.

Adaptive reflected/oblate integration is a numerical implementation choice,
not by itself a missing forward feature. The source's specialized analytic
kernels remain unported. System RV exposures now integrate flux-weighted RV.
System uses a system-wide barycenter, independent Kepler trajectories and
midpoint exposure samples; upstream-wide convention parity has not been proven.
Rotational RV with an oblate primary is explicitly rejected rather than using
an incorrect circular silhouette.

Python angles default to radians (degrees are opt-in); orbital lengths/masses/
times are solar radii, solar masses, and days, and velocities are m/s. No Astropy
unit conversion or lazy symbolic execution is provided. A degree cap of 20
applies to combined polynomial filters. These are material API boundaries,
not full equivalence.

Oblate f=0 deliberately remains an exact sphere. The upstream oblate kernel
floors f at 1e-6; validation records this known difference and checks the exact
sphere against upstream emitted-light kernels instead.

Do not label this repository a completed full port before the remaining numerical
coverage is established. No MESA port, cluster transfers, EB restart, or SN-project
changes are part of this work.
