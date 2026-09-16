# Numerical validation

Updated 2026-09-16. Reference starry commit:
`b72dff08588532f96bd072f2f1005e227d8e4ed8`. Scope and exclusions are in
NUMERICAL_LIMITS.md. Finite regression cases do not establish a universal
accuracy bound or validate astronomy populations/real data.

## Test layers

- Windows: strict formatting and Clippy, **47 Rust tests** (46 integration,
  one unit), **77 Python tests**, and **200 selected original upstream tests**.
- Ubuntu Linux: the same **47 Rust and 77 Python tests**, using a separately
  built native shared library. No claim of macOS validation.
- Original Python tests use an eager adapter to the new API. They cover only
  selected source tests; this is not a claim that the entire original suite runs.
- Direct original C++ references exercise emitted, reflected and oblate
  numerical kernels separately from the Python adapter.
- Source-extracted Doppler solver methods execute unchanged with NumPy
  primitives and shared forward designs. Solver equivalence does not itself
  validate those designs; separate convolution and chord checks do that.
- Exoplanet 0.4.5 orbital methods are vendored only as a validation reference,
  SHA256 f5ab60650ffbaffd7e8c61c35effc3954aa6bc5b831c048431d52b4871d1209a.
  The version is starry's minimum dependency. Its unmodified position/velocity
  methods use an independent Newton solve for eccentric anomaly in the harness.
  Reference URL and license are retained with the reference file.

## Measured comparisons

| Evidence | Cases | Maximum absolute discrepancy |
|---|---:|---:|
| Original emitted/basis/rotation/limb/CEL kernels | 389 | Per-operator maxima in report.json |
| Original reflected flux | 120 | 2.37e-9 |
| Original oblate flux | 54 | 1.10e-9 |
| Exact oblate sphere against emitted sphere | 18 | 1.07e-15 |
| Original reflected shape derivatives | 63 | 1.22e-15 |
| Original reflected illumination derivatives | 63 | 6.04e-16 |
| Original oblate derivatives | 40 | 1.06e-10 |
| Original RV filter and derivatives | 60 | 7.28e-12 |
| Analytic circular System flux/RV and derivatives | 88 | Flux 2.71e-14; RV 2.92e-10 m/s |
| Independent Doppler chord derivatives | 40 | 4.73e-11 |
| Independent physical disk RV integration | 24 | 2.62e-12 m/s |
| Source Doppler inverse modes and covariance | 6 | 5.51e-12 |
| Source orbital position/RV/geometry checks | 90 | Position 1.72e-11 solar radii |

Reports under validation/ contain test parameters, tolerances and reference
provenance. Six additional high-degree specialized source cases, stable harmonic
ring comparisons and Decimal high-precision cases supplement this table.

Degree32 mixed-harmonic occultations are checked against independent surface
quadrature. Upper-degree Doppler/limb and RV products, high flattening, near-pole
views, finite sources, small occultors, contact neighborhoods, overlap, complete
occultation and high-eccentricity orbital derivatives have regression coverage.
They do not imply uniform relative accuracy near vanishing observables.

System surface and nonlinear physical Jacobians also have central/one-sided
perturbation tests, including spectral channels, angle units, mass/period chain
rules, source and barycentric delay conventions, and zero primary amplitude
coupling to reflected maps. These are distinct from direct upstream AD checks.

## Source inventory and boundaries

feature_audit.json inventories 62 public methods (49 numerical counterparts,
7 deliberate API exclusions and 6 state methods), 85 properties, and 85 core
operators with explicit implementation/evidence mappings. Some operators share
component tests. The inventory demonstrates traceability, not universal parity.
source_boundaries_report.json records nine original GradientOp classes without
another grad method, the limb derivative-output restriction, and excluded map
combinations. Arbitrary composed flux Hessians are not an upstream requirement.

Known intentional differences include the default orbital and continuum
conventions, exact zero flattening instead of the source's 1e-6 floor, a new
image grid convention, optimization orchestration, and the right derivative of
a zero-size finite source. See NUMERICAL_LIMITS.md and NUMERICAL_API.md.
High-degree original polynomial cancellation is resolved with independent
harmonic/Decimal references rather than accepting the unstable source answer.

## Reproduction and artifacts

Run `python validation/record_run.py`. It captures stdout/stderr and propagates
the real exit code. Its default command reuses the locally built C++ oracles;
`python validation/run.py --upstream-python` rebuilds them with `CXX`/g++.
The source checkout and vendored Eigen headers are required for reference builds.
Use `validation/current_run.log` for the integrated result, `linux_rust.log` and
`linux_python.log` for Linux, and the JSON files for numerical measurements.

The Windows wheel is built with `python setup.py bdist_wheel`, installed into
an isolated target with pip, and checked by `python -I validation/wheel_smoke.py`.
The smoke check verifies the packaged DLL, new derivatives/source modes and
every Python code block in NUMERICAL_API.md. See wheel_build.log and wheel_smoke.log.
Original upstream parametrization deprecation warnings are retained in the log.
