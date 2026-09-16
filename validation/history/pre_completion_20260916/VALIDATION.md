# Validation of the Rust port

Updated 2026-09-15. The user-confirmed target is numerical functionality in
Rust with a new Python API. The selected upstream greedy tests use an eager
compatibility adapter only as a validation aid.

## Completed checks

- `cargo fmt --all -- --check`: passed.
- `cargo clippy --offline --all-targets -- -D warnings`: passed.
- `cargo test --offline`: **33 integration tests passed**.
- `cargo build --release --offline`: passed; Rust library, DLL, and probe built.
- `python -m unittest discover -s python -v`: **34 Python tests passed**.
- `python validation/upstream_python.py`: **200 selected, unmodified upstream
  greedy tests passed**, with four pytest deprecation warnings in upstream
  parametrizations. See `validation/upstream_python_report.json` for the selection.
- Direct upstream comparisons: **581 cases passed**: 389 emitted/core,
  120 reflected, 54 oblate, and 18 exact-sphere limits.
- The transit example produced 101 finite flux samples in `validation/transit.csv`.

The upstream reference is commit
`b72dff08588532f96bd072f2f1005e227d8e4ed8`. The harness includes the original
C++ headers without editing them; it checks the commit and a clean checkout.
It invokes the same core operations used by Python starry. The separate greedy
runner exercises selected original Python tests against the Rust adapter;
it does **not** install or exercise the complete Python starry package.

## Measured upstream differences

| Quantity | Cases | Largest absolute difference |
|---|---:|---:|
| A1, A2 inverse, disk moments, degrees 0–12 | 13 | 1.46e-11 |
| Analytic flux rows, fixed geometries, degrees 0/1/2/5/10 | 95 | 6.03e-11 |
| Analytic flux rows, seeded random geometries, degrees 1/2/5/8 | 40 | 2.47e-14 |
| Independent numerical flux rows, fixed geometries | 95 | 6.03e-11 |
| Independent numerical flux rows, seeded random geometries | 40 | 1.76e-14 |
| Quadratic limb-darkened harmonic flux rows, map degrees 0/1/3/5 | 20 | 3.75e-15 |
| Boundary geometry gradients vs upstream automatic derivatives | 35 | 1.93e-10 |
| Arbitrary-axis coefficient rotations, degrees 1/2/5 | 27 | 6.45e-15 |
| Scalar CEL | 24 | 3.56e-15 |
| Reflected harmonic flux, Lambert/Oren-Nayar, degrees 0/1/3 | 120 | 2.37e-9 |
| Oblate harmonic flux, degrees 1/3/5, flattening 1e-6/0.1/0.3 | 54 | 1.10e-9 |
| Exact zero-flattening oblate limit vs emitted sphere | 18 | 1.07e-15 |

The fixed flux cases include centered, grazing, partial, complete, and no
occultation; equal radii/separations; small impact parameters; and near-contact
geometries. Geometry derivative comparisons use nondegenerate partial cases
through degree 10. See `validation/compare.py` for the actual grids and
tolerances, and `validation/report.json` for values, failure lists, and hashes.

These are maxima on the tested cases, **not global error guarantees**. The
comparison tolerance for flux and geometry-gradient rows is 2e-8 absolute;
basis coefficients use absolute plus relative tolerances. A map flux combines
these rows with its coefficients and amplitude, which can amplify errors.

## Independent checks

The Rust tests also check quantities independently of upstream output:

- Y00 normalization and explicit first-degree spherical harmonics.
- Axis values of polynomial derivatives without coordinate division.
- Latitude/longitude intensity derivatives versus central differences.
- Rotation field equivalence, per-degree power conservation, inverse rotations,
  and upstream's published first-degree rotation convention.
- Uniform disk overlap across radius/impact-parameter/position-angle grids.
- Exact centered dipole occultation.
- Limb profile center, limb, full-disk normalization, and a closed-form linear
  limb-darkening transit.
- Analytic flux versus a separate integrator at multiple position angles and
  degrees through 10.
- Boundary geometry derivatives versus finite differences of the analytic flux.
- Invalid degree, coefficient, geometry, and rotation arguments.
- Lambert analytic phase curve, inverse-square scaling, and finite-source
  averages against independent closed-form Lambert integrals.
- Kepler equation residuals through high eccentricity, orbital velocities versus
  position differences, transit/eclipse ordering, barycenter, and exposure flux.
- Overlapping occultor union without duplicate subtraction.
- RV antisymmetry and system transit-induced RV versus standalone map evaluation.
- Oblate projected area, exact spherical limit, and gravity-darkened pole intensity.
- Doppler kernel symmetry/normalization, flat continua, and convolution against
  NumPy correlation. Full upstream Doppler design/inference parity is not tested.
- Gaussian posterior/marginal likelihood against independent NumPy linear algebra.
- Exact first emitted parameter Jacobian versus finite differences.
- Doppler map/spectrum design identities, normalized coefficient Jacobian,
  parameter recovery, resampling, and unrestricted spectral-map adjoint identity.
- Scalar and spectral Map/System inference, cross-wavelength priors, and shared
  fixed scalar components.
- Oblate transits/eclipses and reflected background bodies with conic silhouettes.
- Specialized images versus surface integrals and exposure RV as the ratio of
  integrated flux-weighted velocity to integrated flux.
- Pixel round trips and analytic derivative operators.
- Stable normalized harmonic recurrences and rotation round trips, field
  equivalence and degree-power conservation through degree 20; tangent
  derivatives at the poles through degree 20.
- Surface render/load roundtrip, spots, spectral broadcasting, angle units,
  persistent coefficient updates, oriented limb gradients, and ABI shape errors.

The separate integrator integrates polynomial powers of y analytically and
uses adaptive Gaussian quadrature in x, splitting at circle intersections.
The gradient routine integrates the moving occultor boundary and checks
convergence as quadrature order increases. Neither uses upstream Green
recurrences, making them useful cross-checks.

## Issues found and resolved

- The first rotation oracle applied the low-level RHS operator with the wrong
  sign. Reading `maps.py::rotate` established that it calls the RHS operator at
  **negative theta**. The harness now reproduces that behavior before transposing
  to the Rust column-vector convention. The independently tested Rust active
  rotation convention was retained.
- A rotated `b=r=0.5` geometry produced an incorrect linear-limb term when
  floating-point rounding straddled equality. Retaining upstream's five-epsilon
  regularization fixed the independent integration comparison.
- The no/full occultation cases belong to the high-level dispatch rather than
  upstream's low-level Green solver. The oracle now performs that dispatch.
- Upstream oblate `nudge_inputs` floors flattening to `STARRY_MIN_F=1e-6`.
  At f=0 its difference from an exact sphere reached 1.16e-6. Rust preserves
  the exact sphere; those cases are checked against the emitted-light oracle.
  Positive-flattening cases retain the separate oblate oracle. Upstream's
  degree-zero low-level oblate routine accesses coefficient 2, so the oracle
  tests degree >=1, including the uniform first column.
- Scalar array coercion initially changed Python scalar flux into an array;
  scalar shapes are now preserved. Cholesky shapes are checked before the ABI.
- System reflected flux now includes the primary amplitude, matching the
  upstream `OpsSystem.X` source-luminosity scaling.

## Limitations

- The full upstream Python package and complete test suite were not installed
  or exercised. Passing selected cases does not establish drop-in parity,
  which the user has explicitly excluded from the requested API scope.
- Reflected and oblate occultation values use adaptive polynomial integration,
  not the upstream specialized analytic/automatic-derivative kernels.
- Orbital, RV, Doppler, gravity-filter, and inference coverage relies mainly
  on analytic identities, independent algebra, and consistency checks; their
  entire upstream operator suites have not been compared.
- Harmonic degrees above the tested ranges, extreme coefficient scales,
  arbitrarily close contacts, and extreme radius ratios need more validation.
- Only `f64` is implemented. Numerical errors can grow at high degree through
  polynomial cancellation. Degree 20 is a size limit, not an accuracy promise.
- Geometry derivatives are first derivatives. There is no full automatic
  differentiation graph, higher-order derivative interface, or differentiation
  support through statistical inference.
- No speedup, Linux/macOS portability, or population-level scientific accuracy
  has been demonstrated.

The runner `python validation/run.py --upstream-python` includes the selected
greedy tests when pytest is installed. `--reuse-oracles` reuses the previously
built reference executables. It does not download dependencies. Pytest was
installed with permission into `.test-deps` in the workspace. Platform wheel
construction and isolated loading of the bundled Windows DLL are checked
separately from the numerical comparison runner.

The latest Windows wheel passed `python -I validation/wheel_smoke.py` after
installation into `.wheel-smoke`. It loaded the DLL from the installed package,
tested spectral map operations and native transforms, and executed every Python
example in NUMERICAL_API.md. Wheel SHA256:
`5bc7879b722f0b15b5f6eac431fe503d988c9814f0040e29645c931dc8e60b00`.
