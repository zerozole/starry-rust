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

# Starry Rust continuation - 2026-09-16

This section supersedes the September 15 handoff retained below.
Objective: full numerical Rust functionality with a new Python interface.
User reinforced using the upstream repository as the reference.
Reference remains clean at b72dff08588532f96bd072f2f1005e227d8e4ed8.

## Added and verified

- Stable harmonic surface integration for high-degree reflected/oblate fields,
  conic silhouettes, gravity products, and RV products. Emitted degree cap is32.
- Native reflected/oblate x/y/radius shape derivatives, including finite sources.
- Native upstream L1 iterated-ridge solver and Python sparse spectrum fits.
- Joint nonlinear Doppler fitting, fixed baselines and upstream tempered-map
  equations; conditional/local covariance labels. Joint optimization uses
  Gauss-Newton on the source objective rather than source NAdam.
- Native Jacobi SVD, numeric Doppler cube/image/spectrum loading, smoothing,
  image extents, positivity adjustment, and numerical System.render.
- Internal core.py operator inventory in validation/feature_audit.json.

43 Rust integration tests pass on Windows and Ubuntu Linux. The final Windows
Python suite contains48 tests; current_run.log records the combined run.
Linux passed47 before final System.render; linux_python.log records the refresh.
200 selected unmodified upstream Python tests pass.
Existing581 C++ cases pass;63 new reflected shape derivatives match upstream AD
(max1.22e-15);6 mixed degree13/16 specialized cases match upstream (max7.52e-13).
Degree32 oblate sphere agrees with harmonic-ring integration. L1 and SVD have
independent algebraic checks. These are finite tests, not universal guarantees.

## Remaining before a full numerical-port claim

- Reflected source/scattering, oblate flattening/gravity, Doppler physical
  derivatives and full System Jacobian composition; higher flux derivatives.
- Complete internal-operator reference coverage, singular geometries, wider
  Doppler/orbit convention checks and degree32/extreme-parameter stress tests.
- Exact combined upstream Doppler tempering/initialization schedules differ.
  Conditional L1/L2, bilinear, nonlinear and tempered-map modes are available.
- macOS is untested. Linux is source-tested; Windows wheel is distributed.

Do not treat a general spectral-map posterior or primary/reflected bilinear
posterior as missing upstream functionality without a source counterpart.
Upstream System.solve is fixed-design Gaussian inference.

## Reproduction

Windows: C:/Users/ATAL/miniforge3/python.exe validation/record_run.py
Underlying runner: validation/run.py --reuse-oracles --upstream-python.
The reflected oracle now seeds AD derivatives; rebuild its executable if old.
No upstream headers were changed.

Linux Rust: cargo test --offline --release --target-dir target-linux
Linux Python: validation/linux_python.sh under Ubuntu22.04.
Rust is /home/atal/.cargo/bin/cargo. System Python lacks NumPy; the script
reuses the existing EB venv read-only with bytecode writes disabled. Override
STARRY_TEST_PYTHON if needed. STARRY_RUST_LIBRARY selects the shared library.
No EB files were changed.

Wheel: python setup.py bdist_wheel; install into .wheel-smoke; run
python -I validation/wheel_smoke.py. NumPy is the sole runtime dependency.
Historical cluster STOP remains. No agents, EB/SN edits, MESA, SSH or transfers.

## Historical September 15 handoff (superseded where inconsistent)

# Starry Rust port handoff - 2026-09-15

User objective: full port. Latest clarification: complete numerical functionality
through Rust with a new Python interface. Drop-in Python/Theano compatibility
is not required. Do not revive it as a completion blocker.

Project: D:\Games\starry-rust; reference: D:\Games\starry-upstream, pinned
unmodified at b72dff08588532f96bd072f2f1005e227d8e4ed8.
Read NUMERICAL_API.md, PORT_STATUS.md and VALIDATION.md for the current boundary.

Native coverage: emitted/reflected/oblate flux, finite sources, Oren-Nayar,
velocity fields, Keplerian systems, elliptical primary silhouettes, overlapping
occultors, delays and exposure integration, Gaussian inference, Doppler
convolution/design/inverse operations, spectral-map forward/transpose, surface
rendering/fitting/spots/minimization, pixel operators, first emitted derivatives.
Map and System inference support spectral channels and correlated map priors.
Python is array/orchestration glue; there is no upstream C++ runtime.

Recent stability work: normalized harmonic intensity/directional derivatives,
including poles; rotations above degree8 use exact-degree spherical projection,
tested through degree20. Flux/filter polynomial cancellation still needs work
before expanding the combined degree cap above20.

Validation: 33 Rust tests, 34 Python tests, 200 selected original upstream greedy
tests, 581 C++ reference cases. Formatting, Clippy, release build and the combined
runner pass. See validation/*.json. The starry_compat adapter is a testing aid,
not the primary API or a full upstream compatibility promise.
Platform wheels bundle the Rust library. Installed wheels require NumPy only.

Remaining: support/validate degrees above20; broader derivatives beyond first
emitted parameters; unrestricted spectral-map posterior solver; joint luminous
primary/reflected-map inference; broader Doppler/orbit references and cross-
platform tests. Adaptive reflected/oblate integration already supplies forward
functionality. Source analytic kernels remain a performance/derivative task.
Do not label the full numerical port complete before the remaining coverage is
established.

Conventions: Rust radians, Python radians or angle_unit='deg'; Y00=1/pi;
index=l*l+l+m; first-degree harmonics proportional to [y,z,x]. Raw y00 is
independent of amplitude. Keep half-radius contact regularization and the
recorded exact-sphere difference from upstream's minimum flattening floor.

Environment: Rust1.96.1 MSVC, Python3.13.13 at
C:\Users\ATAL\miniforge3\python.exe, g++ at C:\MinGW\bin\g++.exe.
Pytest9.1.1 installed with approval into workspace .test-deps only.
No agents, EB/SN source changes, MESA work, SSH, transfers or cluster jobs.
Historical cluster STOP remains in effect. Memory search found no starry entries.
