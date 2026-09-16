# Remaining numerical-port work

Verified against current code and the pinned upstream checkout on 2026-09-16.
This is the list of identified completion items. The operator audit may expose
additional gaps; public method-name coverage alone does not establish parity.

## 1. Complete System derivatives

- Compose surface-map parameters into the system flux Jacobian: coefficients,
  amplitudes, limb coefficients, viewing inclination/obliquity, reflected
  scattering, and oblate/gravity parameters. Individual Map derivatives and
  fixed-geometry System coefficient designs exist; unified composition remains.
- Add System RV derivatives, including Keplerian velocity, rotational velocity,
  overlaps, light delay, masses/radii and flux-weighted exposure integration.
- Validate joint primary/reflected amplitude dependencies explicitly.

Completed now: System flux derivatives for orbital elements, spin, observation
time, exposure, masses and radii. Map RV derivatives are implemented.

## 2. Resolve derivative limits against upstream

- Reflected rough-scattering phase poles and degenerate/zero-size finite-source
  limits; distinguish a removable numerical singularity from a nonexistent
  two-sided derivative.
- Doppler support edges and the velocity floor; grazing/total contacts,
  coincident silhouettes, foreground-order changes and zero-flux ratios.
- Gravity-darkening zero-effective-gravity limits and extreme flattening.
- Zero body radius/mass and disabled rotation conventions.
- Finish the derivative-order audit. Do not assume arbitrary flux Hessians are
  required: upstream integration.py GradientOp classes have no further grad
  method, and limbdark/limbdark.py explicitly rejects gradients of derivative
  outputs. Orbital Hessians already exist locally. Any additional higher-order
  requirement needs a demonstrated upstream counterpart.

## 3. Finish source-wide numerical coverage audit

- Replace broad class-to-module mappings in feature_audit.json with specific
  operator-to-implementation and operator-to-test evidence.
- Check public properties and numerical branches, not only public method names.
- Compare supported parameter ranges and combined-degree constraints with source.
- Classify each difference as implemented equivalence, intentional new API
  behavior, source limitation, or actual numerical gap.
- Check mixed oblate-primary/rotating-secondary RV behavior against upstream;
  it is currently rejected locally. Do not equate source's prohibition on a
  single map combining oblate+RV with every possible multi-body combination.

## 4. Expand independent reference tests

- Full System flux derivatives versus an independent reference: current tests
  use perturbations of the native forward model and analytic chain rules.
- Doppler physical derivatives and full RV ratios versus upstream/independent
  integration. Current source RV derivative check covers the velocity filter.
- Broader orbital conventions, multi-body delays and exposure rules.
- Doppler inverse modes, normalized fits, component degeneracy, covariance
  semantics, tempering and initialization against source outputs.
- Close remaining internal-operator reference gaps found in item 3.

## 5. Validate the supported numerical domain

- Stress degree32 maps and combined gravity/limb/RV products, including mixed
  coefficients and upper-degree derivatives; individual filters are capped at20.
- Exercise small/large occultors, contact neighborhoods, near-pole viewing,
  finite sources, highly eccentric orbits and extreme supported gravity values.
- Use analytic/high-precision references where upstream itself loses precision.
- Record tolerances, convergence failures and supported bounds; fix unexplained
  discrepancies instead of raising tolerances without evidence.

## 6. Finish Python API and documentation

- Expose remaining numerical modes and derivatives with consistent spectral
  shapes, broadcasting, angle units, names and error behavior.
- Add examples for new System/RV derivatives and parameter-fitting workflows.
- Consolidate historical status sections so superseded limitations are clear.
- Ensure README, API guide, completion checklist and validation reports agree.

## 7. Final release verification

- Run the integrated Rust/Python/upstream suite after the last implementation.
- Rebuild and install-test the final platform wheel; verify it loads its own
  native library and all shipped examples run in isolation.
- Repeat Windows/Linux checks after relevant native changes. macOS remains
  untested; either validate it before advertising support or state that boundary.
- Close every required checklist item with code and validation evidence.

## Scope boundaries

Python/Theano drop-in compatibility, symbolic graphs, GUI plotting wrappers,
arbitrary new joint posteriors and source-unsupported map combinations are not
automatically required. Real-data astronomy validation and performance tuning
can follow the numerical port; they are not proof of numerical equivalence.

Existing solver/loading functionality is not pending: L1/L2 spectra, nonlinear
fits, alternating tempering, deconvolution initialization, numeric image loading
and cube SVD are implemented. Broader source comparisons remain in item 4.
