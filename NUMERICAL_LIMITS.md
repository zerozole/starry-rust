# Numerical scope and boundary behavior

This is the supported numerical contract, not a claim that every real input or
every symbolic derivative has a finite answer. The reference is starry commit
`b72dff08588532f96bd072f2f1005e227d8e4ed8`.

## Supported profile

- Binary64 arithmetic. Map degree 0 through32. Combined emitted/limb, oblate
  gravity/limb and Doppler degrees must be <=32; RV reserves three degrees for
  its velocity field. Python limb/gravity filters individually admit degree20.
- Inclinations are in [0,pi], flattening in [0,1), bound-orbit eccentricity in
  [0,1), positive orbital period/axis. Gravity parameters require omega in
  [0,1], beta>=0 and positive temperature/wavelength. Doppler veq is in [0,c).
- Native flux, intensity, image, convolution, inference and surface functions;
  first parameter Jacobians of Map flux/RV, Doppler flux and System flux/RV.
  System surface Jacobians are opt-in because their cost scales with map size.
- Orbital state Jacobians and Hessians. There is no generic symbolic graph or
  promise of arbitrary-order derivatives of every composed Python expression.
- Error-controlled adaptive quadrature may reject nonconvergence. Reported
  tolerances and maxima apply to finite test cases, not a universal error bound.

## Singular and branch boundaries

| Situation | Behavior |
|---|---|
| Coincident stellar/occultor unit disks | Radius flux gradient rejects; the two-sided derivative is undefined. |
| Ordinary first/last contact | Vanishing covered area uses the zero branch; nearby cases are tested. |
| Coincident foreground silhouettes | Shape Jacobian rejects the ambiguous exposed-boundary decomposition. |
| Foreground ordering changes or exactly coincident depth | Derivatives hold the selected ordering fixed; no derivative through a discrete ordering change is claimed. |
| Completely occulted or zero-flux RV | Returns the forward model's zero branch. At a ratio singularity this is a convention, not a smooth extension. |
| Doppler vsini floor | Forward kernel uses max(veq*sin(inc),1 m/s). Jacobian rejects the kink (within 1e-12). Below the floor the velocity derivative is zero. |
| Doppler chord support edge | Jacobian rejects abs(x)=1 within 1e-12; moving support can have a singular derivative. |
| Rough reflected source exactly on the phase pole | Cartesian source Jacobian rejects the undefined azimuth chart (transverse distance <=1e-12 of source distance). Lambert poles have explicit finite derivatives. Upstream also constructs this chart with atan2(xs,ys). |
| Finite-source radius zero | Point-source forward value. With multiple samples and source distance>=1, returns the analytic right derivative of the source's facing hemisphere. This differs from differentiating the source's selected zero-radius branch. At distance<1 there is no adjacent valid finite-source disk, so the radius Jacobian rejects. |
| Degenerate nonzero effective source disk | Forward boundary can exist, but the source-radius Jacobian rejects the square-root singularity. |
| Zero effective gravity | For beta>0, the Planck intensity and first derivatives take their zero limit. Simultaneous beta=0 does not have that limit and rejects when sampled. |
| Rotation period zero | Rotation is disabled; the derivative of that branch is zero. Rapidly rotating nonzero periods have no continuous zero-period limit. |
| Zero body radius or mass | Fixed branch derivatives; zero-radius occultors contribute no area. Python requires positive primary+secondary orbital mass. |
| Nonpositive limb average or normalized gravity baseline | Rejects invalid normalization. Limb positivity everywhere is a separate physicality check. |

At zero visible flux, a coefficient Jacobian may describe signed mathematical
maps rather than a physically meaningful source. Fitting routines require
finite positive noise/prior covariances; degeneracy is reflected in their stated
conditional or local covariance, not hidden by an exact-posterior claim.

## Deliberate interface and numerical differences

- `orbit_convention='starry'` supplies independent-pair positions, stellar
  reflex RV contributions, exoplanet 0.4.5 quadratic light delays and observed
  spin phases. The default barycentric convention is a numerical extension:
  systemwide barycenter, iterated per-body delays and physical body RVs.
  Source-style `position` and `render` use different coordinate definitions;
  see NUMERICAL_API.md. Neither convention includes N-body interactions.
- Python is a new API. Units are explicit, angles default to radians, images
  use the documented numeric grid convention, and file decoding is caller-side.
- Doppler normalization defaults to an integrated continuum. Set
  continuum_index to an output wavelength index for upstream-style sampled
  continuum normalization. Conditional spectrum designs hold that baseline
  fixed; spectrum_jacobian differentiates the full quotient.
- Doppler nonlinear optimization uses Gauss-Newton/backtracking on the source
  objective instead of NAdam. Returned nonlinear covariances are local.
- Fixed-design System Gaussian inference does not solve a joint bilinear
  primary/reflected posterior; neither does the referenced source solver.
- Exactly zero oblateness is an exact sphere. Upstream floors its oblate
  integration parameter to 1e-6. High-degree source polynomial cancellation
  is resolved against independent high-precision integrals.
- Upstream requires all System maps to agree on RV mode and forbids combining
  oblate/reflected/RV modes within a map. An oblate primary with an RV secondary
  is therefore not an omitted upstream capability. Oblate secondaries are also
  rejected upstream. Some local mixed emitted/reflected and spectral cases are
  numerical extensions, tested separately.
- Source GradientOp classes for integration, rotation, filtering, polynomial
  basis and spots do not implement another grad method. Its specialized limb
  operation explicitly rejects gradients of derivative outputs. Arbitrary
  flux Hessians are not a port-completion requirement; see the machine-readable
  source_boundaries_report.json audit.

## Platform and evidence boundaries

Windows x64 and Ubuntu Linux native builds/Python execution are tested.
The supplied wheel targets Windows x64. macOS is not advertised as tested.
No speedup over upstream and no astronomy population/real-data validation are
claimed. Original upstream tests run through an eager adapter; direct C++ and
independent analytic comparisons are separate evidence.
