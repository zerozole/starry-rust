# Numerical API

The implementation lives in Rust. Python supplies array handling, model
composition and optimization orchestration through a bundled native library.
No upstream C++ or Theano is used at runtime. See PORT_STATUS.md for boundaries.

## Maps and derivatives

```python
import numpy as np
from starry_rust import Map, Primary, Secondary, System, DopplerMap

m = Map(3, udeg=2, angle_unit='deg', inc=65, obl=20)
m[1], m[2] = .3, .1
m[1, 1] = .2
flux = m.flux(theta=np.arange(360), xo=.2, yo=.4, ro=.1)
jac = m.flux_jacobian(theta=30, xo=.2, yo=.4, ro=.1)
```

`flux_jacobian` returns flux plus derivatives for theta, inc, obl, xo, yo,
ro, amp, y and u. Angle derivatives use `angle_unit`. Emitted, reflected and
oblate maps are supported; specialized physical parameters are described below.
Intensity and surface derivatives use normalized harmonic recurrences.
Rotations above degree 8 use spherical quadrature projection to avoid the
ill-conditioned polynomial conversion. Maps and emitted/Doppler calculations
admit combined degree 32. Limb and gravity filters individually have a Python
limit of 20. RV requires map+limb+3 <= 32; oblate gravity products require
map+gravity+limb <= 32. Reflected illumination is evaluated as a local field.

`flux_gradient` supports emitted, reflected and oblate maps, returning
derivatives with respect to occultor x, y and radius. Reflected maps accept
`xs`, `ys`, `zs` and `Rs`, including finite sources. Use `flux_jacobian` for
source/scattering and gravity-parameter derivatives. Coincident unit disks have
an undefined radius derivative.

`kepler_state` returns native relative positions/velocities with Jacobians and
Hessians. Parameter order: a, period, ecc, inc, omega, node, t0, t. Semimajor
axis and period are independent inputs.

## Surface representations

```python
surface = Map(5)
surface.spot(contrast=.2, radius=.3, lat=.4, lon=.8)
image = surface.render(res=100, projection='rect')
transforms = surface.surface_transforms(ridge=1e-8)
weights = surface.amp * surface.y
pixels = transforms['forward'] @ weights
restored = transforms['inverse'] @ pixels
d_latitude = transforms['dlat'] @ restored
minimum = surface.minimize(bounds=((-1., 1.), (-2., 2.)))
```

The transform grids use an equal-area Mollweide projection. You can pass
explicit `lat` and `lon` samples instead. `forward`, `dlat` and `dlon` act on
absolute harmonic weights. `inverse` is a ridge-regularized inverse. To
differentiate pixels, compose the derivative matrix with `inverse`; the
implementation avoids allocating a square pixel-by-pixel matrix. These are
intrinsic surface transforms, excluding viewing, illumination and limb filters.
Reflected-map transforms are in albedo units.

`load` accepts an equirectangular numeric image with north at the top.
Samples lie at pixel centers. Optional `extent=(lon0,lon1,lat0,lat1)` uses
`angle_unit`; `smoothing` is a Gaussian harmonic width in radians.
`force_positive=True` rescales higher coefficients using a sampled minimum,
without certifying global positivity. File decoding is caller-side.
`load_samples` accepts coordinates, intensities and weights. `minimize` is
a bounded multistart search; use `return_info=True` for convergence information.
It does not certify a global minimum. `limbdark_is_physical` checks polynomial
limb intensity positivity and monotonicity.

Rendering supports orthographic, rectangular and Mollweide images. Orthographic
images include the chosen surface mode and limb filtering. Reflected images
accept source coordinates, finite source radius `Rs`, source sample count and
exact or polynomial Oren-Nayar scattering. Nonorthographic images describe the
intrinsic surface coordinates. Spectral image channels are on the final axis.

## Gaussian map and System inference

`set_data(flux, C=...)` takes a variance (not a standard deviation), a vector of
variances or a full observation covariance. Spectral arrays flatten in C order:
epoch/sample first, wavelength last. An array matching the spectral flux shape
is interpreted as per-observation variances.

`set_prior(mu=..., L=...)` takes a mean and covariance on absolute weights
`amp*y`. Its default mean is zero in the new API. A spectral map accepts a
full covariance including wavelength correlations; a Ny-by-Ny covariance is
repeated independently across wavelengths. The coefficient order is harmonic
first, wavelength last. `solve` returns a posterior mean and lower Cholesky
factor. `lnlike` evaluates the marginalized Gaussian likelihood.

```python
star = Map(1, nw=2, inc=1.1, obl=.4)
system = System(Primary(star), Secondary(Map(amp=0), porb=3, r=.1))
times = np.linspace(-.1, .1, 51)
data = system.flux(times)
star.set_prior(L=1.)
system.set_data(data, C=1e-6)
mean, cholesky = system.solve(t=times)
```

Bodies with priors are fitted; others remain fixed. Scalar bodies may be shared
across spectral channels. Geometry is fixed during these linear solves.
Reflection includes primary luminosity: jointly fitting that primary and
reflected map weights would be bilinear and is explicitly rejected.

## Doppler spectra and inverse fits

```python
surface = Map(2, inc=1.1)
wav = np.linspace(500., 500.2, 51)
rest_wav = np.linspace(499.8, 500.4, 151)
rest = 1 - .5*np.exp(-((rest_wav-500.1)/.02)**2)
d = DopplerMap.from_wavelengths(surface, wav, rest_wav, rest,
                               veq=20000., oversample=1)
phases = np.linspace(0, 6, 12)
spectra = d.flux(phases, normalize=False)
mean, cholesky = d.solve(spectra, theta=phases, solver='map',
                        flux_err=.001, normalize=False)
```

Wavelengths may use any consistent positive unit; velocities are m/s.
Rest spectra are linearly resampled onto a padded uniform logarithmic grid,
with linear extrapolation beyond supplied wavelengths. The lower-level
constructor accepts already padded spectra, `log_spacing` and `half_width`.
`wav0` exposes the padded grid for `from_wavelengths` models.

`design_matrix(fix_spectrum=True)` gives the map operator;
`design_matrix(fix_map=True)` gives the rest-spectrum operator. `map_jacobian`
includes the continuum denominator derivative for normalized spectra.
`solve` supports map, spectrum and alternating component fits. Normalized map
fits use a local linearization; alternating fits return conditional covariance,
not a joint posterior covariance. Inspect `fit_converged` before using a fit.

`spectral_map_dot(weights, phases)` accepts shape `(Ny, len(wav0))`, allowing
an independent spectrum for every harmonic. It applies the unnormalized
operator without constructing a dense design. The exact transpose accepts
epoch-by-wavelength residuals with `transpose=True`, including the transpose
of any output wavelength interpolation. It uses the first component's degree
and viewing angle. Multicomponent models can be represented by summing their
outer products `amp*y[:, None]*spectrum[None, :]` when these agree.

## Additional loading and solver modes

`solve(..., solver='nonlinear')` jointly fits maps/spectra with the same
Gaussian-prior objective as upstream, using native designs/solves with Python
Gauss-Newton and line-search orchestration. `joint_covariance` is a local
approximation, not an exact nonlinear posterior.

`solver='spectrum', spectral_method='L1'` uses native OpsDoppler.L1 iteration.
Controls: `spectral_lambda`, `spectral_eps`, `iterations`, `tolerance`. Returns
`spectrum`, iteration/convergence information and `covariance=None`.

For normalized observations, `baseline=` supplies a known per-epoch baseline.
`solve_tempered` implements upstream's fixed-spectrum tempered map equations
with additive epoch-correlated `baseline_var`, `log_temperature=(12,0)` and
`steps=50`. Covariance is conditional on the final iteration. `solve_bilinear_tempered` follows upstream's combined map/spectrum schedule
with an explicitly supplied initial spectrum (the model's current spectra).
It accepts `logT0`, `logTf` (base ten), `steps`, scalar/dictionary priors,
`baseline_var`, and L1/L2 spectrum controls. As in source, equal log endpoints
use their literal value as the temperature; nonpositive temperatures reject.
`initialization='deconvolve'` uses upstream's uniform-kernel L1 residual
initialization; `continuum_index` identifies the internal-grid normalization
sample. It requires shared component inclination. The default initialization
uses current spectra.

`DopplerMap.load(maps=..., spectra=...)` loads numeric components.
`load(cube=...)` accepts latitude-by-longitude-by-rest-wavelength cubes. Native
truncated Jacobi SVD produces one map/spectrum pair per model component.
Spectra must match the padded grid, or supply `rest_wavelengths` for a
wavelength-grid model. Factor signs may differ while preserving the model.

`System.render(t, res=...)` returns per-body unmasked `images`, apparent
`positions` and `radii`. Rotation phases use observation times, as in upstream
OpsSystem.render. Positions and radii are in solar radii.

## Rust modules

Public modules include `map`, `reflected`, `oblate`, `rv`, `doppler`, `orbit`,
`system`, `inference`, `imaging`, `surface`, `pixels` and `differentiation`.
Rust angles are radians; orbital lengths, masses and times are solar radii,
solar masses and days. Native Gaussian routines support full covariances and
a diagonal-noise path that avoids an observation-by-observation matrix.

All reported validation tolerances apply to finite test sets. They do not
establish arbitrary-degree, arbitrary-contact or universal precision guarantees.

## Native first derivatives

`Map.flux_jacobian` supports emitted, reflected and oblate maps. Common keys
include flux, viewing angles, amplitude, harmonic/limb coefficients and occultor
geometry. Reflected maps add source coordinates, roughness and source radius;
oblate maps add flattening, rotation rate, gravity exponent, polar temperature
and wavelength. `illumination_jacobian` and `oblate_jacobian` provide the focused
physical subsets. Angle derivatives follow the map's configured angle unit.

`DopplerMap.flux_jacobian(theta, normalize=True)` differentiates phase,
inclination, equatorial speed, logarithmic wavelength spacing and limb
coefficients. Normalized spectra include continuum derivatives. Rest samples
and output interpolation weights are fixed for the spacing derivative.

`System.flux_jacobian(t, total=True)` returns flux, time and exposure derivatives,
plus `body0.theta0`, `body0.prot`, `body0.t0` and corresponding keys for each
secondary. Secondary keys also include ecc, inc, w, Omega, and porb or a,
according to the constructor's active orbital parameter. These obey Kepler's
law at fixed masses. Secondary t0 changes both transit and spin reference
epochs. `total=False` retains the body axis, matching System.flux. Spectral
channels are retained. Overlaps, light delay and exposure stencils are included.
Body mass and radius derivatives are returned as `bodyN.m` and `bodyN.r`.
Mass derivatives include barycentric motion, Kepler's law and light delay;
radius derivatives include silhouettes and finite-source illumination. Surface
maps and their physical filters are fixed by default. Set `surface=True` to
include `bodyN.map.inc`, `obl`, `amp`, `roughness`, `f`, `omega`, `beta`,
`tpole`, `wav`, `y` and `u`. Inactive physical modes have zero derivatives.
Coefficient derivatives append a coefficient axis after any spectral axis.

Singular limits are not silently approximated: rough reflected phase poles,
degenerate finite sources, exact Doppler support edges and velocity-floor
kinks may reject. Coincident silhouettes and changes in foreground ordering
can lack a two-sided derivative. Rotation period zero disables rotation; its
reported derivative follows that disabled branch. See NUMERICAL_LIMITS.md for
the full boundary contract and source derivative-order audit.

### Radial velocity derivatives

`Map.rv_jacobian(theta=..., xo=..., yo=..., ro=..., zo=...)` returns `rv`,
`theta`, `inc`, `obl`, `xo`, `yo`, `ro`, `amp`, `y`, `u`, `veq` and `alpha`.
Angles follow the map angle unit; velocities are m/s. Spectral channels and
broadcast input dimensions follow Map.rv. The derivative of the amplitude is
zero because amplitude cancels. At zero visible flux, results follow the
forward API's zero branch; this does not assert differentiability across contact.

`System.rv_jacobian(t, keplerian=True, total=True, surface=False)` returns
`rv` and the same orbital/spin/mass/radius keys as flux, plus `bodyN.veq` and
`bodyN.alpha`. Exposure integration uses the flux-weighted rotational ratio;
the orbital term is evaluated at the central time. `surface=True` adds the
map parameters above. Velocities are m/s; masses and radii use solar units.

## System orbital conventions

Set `System(..., orbit_convention='starry')` when reproducing upstream:

- `rv(total=False)` row zero contains only the primary rotational anomaly.
  Row j contains the primary's reflex contribution from companion j plus that
  companion's rotational anomaly. `total=True` sums these contributions.
- Flux/render geometry uses independent primary-relative Kepler orbits.
  With light delay it uses exoplanet 0.4.5's quadratic delay approximation;
  surface phases use observation times. The delay expression is rationalized
  locally to avoid cancellation.
- `position` returns the sum of pairwise stellar reflex positions for the
  primary and each companion's pairwise barycentric position. Only companions
  receive the source's optional light-delay correction. These are different
  from the relative coordinates returned by `render`.

The default `orbit_convention='barycentric'` preserves the port's extension:
all bodies share a systemwide barycenter; each emitting body has an iterated
retarded time used for geometry and rotation; RV rows give physical body
velocities plus rotational anomalies. `position` returns instantaneous
coordinates, while `render` returns apparent coordinates. This mode is not
numerically identical to upstream for multi-body positions or light delays.
Neither convention is an interacting N-body integrator.

```python
primary = Primary(Map(1, rv=True, veq=4000.), m=1.)
companion = Secondary(Map(rv=True, amp=.01), porb=3., m=.01, r=.1)
binary = System(primary, companion, orbit_convention='starry',
                light_delay=True, texp=.001, oversample=3)
t = np.array([-.02, .02, .4])
derivatives = binary.rv_jacobian(t, surface=True)
np.testing.assert_allclose(derivatives['rv'], binary.rv(t), atol=1e-7)
# One linearized weighted least-squares update for the companion mass.
observations = binary.rv(t) + np.array([.1, -.1, .05])
sigma = np.full(t.shape, 2.)
design = derivatives['body1.m'] / sigma
mass_step = np.dot(design, (observations-derivatives['rv'])/sigma) / np.dot(design, design)
assert np.isfinite(mass_step)
```

This is a local parameter update, not a global fit or uncertainty estimate.
Re-evaluate derivatives after updating parameters and respect their bounds.

## Doppler continuum choice

The default integrated continuum divides by the unit-spectrum disk kernel.
Pass `continuum_index=<output wavelength index>` to either Doppler constructor
for source-style normalization by that spectral sample. Each epoch then has
unit flux at that output sample. `spectrum_jacobian` and `map_jacobian` include
the full normalization derivative. Both return `(flattened_flux, Jacobian)`
with flattened observation rows and map-weight/rest-spectrum columns.
A normalized fixed-map `design_matrix`
is conditional on the current baseline; it is not the spectrum Jacobian.
Tempered bilinear fitting caches the initial continuum map design, as upstream.
The deconvolution initializer's `continuum_index` argument separately selects
an internal padded-grid sample.

```python
sampled = DopplerMap(Map(1), np.ones(13), log_spacing=2e-5,
                     half_width=2, veq=15000., continuum_index=2)
np.testing.assert_allclose(sampled.flux([0., .4])[:, 2], 1.)
flat_flux, spectral_design = sampled.spectrum_jacobian([0., .4])
assert np.isfinite(spectral_design).all()
```
