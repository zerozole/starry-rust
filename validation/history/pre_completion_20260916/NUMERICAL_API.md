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
ro, amp, y and u. Angle derivatives use `angle_unit`. It currently supports
emitted maps; coefficient designs are available for specialized maps too.
Intensity and surface derivatives use normalized harmonic recurrences.
Rotations above degree 8 use spherical quadrature projection to avoid the
ill-conditioned polynomial conversion. Maps and emitted/Doppler calculations
admit combined degree 32. Limb and gravity filters individually have a Python
limit of 20. RV requires map+limb+3 <= 32; oblate gravity products require
map+gravity+limb <= 32. Reflected illumination is evaluated as a local field.

`flux_gradient` supports emitted, reflected and oblate maps, returning
derivatives with respect to occultor x, y and radius. Reflected maps accept
`xs`, `ys`, `zs` and `Rs`, including finite sources. Source/scattering and
gravity-parameter derivatives are not included. Coincident unit disks have
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
maps and their physical filters remain fixed in this API.

Singular limits are not silently approximated: rough reflected phase poles,
degenerate finite sources, exact Doppler support edges and velocity-floor
kinks may reject. Coincident silhouettes and changes in foreground ordering
can lack a two-sided derivative. Rotation period zero disables rotation; its
reported derivative follows that disabled branch. Full system surface-map composition and System RV Jacobians remain gaps.
Higher-order requirements are being checked against actual source support;
arbitrary flux Hessians are not assumed to be an upstream feature.

### Radial velocity derivatives

`Map.rv_jacobian(theta=..., xo=..., yo=..., ro=..., zo=...)` returns `rv`,
`theta`, `inc`, `obl`, `xo`, `yo`, `ro`, `amp`, `y`, `u`, `veq` and `alpha`.
Angles follow the map angle unit; velocities are m/s. Spectral channels and
broadcast input dimensions follow Map.rv. The derivative of the amplitude is
zero because amplitude cancels. At zero visible flux, results follow the
forward API's zero branch; this does not assert differentiability across contact.

System flux Jacobians now include `bodyN.m` and `bodyN.r`; these are derivatives
per solar mass and solar radius. Surface-map parameters remain fixed.
