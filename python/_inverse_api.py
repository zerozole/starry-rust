"""Design operators and inference orchestration for native Rust models."""
import ctypes as ct
import math
import numpy as np
from _starry_api import _dll, _ptr, array, ptr, check, covariance

_dll.starry_doppler_design.argtypes = [ct.c_uint64, _ptr, ct.c_size_t, _ptr,
    ct.c_size_t, _ptr, ct.c_size_t, ct.c_uint32, _ptr, ct.c_size_t]
_dll.starry_interpolation.argtypes = [_ptr, ct.c_size_t, _ptr, ct.c_size_t, _ptr]
_dll.starry_spectral_map_dot.argtypes = [ct.c_uint64, _ptr, ct.c_size_t, _ptr,
    ct.c_size_t, ct.c_size_t, _ptr, ct.c_size_t, ct.c_uint32, _ptr, ct.c_size_t]
_dll.starry_linear_solve_diagonal.argtypes = [_ptr, ct.c_size_t, ct.c_size_t,
    _ptr, _ptr, _ptr, _ptr, _ptr]


def interpolate(input_grid, output_grid):
    source, target = array(input_grid), array(output_grid)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("wavelength grids must be one dimensional")
    result = np.empty((len(target), len(source)))
    check(_dll.starry_interpolation(ptr(source), len(source), ptr(target),
                                   len(target), ptr(result)))
    return result


def infer(design, data, noise, mean=None, prior=None):
    design, data, noise = array(design), array(data).ravel(), array(noise)
    if design.ndim != 2 or design.shape[0] != len(data):
        raise ValueError("inference data/design shape mismatch")
    n, p = design.shape
    if (mean is None) != (prior is None):
        raise ValueError("supply both prior mean and covariance")
    if mean is not None:
        mean = array(np.broadcast_to(mean, (p,)))
        prior = covariance(prior, p)
    result = np.empty(p + p*p + 2)
    if noise.ndim < 2:
        noise = array(np.broadcast_to(noise, (n,)))
        operation = _dll.starry_linear_solve_diagonal
    else:
        noise = covariance(noise, n)
        operation = _dll.starry_linear_solve
    check(operation(ptr(design), n, p, ptr(data), ptr(noise),
                    None if mean is None else ptr(mean),
                    None if prior is None else ptr(prior), ptr(result)))
    return result[:p].copy(), result[p:-2].reshape(p, p).copy(), result[-2], result[-1]


class DopplerMixin:
    def flux_jacobian(self,theta=0.,normalize=True):
        """Native phase, inclination, velocity, spacing and limb derivatives.

        Inclination is a common perturbation of all component inclinations;
        each component uses its Map.angle_unit. Spectrum samples and any
        wavelength interpolation operator are held fixed. Exact velocity-floor
        kinks and chord-support edges have no two-sided derivative.
        """
        phases=array(np.atleast_1d(theta))
        if phases.ndim!=1 or not len(phases):raise ValueError('invalid phases')
        rows=5+len(self.u);n=self.spectra.shape[1]-2*self.half_width
        _dll.starry_doppler_jacobian.argtypes=[ct.c_uint64,_ptr,ct.c_size_t,ct.c_double,_ptr,ct.c_size_t,_ptr,ct.c_size_t]
        frames=[]
        for phase in phases:
            total=np.zeros((rows,n+1))
            for m,rest in zip(self.maps,self.spectra):
                m._sync();parameters=array([m.inc*m._angle if self.inc is None else self.inc*m._angle,self.veq,self.log_spacing,self.half_width,*self.u])
                result=np.empty_like(total)
                check(_dll.starry_doppler_jacobian(m._handles[0],ptr(parameters),len(self.u),phase*m._angle,ptr(rest),len(rest),ptr(result),result.size))
                result[1:3]*=m._angle;total+=result
            baseline=total[:,-1].copy();values=total[:,:-1]
            selected=self.continuum_index is not None
            if selected:
                if hasattr(self,'_output_operator'):values=values@self._output_operator.T
                index=self._continuum_column(values.shape[-1]);baseline=values[:,index].copy()
            if normalize:
                if baseline[0]==0:raise ValueError('zero Doppler continuum')
                values[0]/=baseline[0]
                values[1:]=(values[1:]-baseline[1:,None]*values[0])/baseline[0]
            if not selected and hasattr(self,'_output_operator'):values=values@self._output_operator.T
            frames.append(values.T)
        result=frames[0] if np.ndim(theta)==0 else np.stack(frames)
        output={name:result[...,i] for i,name in enumerate(('flux','theta','inc','veq','log_spacing'))}
        output['u']=result[...,5:]
        return output

    def load(self, *, maps=None, spectra=None, cube=None, rest_wavelengths=None,
             smoothing=0.,ridge=1e-9):
        """Load numeric component images/spectra or a spectral surface cube.

        Cubes have (latitude,longitude,rest wavelength) shape, north at top.
        Native truncated SVD produces the requested number of components.
        Rest wavelengths optionally resample spectra onto the padded wav0 grid.
        """
        if cube is not None and (maps is not None or spectra is not None):raise ValueError('choose cube or component inputs')
        count=len(self.maps)
        if cube is not None:
            cube=array(cube)
            if cube.ndim!=3 or np.any(~np.isfinite(cube)):raise ValueError('invalid spectral cube')
            h,w,n=cube.shape;pixels=h*w
            if count>min(pixels,n):raise ValueError('cube rank smaller than component count')
            out=np.empty(count*(pixels+n))
            _dll.starry_low_rank.argtypes=[_ptr,ct.c_size_t,ct.c_size_t,ct.c_size_t,_ptr]
            check(_dll.starry_low_rank(ptr(cube),pixels,n,count,ptr(out)))
            maps=np.moveaxis(out[:pixels*count].reshape(h,w,count),-1,0)
            spectra=out[pixels*count:].reshape(count,n)
        images=None if maps is None else array(maps)
        if images is not None:
            if count==1 and images.ndim==2:images=images[None,...]
            if images.ndim!=3 or len(images)!=count:raise ValueError('supply one image per component')
        rest=None if spectra is None else np.atleast_2d(array(spectra))
        if rest is not None:
            if rest.shape[0]!=count or np.any(~np.isfinite(rest)):raise ValueError('invalid component spectra')
            if rest_wavelengths is not None:
                if not hasattr(self,'wav0'):raise ValueError('rest wavelengths require a wavelength-grid model')
                operator=interpolate(rest_wavelengths,self.wav0)
                if rest.shape[1]!=operator.shape[1]:raise ValueError('spectrum/wavelength mismatch')
                rest=rest@operator.T
            if rest.shape!=self.spectra.shape:raise ValueError('spectra must match padded rest grid')
        saved=[(m.y.copy(),np.array(m.amp,copy=True)) for m in self.maps]
        try:
            if images is not None:
                for m,image in zip(self.maps,images):m.load(image,smoothing=smoothing,ridge=ridge)
        except Exception:
            for m,(y,amp) in zip(self.maps,saved):m.y=y;m.amp=amp;m._sync()
            raise
        if rest is not None:self.spectra=array(rest)

    def render(self,component=0,**kwargs):
        return self.maps[component].render(**kwargs)

    def intensity(self,component=0,**kwargs):
        return self.maps[component].intensity(**kwargs)

    def sht_matrix(self,component=0,**kwargs):
        return self.maps[component].sht_matrix(**kwargs)

    def spectral_map_dot(self, value, theta=0., *, transpose=False):
        """Apply the unrestricted, unnormalized spectral-map operator.

        Forward input has shape (Ny, padded_rest_wavelengths); each harmonic
        has an independent spectrum. Transpose input has shape (epochs, output
        wavelengths). Uses the first component's degree and viewing angle.
        Amplitudes are included in input values. No dense design is allocated.
        """
        m=self.maps[0]
        phases=array(np.atleast_1d(theta)*m._angle)
        if phases.ndim!=1 or not len(phases):raise ValueError('invalid phases')
        n=self.spectra.shape[1];nout=n-2*self.half_width
        values=array(value)
        if transpose:
            nw=self._output_operator.shape[0] if hasattr(self,'_output_operator') else nout
            if values.shape!=(len(phases),nw):raise ValueError('expected epoch by wavelength input')
            if hasattr(self,'_output_operator'):values=array(values@self._output_operator)
            output=np.empty((m.Ny,n))
        else:
            if values.shape!=(m.Ny,n):raise ValueError('expected harmonic by padded wavelength input')
            output=np.empty((len(phases),nout))
        parameters=array([m.inc*m._angle if self.inc is None else self.inc*m._angle,
                          self.veq,self.log_spacing,self.half_width,*self.u])
        check(_dll.starry_spectral_map_dot(m._handles[0],ptr(parameters),len(self.u),
            ptr(phases),len(phases),n,ptr(values),values.size,int(transpose),ptr(output),output.size))
        if not transpose and hasattr(self,'_output_operator'):output=output@self._output_operator.T
        return output

    @classmethod
    def from_wavelengths(cls, maps, wavelengths, rest_wavelengths, spectra, *,
                         veq, maximum_velocity=None, oversample=1, inc=None, u=(),continuum_index=None):
        """Resample input rest spectra to a padded logarithmic internal grid.

        Uses the upstream default linear-spline interpolation convention,
        including linear extrapolation beyond the supplied rest grid.
        """
        wav = array(wavelengths)
        if wav.ndim != 1 or len(wav) < 2 or np.any(~np.isfinite(wav)) or np.any(wav <= 0) or np.any(np.diff(wav) <= 0):
            raise ValueError("observed wavelengths must increase and be positive")
        if not isinstance(oversample, int) or not 1 <= oversample <= 100:
            raise ValueError("invalid oversampling")
        velocity = veq if maximum_velocity is None else maximum_velocity
        if not 0 <= veq <= velocity < 299792458.:
            raise ValueError("invalid maximum velocity")
        count = (len(wav)*oversample) | 1
        log_grid = np.linspace(np.log(wav[0]), np.log(wav[-1]), count)
        spacing = log_grid[1] - log_grid[0]
        half_width = max(1, int(np.ceil(np.arctanh(velocity/299792458.)/spacing)))
        if half_width > 100000:
            raise ValueError("kernel is too large")
        padded = np.exp(log_grid[0] + (np.arange(count+2*half_width)-half_width)*spacing)
        rest = np.atleast_2d(array(spectra))
        interpolation = interpolate(rest_wavelengths, padded)
        if rest.shape[1] != interpolation.shape[1]:
            raise ValueError("rest spectrum/wavelength shape mismatch")
        result = cls(maps, rest @ interpolation.T, log_spacing=spacing,
                     half_width=half_width, veq=veq, inc=inc, u=u,continuum_index=continuum_index)
        result.wav = wav.copy()
        result.wav0 = padded
        result.wav_internal = np.exp(log_grid)
        result._output_operator = interpolate(result.wav_internal, wav)
        return result

    def _native_design(self, theta, mode):
        phases = array(np.atleast_1d(theta))
        if phases.ndim != 1 or not len(phases):
            raise ValueError("phases must be a nonempty vector")
        nout = self.spectra.shape[1] - 2*self.half_width
        designs = []
        for m, spectrum in zip(self.maps, self.spectra):
            m._sync()
            parameters = array([m.inc*m._angle if self.inc is None else self.inc*m._angle,
                                self.veq, self.log_spacing, self.half_width, *self.u])
            radians = array(phases*m._angle)
            cols = len(spectrum) if mode == 2 else m.Ny
            output = np.empty((len(phases)*nout, cols))
            check(_dll.starry_doppler_design(m._handles[0], ptr(parameters), len(self.u),
                ptr(radians), len(phases), ptr(spectrum), len(spectrum), mode,
                ptr(output), output.size))
            if hasattr(self, "_output_operator"):
                output = np.einsum("ij,tjk->tik", self._output_operator,
                                   output.reshape(len(phases), nout, cols)).reshape(-1, cols)
            designs.append(output)
        return np.hstack(designs)

    def _weights(self):
        return np.concatenate([m.amp*m.y for m in self.maps])

    def _set_weights(self, weights):
        offset = 0
        for m in self.maps:
            y = weights[offset:offset+m.Ny]
            m.amp = y[0] if y[0] != 0 else 1.
            m.y = y/m.amp
            offset += m.Ny

    def flux(self, theta=0., normalize=True):
        selected=self.continuum_index is not None
        result = super().flux(theta, normalize and not selected)
        if hasattr(self, "_output_operator"):
            result = result @ self._output_operator.T
        if normalize and selected:
            baseline=result[...,self._continuum_column(result.shape[-1])]
            if np.any(baseline==0):raise ValueError('zero Doppler continuum')
            result=result/baseline[...,None]
        return result

    def _continuum_column(self,n):
        index=self.continuum_index
        if not isinstance(index,int) or not 0<=index<n:raise ValueError('continuum_index must select an output wavelength')
        return index

    def _baseline_design(self,theta):
        if self.continuum_index is None:return self._native_design(theta,1)
        design=self._native_design(theta,0);nt=len(np.atleast_1d(theta));nw=design.shape[0]//nt
        return np.repeat(design.reshape(nt,nw,-1)[:,self._continuum_column(nw),:],nw,axis=0)

    def baseline(self, theta=0., full=False):
        baseline = self._baseline_design(theta) @ self._weights()
        baseline = baseline.reshape(len(np.atleast_1d(theta)), -1)
        return baseline if full else baseline[:, 0]

    def design_matrix(self, theta=0., *, fix_spectrum=False, fix_map=False, normalize=False):
        if fix_map == fix_spectrum:
            raise ValueError("choose exactly one of fix_map or fix_spectrum")
        design = self._native_design(theta, 2 if fix_map else 0)
        if normalize:
            if fix_spectrum:
                raise ValueError("normalized map flux is nonlinear; use map_jacobian")
            baseline = self.baseline(theta, full=True).ravel()
            if np.any(baseline == 0):
                raise ValueError("zero Doppler continuum")
            design = design/baseline[:, None]
        return design

    def map_jacobian(self, theta=0., normalize=True):
        design = self._native_design(theta, 0)
        weights = self._weights()
        flux = design @ weights
        if normalize:
            baseline_design = self._baseline_design(theta)
            baseline = baseline_design @ weights
            if np.any(baseline == 0):
                raise ValueError("zero Doppler continuum")
            flux /= baseline
            design = (design-flux[:, None]*baseline_design)/baseline[:, None]
        return flux, design

    def spectrum_jacobian(self,theta=0.,normalize=True):
        """Exact rest-spectrum derivative, including selected-continuum quotient."""
        design=self._native_design(theta,2);flux=design@self.spectra.ravel()
        if normalize:
            if self.continuum_index is None:
                baseline=self.baseline(theta,full=True).ravel();design=design/baseline[:,None];flux=flux/baseline
            else:
                nt=len(np.atleast_1d(theta));nw=len(flux)//nt
                base_design=np.repeat(design.reshape(nt,nw,-1)[:,self._continuum_column(nw),:],nw,axis=0)
                baseline=base_design@self.spectra.ravel()
                if np.any(baseline==0):raise ValueError('zero Doppler continuum')
                flux=flux/baseline;design=(design-flux[:,None]*base_design)/baseline[:,None]
        return flux,design

    def dot(self, value, theta=0., *, fix_spectrum=False, fix_map=False, transpose=False, normalize=False):
        design = self.design_matrix(theta, fix_spectrum=fix_spectrum,
                                    fix_map=fix_map, normalize=normalize)
        return (design.T if transpose else design) @ value

    def solve(self, flux, *, theta=0., solver="map", flux_err=1., normalize=False,
              prior_mean=None, prior_cov=1., iterations=30, tolerance=1e-9,
              spectral_method='L2', spectral_lambda=1e5, spectral_eps=1e-12,
              baseline=None):
        """Gaussian map/rest-spectrum fits and alternating bilinear fits.

        Normalized map fitting uses the exact quotient Jacobian and a line
        search. Reported covariance is the local Gaussian approximation in that
        case. Bilinear fitting supports normalized or unnormalized data and
        reports conditional covariances plus a joint Gauss-Newton approximation.
        """
        data = array(flux).ravel()
        errors=array(flux_err)
        if errors.ndim==1 and np.ndim(flux)==2 and errors.shape==(np.shape(flux)[0],):errors=errors[:,None]
        variance = np.broadcast_to(errors, np.shape(flux)).ravel()**2
        if np.any(~np.isfinite(variance)) or np.any(variance<=0):raise ValueError('flux errors must be finite and positive')
        if spectral_method not in ('L1','L2'):raise ValueError('spectral_method must be L1 or L2')
        if spectral_method=='L1' and solver!='spectrum':raise ValueError('L1 currently requires solver="spectrum"')
        if baseline is not None:
            if not normalize:raise ValueError('baseline requires normalized observations')
            shape=np.shape(flux)
            if len(shape)!=2:raise ValueError('expected epoch by wavelength data')
            base=np.broadcast_to(array(baseline),(shape[0],))
            if np.any(~np.isfinite(base)) or np.any(base==0):raise ValueError('invalid fixed baseline')
            return self.solve(array(flux)*base[:,None],theta=theta,solver=solver,
                flux_err=np.sqrt(variance).reshape(shape)*np.abs(base[:,None]),normalize=False,
                prior_mean=prior_mean,prior_cov=prior_cov,iterations=iterations,tolerance=tolerance,
                spectral_method=spectral_method,spectral_lambda=spectral_lambda,spectral_eps=spectral_eps)
        if not isinstance(iterations, int) or iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid iteration settings")
        if solver == "bilinear":
            old_maps = self._weights().copy()
            old_spectra = self.spectra.copy()
            map_mean=np.broadcast_to(prior_mean.get('map',old_maps) if isinstance(prior_mean,dict) else old_maps if prior_mean is None else prior_mean,old_maps.shape).copy()
            spectrum_mean=np.broadcast_to(prior_mean.get('spectrum',old_spectra.ravel()) if isinstance(prior_mean,dict) else old_spectra.ravel() if prior_mean is None else prior_mean,old_spectra.size).copy()
            map_prior=covariance(prior_cov['map'] if isinstance(prior_cov,dict) else prior_cov,len(old_maps))
            spectrum_prior=covariance(prior_cov['spectrum'] if isinstance(prior_cov,dict) else prior_cov,old_spectra.size)
            np.linalg.cholesky(map_prior);np.linalg.cholesky(spectrum_prior)
            history = []
            scores=[]
            try:
                for _ in range(iterations):
                    maps = self.solve(flux, theta=theta, solver="map", flux_err=flux_err,
                                      prior_mean=map_mean, prior_cov=map_prior,normalize=normalize,iterations=iterations,tolerance=tolerance)
                    spectra = self.solve(flux, theta=theta, solver="spectrum", flux_err=flux_err,
                                         prior_mean=spectrum_mean, prior_cov=spectrum_prior,normalize=normalize)
                    residual = data-self.flux(theta, normalize=normalize).ravel()
                    history.append(float(np.sum(residual**2/variance)))
                    dm=self._weights()-map_mean;ds=self.spectra.ravel()-spectrum_mean
                    scores.append(history[-1]+dm@np.linalg.solve(map_prior,dm)+ds@np.linalg.solve(spectrum_prior,ds))
                    if len(scores)>1 and abs(scores[-1]-scores[-2]) <= tolerance*(1+abs(scores[-2])):
                        break
                weights=np.r_[self._weights(),self.spectra.ravel()]
                f,map_jac=self.map_jacobian(theta,normalize)
                spectrum_jac=self.spectrum_jacobian(theta,normalize)[1]
                jac=np.hstack((map_jac,spectrum_jac))
                prior=np.zeros((len(weights),len(weights)));p=len(map_mean)
                prior[:p,:p]=map_prior;prior[p:,p:]=spectrum_prior
                local=infer(jac,data-f+jac@weights,variance,np.r_[map_mean,spectrum_mean],prior)
            except Exception:
                self._set_weights(old_maps)
                self.spectra = old_spectra
                raise
            self.fit_converged=len(scores)>1 and abs(scores[-1]-scores[-2]) <= tolerance*(1+abs(scores[-2]))
            return {"map": maps, "spectrum": spectra, "chi_square": history,'objective':scores,
                    'joint_mean':weights,'joint_covariance':local[1],
                    'covariance_kind':'local Gauss-Newton approximation',"converged":self.fit_converged}
        if solver=='nonlinear':
            return self._solve_joint(flux,theta,variance,normalize,prior_mean,prior_cov,iterations,tolerance)
        if solver not in ("map", "spectrum"):
            raise ValueError("solver must be map, spectrum, or bilinear")
        weights = self._weights() if solver == "map" else self.spectra.ravel().copy()
        mean = weights.copy() if prior_mean is None else np.broadcast_to(prior_mean, weights.shape).copy()
        prior = covariance(prior_cov, len(weights))
        if spectral_method=='L1':
            design=self.design_matrix(theta,fix_map=True,normalize=normalize)
            if design.shape[0]!=len(data):raise ValueError('data/design shape mismatch')
            normal=array(design.T@(design/variance[:,None]))
            rhs=array(design.T@((data-design@mean)/variance))
            out=np.empty(len(weights)+2)
            _dll.starry_l1.argtypes=[_ptr,_ptr,ct.c_size_t,ct.c_double,ct.c_size_t,ct.c_double,ct.c_double,_ptr]
            check(_dll.starry_l1(ptr(normal),ptr(rhs),len(weights),spectral_lambda,iterations,spectral_eps,tolerance,ptr(out)))
            weights=mean+out[:-2]
            self.spectra=weights.reshape(self.spectra.shape)
            self.fit_converged=bool(out[-1])
            return dict(spectrum=weights.copy(),iterations=int(out[-2]),converged=self.fit_converged,
                        covariance=None,method='L1 iterated ridge')
        if solver == "spectrum" or not normalize:
            design = self.design_matrix(theta, fix_map=solver=="spectrum",
                                        fix_spectrum=solver=="map", normalize=normalize)
            result = infer(design, data, variance, mean, prior)
            if solver == "map":
                self._set_weights(result[0])
            else:
                self.spectra = result[0].reshape(self.spectra.shape)
            self._solution = result[:2]
            self.fit_converged = True
            return result[0], np.linalg.cholesky(result[1])
        inverse_prior = np.linalg.inv(prior)
        def objective(w):
            f, _ = self.map_jacobian(theta, True)
            delta = w-mean
            return float(np.sum((data-f)**2/variance)+delta@inverse_prior@delta)
        original = weights.copy()
        converged = False
        try:
            score = objective(weights)
            for _ in range(iterations):
                f, jacobian = self.map_jacobian(theta, True)
                result = infer(jacobian, data-f+jacobian@weights, variance, mean, prior)
                direction = result[0]-weights
                if np.linalg.norm(direction) <= tolerance*(1+np.linalg.norm(weights)):
                    converged = True
                    break
                accepted = False
                for reduction in range(25):
                    trial = weights+direction*2.**(-reduction)
                    self._set_weights(trial)
                    new_score = objective(trial)
                    if new_score <= score:
                        accepted = True
                        break
                if not accepted:
                    self._set_weights(weights)
                    break
                weights, score = trial, new_score
            f, jacobian = self.map_jacobian(theta, True)
            result = infer(jacobian, data-f+jacobian@weights, variance, mean, prior)
        except Exception:
            self._set_weights(original)
            raise
        self._solution = weights.copy(), result[1]
        self.fit_converged = converged
        return weights.copy(), np.linalg.cholesky(result[1])

    def _solve_joint(self,flux,theta,variance,normalize,prior_mean,prior_cov,iterations,tolerance):
        """Same joint Gaussian-prior objective as upstream solve_nonlinear.

        Uses exact native design derivatives and line-searched Gauss-Newton
        steps instead of the source's NAdam iteration; geometry remains fixed.
        """
        original_maps=self._weights().copy();original_spectra=self.spectra.copy()
        p=len(original_maps);weights=np.r_[original_maps,original_spectra.ravel()]
        mean=np.r_[np.broadcast_to(prior_mean.get('map',original_maps),p),
            np.broadcast_to(prior_mean.get('spectrum',original_spectra.ravel()),original_spectra.size)] if isinstance(prior_mean,dict) else weights.copy() if prior_mean is None else np.broadcast_to(prior_mean,weights.shape).copy()
        if isinstance(prior_cov,dict):
            prior=np.zeros((len(weights),len(weights)))
            prior[:p,:p]=covariance(prior_cov['map'],p)
            prior[p:,p:]=covariance(prior_cov['spectrum'],len(weights)-p)
        else:prior=covariance(prior_cov,len(weights))
        np.linalg.cholesky(prior)
        precision=np.linalg.inv(prior);data=array(flux).ravel()
        def assign(w):self._set_weights(w[:p]);self.spectra=w[p:].reshape(original_spectra.shape)
        def objective(w):
            residual=data-self.flux(theta,normalize=normalize).ravel();delta=w-mean
            return float(np.sum(residual**2/variance)+delta@precision@delta)
        history=[];converged=False
        try:
            score=objective(weights);history.append(score)
            for _ in range(iterations):
                f,jmap=self.map_jacobian(theta,normalize)
                jac=np.hstack((jmap,self.spectrum_jacobian(theta,normalize)[1]))
                step=infer(jac,data-f+jac@weights,variance,mean,prior)[0]-weights
                if np.linalg.norm(step)<=tolerance*(1+np.linalg.norm(weights)):
                    converged=True;break
                accepted=False
                for reduction in range(30):
                    trial=weights+step*2.**(-reduction);assign(trial);new=objective(trial)
                    if new<=score:accepted=True;break
                if not accepted:assign(weights);break
                weights=trial;score=new;history.append(score)
            f,jmap=self.map_jacobian(theta,normalize)
            jac=np.hstack((jmap,self.spectrum_jacobian(theta,normalize)[1]))
            cov=infer(jac,data-f+jac@weights,variance,mean,prior)[1]
        except Exception:
            self._set_weights(original_maps);self.spectra=original_spectra;raise
        self.fit_converged=converged
        return dict(joint_mean=weights.copy(),joint_covariance=cov,objective=history,
            converged=converged,covariance_kind='local Gauss-Newton approximation')

    def solve_bilinear_tempered(self,flux,*,theta=0.,flux_err=1.,normalize=True,
                               prior_mean=None,prior_cov=1.,logT0=2.,logTf=0.,steps=50,
                               baseline_var=1e-2,spectral_method='L2',spectral_lambda=1e5,
                               spectral_iterations=100,spectral_eps=1e-12,tolerance=1e-9,
                               initialization='current',continuum_index=0):
        """Upstream alternating temperature schedule and optional deconvolution.

        Current spectra initialize the fit. Priors stay fixed over the schedule.
        Map covariances are conditional; no joint posterior is implied. This
        follows solve_for_everything_bilinear. initialization='deconvolve'
        uses its uniform-kernel L1 residual initialization; continuum_index
        indexes the internal wavelength grid. Components must share inclination.
        """
        data=array(flux)
        if data.ndim!=2 or not data.size or data.size>4096:raise ValueError('expected at most 4096 epoch by wavelength observations')
        if not isinstance(steps,int) or steps<1 or not np.isfinite(baseline_var) or baseline_var<0:raise ValueError('invalid tempering settings')
        if not np.all(np.isfinite([logT0,logTf])) or max(abs(logT0),abs(logTf))>300:raise ValueError('invalid temperature schedule')
        errors=array(flux_err)
        if errors.shape==(len(data),):errors=errors[:,None]
        errors=np.broadcast_to(errors,data.shape)
        if np.any(~np.isfinite(errors)) or np.any(errors<=0):raise ValueError('invalid flux errors')
        original=self._weights().copy();spectra=self.spectra.copy()
        def setting(value,key,default):return value.get(key,default) if isinstance(value,dict) else default if value is None else value
        mean=np.broadcast_to(setting(prior_mean,'map',original),original.shape).copy()
        sm=np.broadcast_to(setting(prior_mean,'spectrum',spectra.ravel()),spectra.size).copy()
        prior=covariance(setting(prior_cov,'map',1.),len(original))
        sp=setting(prior_cov,'spectrum',1.)
        # Preserve the upstream equal-endpoint branch, including its literal
        # temperature convention; otherwise use a base-ten logspace.
        temperatures=np.array([10.**logTf]) if steps==1 else np.full(steps,logTf) if logT0==logTf else np.logspace(logT0,logTf,steps)
        if np.any(temperatures<=0):raise ValueError('upstream equal-endpoint schedule requires positive temperature')
        baseline=np.ones(len(data));history=[]
        try:
            if initialization=='deconvolve':
                from _starry_api import Map,DopplerMap
                inclinations=[(m.inc if self.inc is None else self.inc)*m._angle for m in self.maps]
                if not np.allclose(inclinations,inclinations[0],rtol=0,atol=1e-14):raise ValueError('deconvolution initialization requires shared inclination')
                uniform=DopplerMap(Map(inc=inclinations[0]),np.ones(self.spectra.shape[1]),
                    log_spacing=self.log_spacing,half_width=self.half_width,veq=self.veq,u=self.u)
                kernel=uniform.design_matrix(0.,fix_map=True)
                average=data.mean(axis=0)
                if hasattr(self,'wav_internal'):average=interpolate(self.wav,self.wav_internal)@average
                if average.shape!=(kernel.shape[0],):raise ValueError('initialization wavelength shape mismatch')
                if not isinstance(continuum_index,int) or not 0<=continuum_index<len(average) or average[continuum_index]==0:raise ValueError('invalid continuum index or zero continuum')
                residual=average/average[continuum_index]-(kernel@sm.reshape(self.spectra.shape).T).mean(axis=1)
                guess=uniform.solve(residual,theta=0.,solver='spectrum',flux_err=float(errors.mean()),
                    prior_mean=0.,spectral_method='L1',spectral_lambda=spectral_lambda,
                    spectral_eps=spectral_eps,iterations=spectral_iterations,tolerance=tolerance)
                self.spectra=sm.reshape(self.spectra.shape)+guess['spectrum']
            elif initialization!='current':raise ValueError('initialization must be current or deconvolve')
            spectrum_guess=self.spectra.copy()
            continuum=self._baseline_design(theta)
            for temperature in temperatures:
                design=self.design_matrix(theta,fix_spectrum=True)
                if design.shape[0]!=data.size:raise ValueError('data/design shape mismatch')
                noise=(errors*baseline[:,None])**2*temperature
                if normalize and baseline_var:
                    noise=np.diag(noise.ravel());nw=data.shape[1]
                    for i in range(len(data)):noise[i*nw:(i+1)*nw,i*nw:(i+1)*nw]+=baseline_var
                else:noise=noise.ravel()
                fitted=infer(design,(data*baseline[:,None]).ravel(),noise,mean,prior)
                self._set_weights(fitted[0])
                if normalize:baseline=(continuum@fitted[0]).reshape(data.shape)[:,0]
                spectral=self.solve(data*baseline[:,None],theta=theta,solver='spectrum',
                    flux_err=errors*np.abs(baseline[:,None]),prior_mean=sm,prior_cov=sp,
                    spectral_method=spectral_method,spectral_lambda=spectral_lambda,
                    spectral_eps=spectral_eps,iterations=spectral_iterations,tolerance=tolerance)
                history.append(baseline.copy())
        except Exception:
            self._set_weights(original);self.spectra=spectra
            raise
        return dict(map=fitted[0],map_cholesky=np.linalg.cholesky(fitted[1]),spectrum=spectral,
                    spectrum_guess=spectrum_guess,
                    temperatures=temperatures,baseline=baseline,baseline_history=np.array(history),
                    covariance_kind='conditional covariances at alternating steps')

    def solve_tempered(self,flux,*,theta=0.,flux_err=1.,prior_mean=None,prior_cov=1.,
                       baseline_var=1e-2,log_temperature=(12.,0.),steps=50):
        """Port of upstream Solve.solve_for_map_tempered for normalized spectra.

        Iterates the continuum estimate with an epoch-correlated additive
        baseline covariance and an exponential noise-temperature schedule.
        Returns the final conditional Gaussian map covariance, not a joint
        posterior over the baseline and map. The rest spectrum is fixed.
        """
        data=array(flux)
        if data.ndim!=2 or not data.size:raise ValueError('expected epoch by wavelength data')
        if not isinstance(steps,int) or steps<1 or not np.isfinite(baseline_var) or baseline_var<0:raise ValueError('invalid tempering settings')
        logs=array(log_temperature)
        if logs.shape!=(2,) or np.any(~np.isfinite(logs)) or np.any(np.abs(logs)>700):raise ValueError('invalid temperature schedule')
        errors=array(flux_err)
        if errors.shape==(len(data),):errors=errors[:,None]
        variance=np.broadcast_to(errors,data.shape).ravel()**2
        if np.any(~np.isfinite(variance)) or np.any(variance<=0):raise ValueError('invalid flux errors')
        original=self._weights().copy();mean=original if prior_mean is None else np.broadcast_to(prior_mean,original.shape).copy()
        prior=covariance(prior_cov,len(original));design=self._native_design(theta,0);continuum=self._baseline_design(theta)
        if design.shape[0]!=data.size:raise ValueError('data/design shape mismatch')
        if data.size>4096:raise ValueError('dense baseline covariance supports at most 4096 observations')
        baseline=np.ones(data.size);nw=data.shape[1];history=[]
        for temperature in np.exp(np.linspace(*logs,steps)):
            noise=np.diag(temperature*variance*baseline**2)
            for i in range(len(data)):noise[i*nw:(i+1)*nw,i*nw:(i+1)*nw]+=baseline_var
            solution=infer(design,data.ravel()*baseline,noise,mean,prior)
            baseline=continuum@solution[0];history.append(baseline.reshape(data.shape)[:,0].copy())
        self._set_weights(solution[0]);self._solution=solution[:2]
        return dict(map=solution[0],cholesky=np.linalg.cholesky(solution[1]),baseline=history[-1],
            baseline_history=np.array(history),covariance_kind='final conditional Gaussian covariance')


class SystemMixin:
    def flux_jacobian(self,t,total=True,*,surface=False):
        """Orbital, spin, mass and radius derivatives at fixed surface maps.

        Orbital derivatives obey the constructor's Kepler relation. Each
        secondary t0 changes both transit and rotation reference epochs.
        Derivatives at contact or foreground-order changes need not exist.
        """
        return self._system_jacobian(t,total,8 if surface else 5)

    def rv_jacobian(self,t,keplerian=True,total=True,*,surface=False):
        """Native system RV derivatives, including flux-weighted exposure averaging."""
        return self._system_jacobian(t,total,(9 if keplerian else 10) if surface else (6 if keplerian else 7))

    def _system_jacobian(self,t,total,mode):
        counts={b.map.nw for b in self.bodies if b.map.nw is not None}
        if len(counts)>1:raise ValueError('inconsistent wavelength counts')
        raw=np.stack([self._evaluate(t,mode,c) for c in range(next(iter(counts)))],axis=-1) if counts else self._evaluate(t,mode)
        def column(k):
            value=raw[:,:,k]
            return value.sum(axis=1) if total else np.swapaxes(value,0,1)
        result={'flux' if mode in (5,8) else 'rv':column(0),'time':column(1),'texp':column(2)}
        n=len(self.bodies);spin=3+7*(n-1)
        physical=spin+3*n
        for i in range(n):
            result[f'body{i}.m']=column(physical+2*i)
            result[f'body{i}.r']=column(physical+2*i+1)
            if mode not in (5,8):
                result[f'body{i}.veq']=column(physical+2*n+2*i)
                result[f'body{i}.alpha']=column(physical+2*n+2*i+1)
        for i,b in enumerate(self.bodies):
            prefix=f'body{i}.'
            result[prefix+'theta0']=column(spin+3*i)*b.map._angle
            result[prefix+'prot']=column(spin+3*i+1)
            result[prefix+'t0']=column(spin+3*i+2)
            if i:
                start=3+7*(i-1)
                gm=1.3271244e20*(self.primary.m+b.m)
                if b.porb is not None:
                    period=b.porb;a=(gm*(period*86400/(2*np.pi))**2)**(1/3)/6.957e8
                    result[prefix+'porb']=column(start+1)+column(start)*2*a/(3*period)
                    mass_chain=column(start)*a/(3*(self.primary.m+b.m))
                else:
                    a=b.a;period=2*np.pi*np.sqrt((a*6.957e8)**3/gm)/86400
                    result[prefix+'a']=column(start)+column(start+1)*1.5*period/a
                    mass_chain=-column(start+1)*period/(2*(self.primary.m+b.m))
                result['body0.m']+=mass_chain
                result[prefix+'m']+=mass_chain
                result[prefix+'ecc']=column(start+2)
                for j,key in enumerate(('inc','w','Omega')):result[prefix+key]=column(start+3+j)*b.map._angle
                result[prefix+'t0']+=column(start+6)
        if mode>=8:
            start=physical+4*n
            for i,b in enumerate(self.bodies):
                prefix=f'body{i}.map.'
                for j,key in enumerate(('inc','obl','amp','roughness','f','omega','beta','tpole','wav')):
                    result[prefix+key]=column(start+j)*(b.map._angle if key in ('inc','obl','roughness') else 1.)
                result[prefix+'y']=np.stack([column(start+9+j) for j in range(b.map.Ny)],axis=-1)
                result[prefix+'u']=np.stack([column(start+9+b.map.Ny+j) for j in range(b.map.udeg)],axis=-1) if b.map.udeg else np.empty((*column(0).shape,0))
                start+=9+b.map.Ny+b.map.udeg
        return result

    def render(self,t,*,res=100):
        """Numerical images and apparent positions, analogous to OpsSystem.render.

        Returns per-body unmasked orthographic frames; each body's frame uses
        its own radius as the length unit. Positions and radii are in solar
        radii. Images include surface illumination, not foreground silhouettes.
        As in upstream rendering, rotational phases use observation times.
        """
        times=array(np.atleast_1d(t))
        if times.ndim!=1:raise ValueError('times must be one dimensional')
        positions=self._evaluate(times,14)
        frames=[]
        for i,body in enumerate(self.bodies):
            phase=body.theta0+((times-body.t0)*2*np.pi/(body.prot*body.map._angle) if body.prot else np.zeros_like(times))
            images=[]
            for k,theta in enumerate(phase):
                kw={}
                if body.map.reflected:
                    if body.r<=0:raise ValueError('reflected rendering requires a positive radius')
                    source=(positions[k,0]-positions[k,i])/body.r
                    kw=dict(zip(('xs','ys','zs'),source));kw['Rs']=self.primary.r/body.r
                image=body.map.render(res=res,projection='ortho',theta=theta,**kw)
                if body.map.reflected:image=image*self.primary.map.amp
                images.append(image)
            frames.append(np.stack(images))
        return dict(images=tuple(frames),positions=positions,radii=np.array([b.r for b in self.bodies]))

    def design_matrix(self, t):
        """Per-body coefficient columns with all physical parameters fixed.

        A reflected secondary's columns include the current primary amplitude.
        Simultaneously solving that primary amplitude makes the model bilinear.
        """
        columns = []
        for index, body in enumerate(self.bodies):
            m = body.map
            original = m.y.copy(), m.amp
            try:
                m.amp = 1.
                for j in range(m.y.size):
                    m.y = np.zeros_like(original[0])
                    m.y.flat[j] = 1.
                    columns.append(self.flux(t, total=False)[index].ravel())
            finally:
                m.y, m.amp = original
                m._sync()
        return np.stack(columns, axis=-1)

    def set_data(self, flux, C=1., cho_C=None):
        data = array(flux).ravel()
        noise = array(C) if cho_C is None else array(cho_C)@array(cho_C).T
        if cho_C is None and np.ndim(flux)>1 and noise.shape==np.shape(flux):
            noise=noise.ravel()
        if noise.ndim == 2:
            noise = covariance(noise, len(data))
        else:
            noise = array(np.broadcast_to(noise, (len(data),)))
        self._data = data, noise

    def _inference_inputs(self, t, design_matrix):
        if not hasattr(self, "_data"):
            raise ValueError("call set_data first")
        design = self.design_matrix(t) if design_matrix is None else array(design_matrix)
        data, noise = self._data
        if design.shape != (len(data), sum(b.map.y.size for b in self.bodies)):
            raise ValueError("system design shape mismatch")
        solved = [b for b in self.bodies if b.map._prior is not None]
        if not solved:
            raise ValueError("set a Gaussian prior on at least one body map")
        if self.primary in solved and any(b.map.reflected for b in self.secondaries):
            raise ValueError("reflected-system inference requires fixed primary amplitude")
        data = data.copy()
        offset = 0
        selected, means, priors = [], [], []
        for b in self.bodies:
            block = design[:, offset:offset+b.map.y.size]
            if b in solved:
                selected.append(block)
                means.append(b.map._prior[0])
                priors.append(b.map._prior[1])
            else:
                data -= block@(b.map.amp*b.map.y).ravel()
            offset += b.map.y.size
        count = sum(len(mu) for mu in means)
        prior = np.zeros((count, count))
        offset = 0
        for cov in priors:
            prior[offset:offset+len(cov), offset:offset+len(cov)] = cov
            offset += len(cov)
        return np.hstack(selected), data, noise, np.concatenate(means), prior, solved

    def solve(self, *, t=None, design_matrix=None):
        x, data, noise, mean, prior, solved = self._inference_inputs(t, design_matrix)
        result = infer(x, data, noise, mean, prior)
        offset = 0
        for body in solved:
            m = body.map
            count=m.y.size
            weights = result[0][offset:offset+count].reshape(m.y.shape)
            m.amp = np.where(weights[0]!=0,weights[0],1.)
            m.y = weights/m.amp
            offset += count
        self._solved_bodies = solved
        self._solution = result[0], np.linalg.cholesky(result[1])
        return self._solution

    def lnlike(self, *, t=None, design_matrix=None, woodbury=True):
        x, data, noise, mean, prior, _ = self._inference_inputs(t, design_matrix)
        return infer(x, data, noise, mean, prior)[3]

    @property
    def solution(self):
        if not hasattr(self, "_solution"):
            raise ValueError("call solve first")
        return self._solution

    def draw(self, rng=None):
        mean, chol = self.solution
        rng = np.random.default_rng() if rng is None else rng
        draw = mean+chol@rng.standard_normal(len(mean))
        offset = 0
        for body in self._solved_bodies:
            m = body.map
            count=m.y.size
            weights = draw[offset:offset+count].reshape(m.y.shape)
            m.amp = np.where(weights[0]!=0,weights[0],1.)
            m.y = weights/m.amp
            offset += count
        return draw
