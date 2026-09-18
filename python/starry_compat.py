"""Eager Rust-backed compatibility adapter for upstream starry conventions.

Use ``import starry_compat as starry``. This does not emulate symbolic execution.
See NUMERICAL_LIMITS.md for the supported domain and intentional differences.
"""
import numpy as np
import starry_rust as native
from _starry_api import _dll, array, ptr, check


class _Config:
    quiet = True
    @property
    def lazy(self):
        return False
    @lazy.setter
    def lazy(self, value):
        if value:
            raise NotImplementedError("symbolic starry execution is not ported")


config = _Config()
__version__ = "0.1.0-rust"


class Map(native.Map):
    def __init__(self, ydeg=0, udeg=0, nw=None, *, lazy=False, **kwargs):
        if lazy:
            raise NotImplementedError("symbolic starry execution is not ported")
        kwargs.setdefault('angle_unit', 'deg')
        super().__init__(ydeg, udeg, nw, **kwargs)

    def reset(self, amp=1.):
        self.y[:] = 0.
        self.y[0] = 1.
        self.u[:] = 0.
        self.u[0] = -1.
        self.amp = float(amp) if self.nw is None else np.broadcast_to(amp, (self.nw,)).copy()
        self._sync()

    def set_prior(self, mu=None, L=1., cho_L=None):
        if mu is None:
            mu = np.zeros(self.Ny)
            mu[0] = 1.
        return super().set_prior(mu=mu, L=L, cho_L=cho_L)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            index = self._indices(key)
            if np.ndim(index)==0 and index==0:
                raise ValueError('y00 is fixed at one; set amp instead')
            target = self.__getitem__(key)
            value = np.asarray(value)
            if value.ndim>np.ndim(target) and np.squeeze(value).shape==np.shape(target):
                value=np.squeeze(value)
            if value.ndim==2 and value.shape != np.shape(target) and value.T.shape == np.shape(target):
                value = value.T
        super().__setitem__(key, value)

    def intensity(self, *args, x=None, y=None, **kwargs):
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError('provide both x and y')
            radius2 = np.asarray(x)**2+np.asarray(y)**2
            if np.any(radius2 > 1.):
                raise ValueError('point is outside the disk')
            kwargs['mu'] = np.sqrt(1-radius2)
        if self.reflected and x is None:
            names = ('xs', 'ys', 'zs', 'Rs')
            inputs = np.broadcast_arrays(*[np.atleast_1d(kwargs.pop(name, default)) for name, default in zip(names, (0., 0., 1., 0.))])
            channels = []
            for i in range(inputs[0].size):
                source = dict(zip(names, [value.ravel()[i] for value in inputs]))
                value = np.asarray(super().intensity(*args, **kwargs, **source))
                channels.append(np.atleast_1d(value) if self.nw is None else np.atleast_2d(value))
            return np.stack(channels, axis=-1)
        return np.atleast_1d(super().intensity(*args, **kwargs)) if self.nw is None else np.atleast_2d(super().intensity(*args, **kwargs))

    def flux(self, *args, **kwargs):
        value = super().flux(*args, **kwargs)
        return np.atleast_1d(value) if self.nw is None or kwargs.get('integrated',False) else np.atleast_2d(value)

    def rv(self, **kwargs):
        value = super().rv(**kwargs)
        return np.atleast_1d(value) if self.nw is None else np.atleast_2d(value)

    def render(self, *args, **kwargs):
        kwargs['_upstream_grid'] = True
        value = super().render(*args, **kwargs)
        return value if self.nw is None else np.moveaxis(value, -1, 0)

    @property
    def solution(self):
        if self._solution is None:
            raise ValueError('call solve first')
        return self._solution[0], np.linalg.cholesky(self._solution[1])


class Primary(native.Primary):
    def __init__(self, map, *, prot=0., **kwargs):
        super().__init__(map, prot=prot, **kwargs)
    @property
    def _r(self):
        return self.r


class Secondary(native.Secondary):
    def __init__(self, map, *, prot=0., **kwargs):
        super().__init__(map, prot=prot, **kwargs)
    @property
    def _r(self):
        return self.r


class System(native.System):
    def _get_periods(self):
        return np.array([b.porb if b.porb is not None else
                         2*np.pi*np.sqrt((b.a*6.957e8)**3/(1.3271244e20*(self.primary.m+b.m)))/86400
                         for b in self.secondaries])
    def flux(self, t, total=True, integrated=False):
        result = super().flux(t, total=total)
        if integrated and any(b.map.nw is not None for b in self.bodies):
            result = result.sum(axis=-1)
        return result


DopplerMap = native.DopplerMap


class _Ops:
    @staticmethod
    def nroots(coefficients, lo, hi):
        p=array([lo,hi,*np.asarray(coefficients)[::-1]])
        out=np.empty(1)
        check(_dll.starry_surface_operation(0,2,ptr(p),len(p),ptr(out),1))
        return int(out[0])


_c_ops = _Ops()
