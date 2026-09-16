"""Small ctypes interface to the native Rust core. Angles are radians.

This is not a drop-in replacement for the upstream starry Python package.
Build first with `cargo build --release` in the repository root.
"""
import ctypes
import math
import pathlib
import os
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
_name = "starry_rust.dll" if sys.platform == "win32" else (
    "libstarry_rust.dylib" if sys.platform == "darwin" else "libstarry_rust.so")
_packaged = pathlib.Path(__file__).resolve().parent / "_starry_native" / _name
_library = os.environ.get("STARRY_RUST_LIBRARY") or (_packaged if _packaged.is_file() else _root / "target" / "release" / _name)
_dll = ctypes.CDLL(str(_library))
_ptr = ctypes.POINTER(ctypes.c_double)
_dll.starry_flux_design.argtypes = [ctypes.c_uint32] + [ctypes.c_double] * 3 + [_ptr, ctypes.c_size_t]
_dll.starry_flux_design.restype = ctypes.c_int
_dll.starry_intensity_design.argtypes = [ctypes.c_uint32] + [ctypes.c_double] * 2 + [_ptr, ctypes.c_size_t]
_dll.starry_intensity_design.restype = ctypes.c_int
_dll.starry_rotate.argtypes = [ctypes.c_uint32, _ptr, ctypes.c_size_t] + [ctypes.c_double] * 4 + [_ptr]
_dll.starry_rotate.restype = ctypes.c_int
_dll.starry_flux_gradient.argtypes = [ctypes.c_uint32, _ptr, ctypes.c_size_t] + [ctypes.c_double] * 3 + [_ptr]
_dll.starry_flux_gradient.restype = ctypes.c_int
_dll.starry_limb_flux.argtypes = [ctypes.c_uint32, _ptr, ctypes.c_size_t, _ptr, ctypes.c_size_t] + [ctypes.c_double] * 3 + [_ptr]
_dll.starry_limb_flux.restype = ctypes.c_int


class Map:
    def __init__(self, ydeg=0):
        if isinstance(ydeg, bool) or not isinstance(ydeg, int) or not 0 <= ydeg <= 32:
            raise ValueError("ydeg must be an integer in [0, 32]")
        self.ydeg = ydeg
        self.y = [1.0] + [0.0] * ((ydeg + 1) ** 2 - 1)
        self.amp = 1.0

    def _row(self, func, *args):
        n = (self.ydeg + 1) ** 2
        out = (ctypes.c_double * n)()
        status = func(self.ydeg, *args, out, n)
        if status:
            raise ValueError("invalid arguments or numerical integration failed")
        return list(out)

    def _dot(self, row):
        if len(self.y) != len(row) or not all(math.isfinite(v) for v in [self.amp, *self.y]):
            raise ValueError("invalid map coefficients or amplitude")
        return self.amp * math.fsum(a * b for a, b in zip(row, self.y))

    def flux_design(self, xo=0.0, yo=0.0, ro=0.0):
        return self._row(_dll.starry_flux_design, xo, yo, ro)

    def flux(self, xo=0.0, yo=0.0, ro=0.0):
        return self._dot(self.flux_design(xo, yo, ro))

    def intensity(self, lat=0.0, lon=0.0):
        return self._dot(self._row(_dll.starry_intensity_design, lat, lon))

    def _coeffs(self):
        n = (self.ydeg + 1) ** 2
        if len(self.y) != n or not all(math.isfinite(v) for v in [self.amp, *self.y]):
            raise ValueError("invalid map coefficients or amplitude")
        return (ctypes.c_double * n)(*self.y)

    def rotate(self, axis, theta):
        if len(axis) != 3:
            raise ValueError("axis must have three components")
        coeffs = self._coeffs()
        out = (ctypes.c_double * len(coeffs))()
        status = _dll.starry_rotate(self.ydeg, coeffs, len(coeffs), *axis, theta, out)
        if status:
            raise ValueError("invalid rotation or numerical failure")
        self.y = list(out)

    def flux_gradient(self, xo=0.0, yo=0.0, ro=0.0):
        """Derivatives with respect to xo, yo, ro; includes amplitude."""
        coeffs = self._coeffs()
        out = (ctypes.c_double * 3)()
        status = _dll.starry_flux_gradient(self.ydeg, coeffs, len(coeffs), xo, yo, ro, out)
        if status:
            raise ValueError("invalid/undefined gradient or integration failed")
        return [self.amp * v for v in out]

    def flux_limb_darkened(self, u, xo=0.0, yo=0.0, ro=0.0):
        coeffs = self._coeffs()
        limb = (ctypes.c_double * len(u))(*u)
        out = ctypes.c_double()
        status = _dll.starry_limb_flux(self.ydeg, coeffs, len(coeffs), limb, len(u), xo, yo, ro, ctypes.byref(out))
        if status:
            raise ValueError("invalid limb-darkened map or numerical failure")
        return self.amp * out.value

# Retain the small ABI interface for callers that do not need array operations.
CoreMap = Map
from _starry_api import Map, Primary, Secondary, System, DopplerMap
from _orbit_api import kepler_state
