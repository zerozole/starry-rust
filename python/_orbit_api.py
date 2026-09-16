"""Native Keplerian state values, Jacobians and Hessians."""
import ctypes as ct
import numpy as np
from _starry_api import _dll,_ptr,array,ptr,check

_dll.starry_orbit_derivatives.argtypes=[_ptr,ct.c_size_t,_ptr]

def kepler_state(t,*,a,period,ecc=0.,inc=None,omega=None,node=0.,t0=0.,angle_unit='rad'):
    """Relative positions (solar radii), velocities (solar radii/day), derivatives.

    a and period are independent parameters. Jacobian/Hessian parameter order:
    a, period, ecc, inc, omega, node, t0, t. Angle derivatives use angle_unit.
    All inputs broadcast. inc and omega default to a quarter turn in angle_unit.
    Derivatives use implicit Kepler differentiation.
    """
    if angle_unit not in ('rad','deg'):raise ValueError('angle_unit must be rad or deg')
    scale=np.array([1.,1.,1.,*(3*[np.pi/180. if angle_unit=='deg' else 1.]),1.,1.])
    if inc is None:inc=np.pi/2/scale[3]
    if omega is None:omega=np.pi/2/scale[4]
    inputs=np.broadcast_arrays(*map(array,[a,period,ecc,inc,omega,node,t0,t]))
    shape=inputs[0].shape;parameters=array(np.stack([v.ravel() for v in inputs],axis=-1)*scale)
    output=np.empty((len(parameters),6,73))
    check(_dll.starry_orbit_derivatives(ptr(parameters),len(parameters),ptr(output)))
    values=output[:,:,0].reshape((*shape,6))
    jacobian=(output[:,:,1:9]*scale).reshape((*shape,6,8))
    hessian=(output[:,:,9:].reshape(-1,6,8,8)*scale[:,None]*scale[None,:]).reshape((*shape,6,8,8))
    return dict(position=values[...,:3],velocity=values[...,3:],jacobian=jacobian,hessian=hessian,
                parameters=('a','period','ecc','inc','omega','node','t0','t'))
