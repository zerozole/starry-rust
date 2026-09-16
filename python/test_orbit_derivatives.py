import unittest
import numpy as np
from starry_rust import kepler_state

class KeplerDerivatives(unittest.TestCase):
    def test_broadcast_values_and_angle_derivatives(self):
        t=np.array([0.,.2,.8]);a=5.;period=3.2
        base=kepler_state(t,a=a,period=period)
        np.testing.assert_allclose(base['position'][:,0],a*np.sin(2*np.pi*t/period),atol=1e-13)
        np.testing.assert_allclose(base['position'][:,2],a*np.cos(2*np.pi*t/period),atol=1e-13)
        np.testing.assert_allclose(base['jacobian'][:,:3,7],base['velocity'],atol=1e-13)
        np.testing.assert_allclose(base['hessian'][:,:3,7,:],base['jacobian'][:,3:,:],atol=1e-12)
        rad=kepler_state(t,a=a,period=period,ecc=.3,inc=1.1,omega=.7,node=.3)
        deg=kepler_state(t,a=a,period=period,ecc=.3,inc=np.rad2deg(1.1),omega=np.rad2deg(.7),node=np.rad2deg(.3),angle_unit='deg')
        np.testing.assert_allclose(rad['position'],deg['position'],atol=1e-13)
        scale=np.array([1,1,1,np.pi/180,np.pi/180,np.pi/180,1,1])
        np.testing.assert_allclose(deg['jacobian'],rad['jacobian']*scale,atol=1e-12)
        np.testing.assert_allclose(deg['hessian'],rad['hessian']*scale[:,None]*scale[None,:],atol=1e-11)
        with self.assertRaises(ValueError):kepler_state(0.,a=5.,period=3.,ecc=1.)

if __name__=='__main__':unittest.main()
