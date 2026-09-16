import unittest
import numpy as np
from starry_rust import Map


class OblateDerivatives(unittest.TestCase):
    def test_physical_derivatives_and_normalization(self):
        for normalized in [False,True]:
            for degree in [0,4]:
                m=Map(2,oblate=True,f=.2,fdeg=degree,omega=.4,beta=.23,tpole=5800.,wav=600e-9,
                      inc=1.1,obl=.3,normalized=normalized,udeg=1,amp=1.2)
                m[1]=.2;m[1,1]=.1;m[2,-1]=-.06
                kw=dict(xo=.5,yo=.25,ro=.3,theta=.4)
                jac=m.oblate_jacobian(**kw)
                np.testing.assert_allclose(jac['flux'],m.flux(**kw),rtol=2e-11,atol=2e-10)
                for key in ('theta','inc','obl','f','omega','beta','tpole','wav','amp'):
                    old=kw[key] if key=='theta' else getattr(m,key)
                    h=abs(old)*1e-5 if key in ('tpole','wav') else 1e-5
                    if key=='theta':
                        kw[key]=old+h;a=m.flux(**kw);kw[key]=old-h;b=m.flux(**kw);kw[key]=old
                    else:
                        setattr(m,key,old+h);a=m.flux(**kw);setattr(m,key,old-h);b=m.flux(**kw);setattr(m,key,old)
                    expected=(a-b)/(2*h)
                    np.testing.assert_allclose(jac[key],expected,rtol=3e-5,atol=1e-7*max(1.,abs(jac['flux'])),err_msg=key)

    def test_projected_area_derivative_and_angle_units(self):
        m=Map(0,oblate=True,f=.3,inc=65.,obl=10.,angle_unit='deg',normalized=False)
        jac=m.oblate_jacobian()
        inc=np.deg2rad(65.);q=np.sqrt(1-.3*1.7*np.sin(inc)**2)
        self.assertAlmostEqual(jac['f'],-.7*np.sin(inc)**2/q,places=11)
        self.assertAlmostEqual(jac['inc'],-.3*1.7*np.sin(inc)*np.cos(inc)/q*np.pi/180,places=11)
        m.amp=0.;jac=m.oblate_jacobian()
        self.assertAlmostEqual(jac['amp'],q,places=11)
        self.assertEqual(jac['f'],0.)


if __name__=='__main__':unittest.main()
