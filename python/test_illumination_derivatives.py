import unittest
import numpy as np
from starry_rust import Map


class IlluminationDerivatives(unittest.TestCase):
    def test_point_source_and_inverse_square_identity(self):
        for rough in [0.,.3]:
            m=Map(2,reflected=True,roughness=rough,udeg=1,inc=1.1,obl=.3)
            m[1]=.2;m[1,1]=.1
            kw=dict(xs=.6,ys=.8,zs=1.2,theta=.2,xo=.4,yo=.2,ro=.3)
            jac=m.illumination_jacobian(**kw)
            for name in ('xs','ys','zs'):
                a=kw.copy();b=kw.copy();a[name]+=1e-5;b[name]-=1e-5
                expected=(m.flux(**a)-m.flux(**b))/2e-5
                np.testing.assert_allclose(jac[name],expected,atol=2e-8,rtol=1e-6)
            self.assertAlmostEqual(sum(kw[k]*jac[k] for k in ('xs','ys','zs')),-2*jac['flux'],places=10)
            if rough:
                m.roughness=rough+1e-5;a=m.flux(**kw)
                m.roughness=rough-1e-5;b=m.flux(**kw)
                np.testing.assert_allclose(jac['roughness'],(a-b)/2e-5,atol=2e-8)

    def test_finite_source_radius_and_angle_units(self):
        m=Map(1,reflected=True,roughness=15.,angle_unit='deg',source_npts=4)
        m[1,1]=.1
        kw=dict(xs=.6,ys=.8,zs=2.,Rs=.2,xo=.4,yo=.2,ro=.2)
        jac=m.illumination_jacobian(**kw)
        for name in ('xs','ys','zs','Rs'):
            a=kw.copy();b=kw.copy();a[name]+=1e-5;b[name]-=1e-5
            expected=(m.flux(**a)-m.flux(**b))/2e-5
            np.testing.assert_allclose(jac[name],expected,atol=3e-8,rtol=2e-6)
        m.roughness+=1e-4;a=m.flux(**kw);m.roughness-=2e-4;b=m.flux(**kw)
        np.testing.assert_allclose(jac['roughness'],(a-b)/2e-4,atol=1e-9)

    def test_lambert_phase_poles(self):
        m=Map(0,reflected=True)
        jac=m.illumination_jacobian(zs=2.)
        self.assertAlmostEqual(jac['flux'],1/6.,places=12)
        self.assertAlmostEqual(jac['zs'],-1/6.,places=12)
        self.assertAlmostEqual(jac['xs'],0.,places=12)
        self.assertAlmostEqual(m.illumination_jacobian(zs=-2.)['zs'],0.,places=12)

    def test_zero_finite_source_radius_limit(self):
        m=Map(1,reflected=True,roughness=.2,source_npts=4);m.y[1]=.03
        kw=dict(xs=.6,ys=.8,zs=2.)
        derivative=m.illumination_jacobian(**kw,Rs=0.)['Rs']
        f=m.flux(**kw,Rs=0.)
        h=1e-5
        expected=(-3*f+4*m.flux(**kw,Rs=h)-m.flux(**kw,Rs=2*h))/(2*h)
        np.testing.assert_allclose(derivative,expected,atol=2e-9)


if __name__=='__main__':unittest.main()
