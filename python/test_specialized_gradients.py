import unittest
import numpy as np
from starry_rust import Map


class SpecializedGradients(unittest.TestCase):
    def test_shape_derivatives_against_flux_differences(self):
        cases = [dict(reflected=True, roughness=.3), dict(oblate=True, f=.2),
                 dict(oblate=True, f=0.)]
        for options in cases:
            m=Map(3,udeg=2,inc=1.1,obl=.4,**options)
            m[1],m[2]=.3,.1
            m[1,1]=.2
            for geometry in [(0.2,.3,.1),(.8,.3,.4),(0.,0.,2.)]:
                kw=dict(theta=.3,xs=.7,ys=.5,zs=1.2)
                g=m.flux_gradient(*geometry,**kw)
                for k in range(3):
                    a=list(geometry);b=list(geometry);a[k]+=2e-5;b[k]-=2e-5
                    d=(m.flux(*a,**kw)-m.flux(*b,**kw))/4e-5
                    np.testing.assert_allclose(g[k],d,atol=2e-7,rtol=2e-6)

    def test_finite_source_spectral_and_hidden(self):
        m=Map(1,nw=2,reflected=True,source_npts=4)
        m.amp=np.array([1.,2.])
        kw=dict(xs=.5,ys=.4,zs=2.,Rs=.1)
        g=m.flux_gradient(.4,.2,.2,**kw)
        self.assertEqual(g.shape,(2,3))
        np.testing.assert_allclose(g[1],2*g[0],atol=1e-12)
        np.testing.assert_array_equal(m.flux_gradient(.4,.2,.2,zo=-1.,**kw),0.)


if __name__=='__main__':unittest.main()
