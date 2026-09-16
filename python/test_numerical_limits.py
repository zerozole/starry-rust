import unittest
import numpy as np
from starry_rust import Map,DopplerMap,Primary,Secondary,System

class NumericalLimits(unittest.TestCase):
    def test_reflected_singular_domains(self):
        m=Map(reflected=True,roughness=.2,source_npts=4)
        with self.assertRaisesRegex(ValueError,'phase pole'):m.illumination_jacobian(zs=2.)
        with self.assertRaisesRegex(ValueError,'adjacent valid'):m.illumination_jacobian(xs=.2,ys=.2,zs=.2,Rs=0.)
        m.source_npts=1
        self.assertTrue(np.isfinite(m.illumination_jacobian(xs=.2,ys=.2,zs=.2)['xs']))

    def test_doppler_kinks_and_bounds(self):
        d=DopplerMap(Map(inc=np.pi/2),np.ones(9),log_spacing=2e-5,half_width=1,veq=1.)
        with self.assertRaisesRegex(ValueError,'floor kink'):d.flux_jacobian()
        d.veq=10000.;d.log_spacing=np.arctanh(d.veq/299792458.)
        with self.assertRaisesRegex(ValueError,'support edge'):d.flux_jacobian()
        d.log_spacing*=1.01
        self.assertTrue(np.all(np.isfinite(d.flux_jacobian()['veq'])))
        with self.assertRaises(ValueError):Map(33)

    def test_contact_and_disabled_branches(self):
        m=Map(1,rv=True,veq=10000.)
        with self.assertRaisesRegex(ValueError,'coincident'):m.flux_gradient(ro=1.)
        np.testing.assert_array_equal(m.flux_gradient(ro=.2,xo=1.2),[0.,0.,0.])
        self.assertEqual(m.rv_jacobian(ro=2.)['rv'],0.)
        p=Primary(Map(1),prot=0.)
        a=Secondary(Map(amp=0.),porb=2.,r=0.,m=0.)
        s=System(p,a);j=s.flux_jacobian([.2])
        np.testing.assert_array_equal(j['body0.prot'],0.)
        np.testing.assert_array_equal(j['body1.r'],0.)

if __name__=='__main__':unittest.main()
