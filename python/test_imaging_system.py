import math
import unittest
import numpy as np
from starry_rust import Map, Primary, Secondary, System


class ImagingSystemTests(unittest.TestCase):
    def test_rendered_flux_matches_forward_models(self):
        maps = [Map(udeg=2), Map(reflected=True), Map(oblate=True, f=.2)]
        maps[0][1], maps[0][2] = .3, .1
        for m in maps:
            image = m.render(res=240)
            self.assertAlmostEqual(np.nansum(image)*4/image.size, m.flux(), delta=6e-4)
        reflected = maps[1]
        reflected.source_npts = 25
        image = reflected.render(160, zs=8., Rs=1.)
        self.assertAlmostEqual(np.nansum(image)*4/image.size,
                               reflected.flux(zs=8., Rs=1.), delta=2e-6)

    def test_local_filters_and_intrinsic_rectangular_projection(self):
        limb = Map(udeg=1)
        limb[1] = .3
        self.assertAlmostEqual(limb.intensity(), 1/(math.pi*.9), places=13)
        self.assertAlmostEqual(limb.intensity(limbdarken=False), 1/math.pi, places=13)
        rv = Map(rv=True, veq=10000.)
        self.assertAlmostEqual(rv.intensity(lon=.4), 10000*math.sin(.4)/math.pi, places=9)
        self.assertAlmostEqual(rv.intensity(lon=.4, rv=False), 1/math.pi, places=13)
        m = Map(1, inc=.3, obl=.2)
        m[1, 1] = .1
        np.testing.assert_allclose(m.render(10, projection='rect', theta=.2),
                                   m.render(10, projection='rect', theta=.7))

    def test_exact_scattering_and_oblate_intensity(self):
        m = Map(reflected=True, roughness=.3)
        sigma2 = .09
        expected = (1-.5*sigma2/(sigma2+.33))/math.pi
        self.assertAlmostEqual(m.intensity(on94_exact=True), expected, places=13)
        np.testing.assert_allclose(m.render(12, illuminate=False)[3:9, 3:9], 1.)
        oblate = Map(oblate=True, f=.2, fdeg=2, omega=.3)
        self.assertTrue(np.isfinite(oblate.intensity(lat=.3)))
        self.assertTrue(np.isfinite(oblate.render(16)[8, 8]))

    def test_oblate_system_transit_and_spectral_channels(self):
        star = Map(1, oblate=True, f=.2, obl=.3, nw=2, amp=[1., 2.])
        star[1, 1] = .1
        system = System(Primary(star), Secondary(Map(amp=0.), a=8., r=.1, m=0.))
        times = np.array([-.01, 0., .01])
        x, y, z = system.position(times)
        expected = star.flux(xo=x[1]-x[0], yo=y[1]-y[0], zo=z[1]-z[0], ro=.1,
                             theta=2*np.pi*times)
        np.testing.assert_allclose(system.flux(times), expected, atol=1e-10)

    def test_exposure_rv_is_ratio_of_integrated_fluxes(self):
        star = Map(rv=True, veq=10000.)
        primary = Primary(star)
        secondary = Secondary(Map(amp=0.), a=8., r=.2, m=0.)
        instant = System(primary, secondary)
        integrated = System(primary, secondary, texp=.01, oversample=15)
        time = .01
        sub = time+.01*((np.arange(15)+.5)/15-.5)
        flux = instant.flux(sub, total=False)[0]
        velocity = instant.rv(sub, keplerian=False, total=False)[0]
        expected = np.sum(flux*velocity)/np.sum(flux)
        self.assertAlmostEqual(integrated.rv([time], keplerian=False)[0], expected, places=9)


if __name__ == '__main__':
    unittest.main()
