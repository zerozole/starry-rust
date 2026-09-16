import unittest
import numpy as np
from starry_rust import Map, DopplerMap, Primary, Secondary, System
from _inverse_api import infer


class InverseTests(unittest.TestCase):
    def test_emitted_jacobian_and_surface_minimum(self):
        m=Map(2,udeg=1,angle_unit='deg',inc=55,obl=25)
        m[1,1]=.2;m[1]=.3
        result=m.flux_jacobian(xo=.2,yo=.4,ro=.1,theta=30)
        h=1e-3
        numerical=(m.flux(xo=.2,yo=.4,ro=.1,theta=30+h)-m.flux(xo=.2,yo=.4,ro=.1,theta=30-h))/(2*h)
        self.assertAlmostEqual(float(result['theta']),numerical,places=10)
        self.assertAlmostEqual(float(result['flux']),m.flux(xo=.2,yo=.4,ro=.1,theta=30),places=13)
        dipole=Map(1);dipole[1,1]=.2
        lat,lon,value=dipole.minimize()
        self.assertAlmostEqual(value,(1-.2*np.sqrt(3))/np.pi,places=10)
        self.assertAlmostEqual(lat,0.,places=6)
        self.assertAlmostEqual(lon,-np.pi/2,places=6)

    def test_limb_physical(self):
        rng=np.random.default_rng(25);m=Map(udeg=2)
        for u1,u2 in rng.normal(size=(100,2)):
            m[1],m[2]=u1,u2
            self.assertEqual(m.limbdark_is_physical(),bool(u1+u2<1 and u1>0 and u1+2*u2>0))

    def doppler(self):
        m = Map(1, inc=1.1)
        m[1, 1], m[1, -1] = .12, -.08
        spectrum = 1-.6*np.exp(-((np.arange(25)-12)/1.7)**2)
        return DopplerMap(m, spectrum, log_spacing=2e-5, half_width=4,
                          veq=21000., u=[.3])

    def test_diagonal_inference_matches_full_covariance(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(11, 4))
        y = rng.normal(size=11)
        v = np.linspace(.1, .5, 11)
        a = infer(x, y, v, np.ones(4), .7)
        b = infer(x, y, np.diag(v), np.ones(4), .7)
        for first, second in zip(a, b):
            np.testing.assert_allclose(first, second, atol=1e-12)

    def test_doppler_design_fit_and_jacobian(self):
        d = self.doppler()
        phases = [0., .8, 1.7, 2.6]
        flux = d.flux(phases, normalize=False)
        weights = d._weights().copy()
        x = d.design_matrix(phases, fix_spectrum=True)
        s = d.design_matrix(phases, fix_map=True)
        np.testing.assert_allclose(x@weights, flux.ravel(), atol=1e-13)
        np.testing.assert_allclose(s@d.spectra.ravel(), flux.ravel(), atol=1e-13)
        f, jacobian = d.map_jacobian(phases)
        np.testing.assert_allclose(f, d.flux(phases).ravel(), atol=1e-13)
        np.testing.assert_allclose(jacobian@weights, 0., atol=1e-13)
        d.maps[0].y[1:] = 0.
        solution, chol = d.solve(flux, theta=phases, flux_err=1e-5, prior_cov=1e6)
        np.testing.assert_allclose(solution, weights, atol=1e-7)
        self.assertEqual(chol.shape, (4, 4))
        d.spectra[:] = 1.
        d.solve(flux, theta=phases, solver='spectrum', flux_err=1e-5)
        np.testing.assert_allclose(d.flux(phases, normalize=False), flux, atol=1e-7)

    def test_normalized_map_fit(self):
        d = self.doppler()
        phases = [0., .8, 1.7, 2.6]
        flux = d.flux(phases)
        d.maps[0].y[1:] = 0.
        d.solve(flux, theta=phases, normalize=True, flux_err=1e-5, prior_cov=1e4)
        np.testing.assert_allclose(d.flux(phases), flux, atol=1e-7)

    def test_wavelength_resampling(self):
        rest = np.linspace(499.8, 500.3, 101)
        observed = np.linspace(500., 500.1, 12)
        d = DopplerMap.from_wavelengths(Map(), observed, rest, np.ones(101),
                                        veq=20000., oversample=2)
        np.testing.assert_allclose(d.flux([0., .5]), 1., atol=1e-14)
        self.assertEqual(d.flux().shape, (12,))
        x = d.design_matrix([0., .5], fix_map=True)
        np.testing.assert_allclose(x@d.spectra.ravel(), d.flux([0., .5], normalize=False).ravel(), atol=1e-13)

    def test_system_design_and_map_inference(self):
        star = Map(1, inc=1.1)
        star[1, 1], star[1, -1] = .12, -.07
        planet = Map(amp=.01)
        system = System(Primary(star), Secondary(planet, porb=5., r=.1, m=.001))
        times = np.linspace(-.3, .3, 24)
        flux = system.flux(times)
        design = system.design_matrix(times)
        weights = np.r_[star.amp*star.y, planet.amp*planet.y]
        np.testing.assert_allclose(design@weights, flux, atol=1e-13)
        star.set_prior(L=100.)
        system.set_data(flux, C=1e-10)
        star.y[1:] = 0.
        mean, chol = system.solve(t=times)
        np.testing.assert_allclose(mean, weights[:4], atol=1e-5)
        np.testing.assert_allclose(system.flux(times), flux, atol=1e-8)
        self.assertEqual(chol.shape, (4, 4))
        self.assertTrue(np.isfinite(system.lnlike(t=times)))
        self.assertEqual(system.draw(np.random.default_rng(23)).shape, (4,))
        self.assertEqual(planet.amp, .01)


if __name__ == '__main__':
    unittest.main()
