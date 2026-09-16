import unittest
import numpy as np
from starry_rust import Map,DopplerMap

class SelectedContinuum(unittest.TestCase):
    def test_quotients_and_all_derivatives(self):
        m=Map(1,inc=1.1);m.y[1:]=[.1,.03,-.04]
        rest=1-.4*np.exp(-((np.arange(17)-8)/2)**2)
        d=DopplerMap(m,rest,log_spacing=2e-5,half_width=3,veq=15000.,u=[.2],continuum_index=4)
        phases=np.array([.2,.8]);raw=d.flux(phases,False);flux=d.flux(phases)
        np.testing.assert_allclose(flux,raw/raw[:,4,None],atol=1e-14)
        jac=d.flux_jacobian(phases)
        np.testing.assert_allclose(jac['flux'],flux,atol=1e-13)
        np.testing.assert_allclose(jac['theta'],(d.flux(phases+1e-5)-d.flux(phases-1e-5))/2e-5,atol=1e-9)
        f,j=d.spectrum_jacobian(phases);np.testing.assert_allclose(f,flux.ravel(),atol=1e-13)
        for i in [0,7,8,16]:
            old=d.spectra[0,i];d.spectra[0,i]=old+1e-5;a=d.flux(phases);d.spectra[0,i]=old-1e-5;b=d.flux(phases);d.spectra[0,i]=old
            np.testing.assert_allclose(j[:,i],((a-b)/2e-5).ravel(),atol=1e-9)
        f,j=d.map_jacobian(phases)
        for i in range(4):
            old=m.y[i];m.y[i]=old+1e-5;a=d.flux(phases);m.y[i]=old-1e-5;b=d.flux(phases);m.y[i]=old
            np.testing.assert_allclose(j[:,i],((a-b)/2e-5).ravel(),atol=1e-9)
        d.spectra*=.98
        result=d.solve(flux,theta=phases,solver='nonlinear',normalize=True,flux_err=.01,iterations=5)
        self.assertTrue(np.all(np.diff(result['objective'])<=1e-8))

    def test_resampled_normalization(self):
        wav=np.linspace(500.,500.2,20);restwave=np.linspace(499.8,500.4,31)
        d=DopplerMap.from_wavelengths(Map(1,inc=1.1),wav,restwave,1-.3*np.exp(-((restwave-500.1)/.04)**2),veq=20000.,continuum_index=6)
        raw=d.flux(.3,False);np.testing.assert_allclose(d.flux(.3),raw/raw[6],atol=1e-14)
        np.testing.assert_allclose(d.flux_jacobian(.3)['flux'],d.flux(.3),atol=1e-13)
        d.continuum_index=99
        with self.assertRaises(ValueError):d.flux(.3)

if __name__=='__main__':unittest.main()
