import unittest
import numpy as np
from starry_rust import Map,DopplerMap,Primary,Secondary,System

class NativeTransforms(unittest.TestCase):
    def test_spectral_map_inference_correlated_prior(self):
        m=Map(1,nw=2,amp=[1.,2.],inc=1.1,obl=.4)
        m.y[1:]=[[.1,.03],[-.04,.06],[.08,-.1]]
        lat,lon=m.pixel_grid();x=m.intensity_design_matrix(lat,lon)
        weights=(m.y*m.amp).ravel().copy();flux=m.intensity(lat,lon)
        np.testing.assert_allclose(x@weights,flux.ravel(),atol=1e-12)
        prior=np.eye(8)*10.;prior[0,1]=prior[1,0]=2.
        m.set_prior(L=prior);m.set_data(flux,C=np.full(flux.shape,1e-10))
        m.amp=1.;m.y[1:]=0.
        mean,_=m.solve(design_matrix=x)
        np.testing.assert_allclose(mean,weights,atol=1e-9)
        m.draw(np.random.default_rng(49));self.assertEqual(m.y.shape,(4,2))
        self.assertTrue(np.isfinite(m.lnlike(design_matrix=x)))

    def test_spectral_system_joint_inference(self):
        m=Map(1,nw=2,amp=[1.,2.],inc=1.1,obl=.4);m.y[1:]=[[.1,.03],[-.04,.06],[.08,-.1]]
        system=System(Primary(m),Secondary(Map(amp=.01),porb=2.,r=.15))
        times=np.r_[np.linspace(-.4,.6,25),np.linspace(-.04,.04,11)];flux=system.flux(times)
        weights=(m.y*m.amp).ravel().copy()
        design=system.design_matrix(times)
        np.testing.assert_allclose(design@np.r_[weights,.01],flux.ravel(),atol=1e-12)
        m.set_prior(L=1e3);m.y[1:]=0.;m.amp=1.
        noise=np.full(flux.shape,1e-12);system.set_data(flux,C=noise)
        mean,chol=system.solve(t=times)
        np.testing.assert_allclose(mean,weights,atol=2e-7)
        np.testing.assert_allclose(system.flux(times),flux,atol=1e-10)
        self.assertEqual(chol.shape,(8,8))
        self.assertTrue(np.isfinite(system.lnlike(t=times)))
        system.draw(np.random.default_rng(15));self.assertEqual(m.y.shape,(4,2))

    def test_pixels(self):
        m=Map(3,angle_unit='deg');m.y[1:]=np.arange(15)*.007
        t=m.surface_transforms(ridge=0.)
        values=t['forward']@m.y
        np.testing.assert_allclose(values,m.intensity(t['lat'],t['lon']),atol=1e-12)
        np.testing.assert_allclose(t['inverse']@values,m.y,atol=1e-12)
        h=1e-4
        finite=(m.intensity(t['lat'],t['lon']+h)-m.intensity(t['lat'],t['lon']-h))/(2*h)
        np.testing.assert_allclose(t['dlon']@m.y,finite,atol=1e-10)

    def test_spectral_scalar_amplitude_transforms(self):
        m=Map(2,nw=2);m.amp=3.;m.rotate([1,0,0],.2)
        np.testing.assert_allclose(m.amp,[3.,3.])
        lat,lon=m.pixel_grid();m.load_samples(lat,lon,np.full((len(lat),2),2./np.pi),ridge=0.)
        np.testing.assert_allclose(m.amp,[2.,2.],atol=1e-12)

    def test_spectral_operator_resampled_adjoint(self):
        m=Map(1);m.y[1:]=[.1,-.03,.07]
        wav=np.linspace(500.,500.2,17);rest=np.linspace(499.8,500.4,61)
        spectrum=1.-.4*np.exp(-((rest-500.1)/.018)**2)
        d=DopplerMap.from_wavelengths([m],wav,rest,[spectrum],veq=12000.)
        phase=[0.,.8,1.5]
        weights=m.y[:,None]*m.amp*d.spectra[0]
        np.testing.assert_allclose(d.spectral_map_dot(weights,phase),d.flux(phase,normalize=False),atol=1e-12)
        rng=np.random.default_rng(48);weights=rng.normal(size=weights.shape)
        r=rng.normal(size=(3,len(wav)))
        lhs=np.sum(d.spectral_map_dot(weights,phase)*r)
        rhs=np.sum(weights*d.spectral_map_dot(r,phase,transpose=True))
        self.assertAlmostEqual(lhs,rhs,places=12)

if __name__=='__main__':unittest.main()
