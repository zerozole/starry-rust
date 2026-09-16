import unittest
import numpy as np
from starry_rust import Map,Primary,Secondary,System

class ExposureRules(unittest.TestCase):
    def test_render_coordinates_and_smoothing(self):
        m=Map(3,inc=1.1,obl=.5);m.y[1:]=np.sin(np.arange(15))*.03
        for projection in ['ortho','rect','moll']:
            lat,lon=m.get_latlon_grid(res=25,projection=projection,theta=.8)
            image=m.render(res=25,projection=projection,theta=.8)
            valid=np.isfinite(lat)
            np.testing.assert_array_equal(np.isfinite(image),valid)
            np.testing.assert_allclose(image[valid],m.intensity(lat[valid],lon[valid]),atol=1e-12)
        inverse,grid=m.sht_matrix(smoothing=0.,lam=0.,return_grid=True)
        forward=m.sht_matrix(inverse=True,lam=0.)
        np.testing.assert_allclose(inverse@forward,np.eye(m.Ny),atol=1e-12)
        smooth=m.sht_matrix(smoothing=.4,lam=0.)
        degrees=np.floor(np.sqrt(np.arange(m.Ny)))
        np.testing.assert_allclose(smooth@forward,np.diag(np.exp(-.5*degrees*(degrees+1)*.4**2)),atol=1e-12)
        self.assertEqual(grid.shape[1],2)

    def test_vector_exposures_and_rv(self):
        m=Map(1,rv=True,veq=15000.,inc=1.1);m.y[1:]=[.1,-.04,.07]
        primary=Primary(m,prot=.7)
        secondary=Secondary(Map(amp=0.),porb=2.,r=.15)
        times=np.array([-.02,0.,.035]);exposures=np.array([0.,.025,.04])
        instant=System(primary,secondary)
        for order in range(3):
            n=5;offset=(np.arange(n)+.5)/n-.5 if order==0 else np.linspace(-.5,.5,n)
            w=np.ones(n) if order==0 else np.array([1,2,2,2,1]) if order==1 else np.array([1,4,2,4,1])
            w=w/w.sum()
            s=System(primary,secondary,texp=exposures,oversample=n,order=order)
            flux=[];rv=[]
            for t,e in zip(times,exposures):
                sample=t+e*offset;f=instant.flux(sample,total=False)[0]
                v=instant.rv(sample,keplerian=False,total=False)[0]
                flux.append(w@f);rv.append((w@(f*v))/(w@f))
            np.testing.assert_allclose(s.flux(times),flux,atol=1e-13)
            np.testing.assert_allclose(s.rv(times,keplerian=False),rv,atol=1e-8)
        with self.assertRaises(ValueError):System(primary,secondary,texp=[0.,-1.,0.]).flux(times)
        with self.assertRaises(ValueError):System(primary,order=3)

if __name__=='__main__':unittest.main()
