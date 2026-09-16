import math
import unittest
import numpy as np
from starry_rust import Map, Primary, Secondary, System, DopplerMap


class ExtendedInterfaceTests(unittest.TestCase):
    def test_reflected_broadcast_and_spectral(self):
        m=Map(reflected=True,nw=2,amp=[1.,2.])
        np.testing.assert_allclose(m.flux(zs=[1.,2.]),[[2/3,4/3],[1/6,1/3]],rtol=1e-13)
        np.testing.assert_allclose(m.flux(zs=[1.,2.],integrated=True),[2.,.5])
        np.testing.assert_allclose(m.intensity(illuminate=False),[1.,2.])
        self.assertEqual(Map().flux(),1.)
        self.assertEqual(np.shape(Map().flux(theta=np.zeros((2,3)))),(2,3))

    def test_persistent_updates_and_slice_indices(self):
        m=Map(2);m[1:,:]=np.arange(8)/30
        np.testing.assert_allclose(m.y[1:],np.arange(8)/30)
        first=m.flux(theta=.3);m[1,0]+=.1
        self.assertNotEqual(first,m.flux(theta=.3))
        m.amp=3.;self.assertAlmostEqual(m.flux(),3*Map(2,amp=1.).flux_design()@m.y)
        with self.assertRaises(ValueError):m[:]=1

    def test_degree_angles_match_radians(self):
        a=Map(2,angle_unit='deg',inc=57,obl=19);b=Map(2,inc=np.deg2rad(57),obl=np.deg2rad(19))
        a[1,1]=b[1,1]=.2
        np.testing.assert_allclose(a.flux(theta=[0,30,90]),b.flux(theta=np.deg2rad([0,30,90])),atol=1e-14)

    def test_oriented_limb_geometry_gradient(self):
        m=Map(2,udeg=2,inc=.7,obl=.4);m[1,1]=.15;m[1]=.3;m[2]=.1
        geometry=np.array([.2,.5,.13]);gradient=m.flux_gradient(*geometry,theta=.6);difference=[]
        for i in range(3):
            a=geometry.copy();b=geometry.copy();a[i]+=1e-5;b[i]-=1e-5
            difference.append((m.flux(*a,theta=.6)-m.flux(*b,theta=.6))/2e-5)
        np.testing.assert_allclose(gradient,difference,atol=2e-9)

    def test_gaussian_inference_matches_numpy(self):
        m=Map(1);rng=np.random.default_rng(452)
        x=rng.normal(size=(12,4));truth=np.array([1.,.1,.2,-.1]);data=x@truth
        noise=.03;prior=.5
        m.set_data(data,C=noise);m.set_prior(L=prior)
        mean,chol=m.solve(design_matrix=x)
        covariance=np.linalg.inv(x.T@x/noise+np.eye(4)/prior)
        np.testing.assert_allclose(mean,covariance@x.T@data/noise,atol=1e-13)
        np.testing.assert_allclose(chol@chol.T,covariance,atol=1e-13)
        c=noise*np.eye(12)+prior*x@x.T
        expected=-.5*(data@np.linalg.solve(c,data)+np.linalg.slogdet(c)[1]+12*np.log(2*np.pi))
        self.assertAlmostEqual(m.lnlike(design_matrix=x),expected,places=10)
        np.testing.assert_allclose(m.amp*m.y,mean)
        self.assertEqual(m.draw(rng).shape,(4,))

    def test_surface_roundtrip(self):
        m=Map(2);m[1,-1]=.1;m[2,0]=-.07
        image=m.render(16,projection='rect');n=Map(2);n.load(image,ridge=0.)
        np.testing.assert_allclose(n.amp*n.y,m.amp*m.y,atol=2e-14)
        self.assertEqual(m.render(10,theta=[0,.5]).shape,(2,10,10))
        self.assertTrue(np.isnan(m.render(10)[0,0]))
        n.spot(.2,.3,.1,.2);self.assertLess(n.amp,m.amp)

    def test_doppler_convolution(self):
        m=Map();spectrum=np.ones(101);spectrum[50]=.2
        d=DopplerMap(m,spectrum,log_spacing=1e-5,half_width=10,veq=30000.)
        k=d.kernel(normalize=True);result=d.flux()
        np.testing.assert_allclose(result,np.correlate(spectrum,k,'valid'),atol=1e-14)
        self.assertAlmostEqual(k.sum(),1.,places=14)
        self.assertGreater(result.min(),.2)
        flat=DopplerMap([m,Map(amp=2.)],np.ones((2,101)),log_spacing=1e-5,half_width=10,veq=30000.)
        np.testing.assert_allclose(flat.flux(theta=[0,.5]),1.,atol=1e-14)

    def test_system_transit_spectral_and_positions(self):
        star=Primary(Map(nw=2,amp=[1.,2.]),m=1.)
        planet=Secondary(Map(amp=0.),r=.1,m=.001,porb=5.)
        s=System(star,planet)
        np.testing.assert_allclose(s.flux([0.]),[[.99,1.98]],atol=1e-13)
        p=s.position([0.,1.]);self.assertEqual(p[0].shape,(2,2))
        self.assertGreater(p[2][1,0],p[2][0,0])
        for axis in p:np.testing.assert_allclose(axis[0]+.001*axis[1],0.,atol=1e-14)
        total=s.flux([0.,1.]);parts=s.flux([0.,1.],total=False)
        np.testing.assert_allclose(total,parts.sum(axis=0))

    def test_specialized_maps_and_errors(self):
        self.assertAlmostEqual(Map(oblate=True,f=.2,normalized=False).flux(),.8,places=12)
        self.assertAlmostEqual(Map(rv=True,veq=10000.).rv(),0.,places=10)
        with self.assertRaises(ValueError):Map(reflected=True).flux(zs=0.)
        with self.assertRaises(ValueError):System(Primary(Map()),Secondary(Map(),porb=-1.)).flux([0.])
        with self.assertRaises(ValueError):Map().set_data([1.],C=np.eye(2))
        with self.assertRaises(ValueError):Map().set_data([1.,2.],cho_C=np.eye(1))
        with self.assertRaises(ValueError):Map(1).set_prior(cho_L=np.eye(1))

    def test_finite_source_against_lambert_integrals(self):
        requested=25;radius=1.;distance=8.
        m=Map(reflected=True,source_npts=requested)
        n=int(2+np.sqrt(requested*4/np.pi));x,y=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n));valid=x*x+y*y<1
        x=x[valid];y=y[valid];z=distance-np.sqrt(1-x*x-y*y);r=np.sqrt(x*x+y*y+z*z);alpha=np.arccos(z/r)
        expected=np.mean(2/(3*np.pi*r*r)*(np.sin(alpha)+(np.pi-alpha)*np.cos(alpha)))
        self.assertAlmostEqual(m.flux(zs=distance,Rs=radius),expected,places=13)
        self.assertAlmostEqual(m.flux(zs=distance,Rs=0.),2/(3*distance**2),places=14)

    def test_system_rotational_rv(self):
        m=Map(rv=True,veq=10000.)
        planet=Secondary(Map(amp=0.),r=.15,m=0.,a=8.)
        s=System(Primary(m),planet)
        times=np.array([-.01,.0,.01]);x,y,z=s.position(times)
        expected=m.rv(xo=x[1]-x[0],yo=y[1]-y[0],zo=z[1]-z[0],ro=.15)
        np.testing.assert_allclose(s.rv(times,keplerian=False),expected,atol=1e-8)
        self.assertLess(expected[0]*expected[2],0.)

    def test_reflected_system_scales_with_source_luminosity(self):
        source=Map(amp=2.)
        planet=Secondary(Map(reflected=True),a=8.,r=.1,m=0.)
        system=System(Primary(source),planet)
        first=system.flux([1.],total=False)[1,0]
        source.amp=3.
        self.assertGreater(first,0.)
        self.assertAlmostEqual(system.flux([1.],total=False)[1,0],1.5*first,places=14)


if __name__=='__main__':unittest.main()
