import unittest
import numpy as np
from starry_rust import Map,Primary,Secondary,System


class SystemDerivatives(unittest.TestCase):
    def check_system(self,system,times,rv=None):
        evaluate=(lambda t:system.flux(t,total=False)) if rv is None else (lambda t:system.rv(t,keplerian=rv,total=False))
        jac=system.flux_jacobian(times,total=False) if rv is None else system.rv_jacobian(times,keplerian=rv,total=False)
        name='flux' if rv is None else 'rv'
        np.testing.assert_allclose(jac[name],evaluate(times),atol=2e-7 if rv is not None else 1e-12)
        for key,value in jac.items():
            if key==name:continue
            h=2e-6
            if key=='time':
                a=evaluate(times+h);b=evaluate(times-h)
            else:
                if key=='texp':obj=system;attr='texp'
                else:
                    body,attr=key.split('.');obj=system.bodies[int(body[4:])]
                    if attr in ('veq','alpha'):obj=obj.map
                old=getattr(obj,attr)
                if attr=='texp' and old==0:continue
                setattr(obj,attr,old+h);a=evaluate(times)
                setattr(obj,attr,old-h);b=evaluate(times)
                setattr(obj,attr,old)
            np.testing.assert_allclose(value,(a-b)/(2*h),rtol=3e-4,atol=3e-7 if rv is None else 2e-3,err_msg=key)

    def test_orbits_spins_delay_exposure_and_overlap(self):
        m=Map(2,inc=1.2,obl=.2,udeg=1);m.y[1:]=np.arange(8)*.01;m.u[1]=.3
        p=Primary(m,prot=.8,theta0=.3)
        a=Secondary(Map(amp=.03),porb=2.,r=.2,m=.01,ecc=.12,inc=1.54,w=1.2,t0=.001)
        b=Secondary(Map(amp=.01),a=8.,ecc=.05,r=.15,m=.005,inc=1.55,t0=.006,Omega=.03)
        s=System(p,a,b,light_delay=True,texp=.003,oversample=3,order=2)
        self.check_system(s,np.array([-.008,.012]))

    def test_reflected_and_oblate(self):
        m=Map(1,oblate=True,f=.15,inc=1.1,obl=.3);m.y[1:]=[.1,.03,-.04]
        p=Primary(m,prot=1.3)
        r=Map(1,reflected=True,roughness=.2,source_npts=1);r.y[1:]=[.04,.03,.02]
        a=Secondary(r,porb=2.,r=.15,m=.002,ecc=.1,inc=1.54)
        self.check_system(System(p,a,texp=.002,oversample=3),np.array([.01,.8]))

    def test_spectral_angles_and_gravity(self):
        m=Map(1,nw=2,oblate=True,f=.15,fdeg=2,omega=.3,inc=70.,angle_unit='deg')
        m.y[1:]=[[.03,.02],[.01,.02],[-.02,.04]]
        p=Primary(m,prot=.8,theta0=20.)
        a=Secondary(Map(amp=.01,angle_unit='deg'),porb=2.,r=.15,ecc=.1,inc=88.,w=75.)
        self.check_system(System(p,a,texp=.002,oversample=3),np.array([.01]))

    def test_finite_source_radius_and_mass(self):
        p=Primary(Map(amp=1.2),m=1.1,r=.8)
        r=Map(1,reflected=True,roughness=.2,source_npts=4);r.y[1:]=[.03,.01,.02]
        a=Secondary(r,porb=2.,m=.003,r=.2,ecc=.1,inc=1.5)
        self.check_system(System(p,a,light_delay=True),np.array([.3,.7]))

    def test_rv_orbits_spins_masses_radii_and_exposure(self):
        m=Map(2,rv=True,inc=1.2,obl=.2,udeg=1,veq=12000.,alpha=.2)
        m.y[1:]=np.arange(8)*.01;m.u[1]=.3
        p=Primary(m,prot=.8,theta0=.3)
        a=Secondary(Map(1,rv=True,veq=5000.),porb=2.,r=.2,m=.01,ecc=.12,inc=1.54,w=1.2,t0=.001)
        b=Secondary(Map(amp=.01),a=8.,ecc=.05,r=.15,m=.005,inc=1.55,t0=.006,Omega=.03)
        for exposure in [0.,.003]:
            s=System(p,a,b,light_delay=True,texp=exposure,oversample=3,order=2)
            for keplerian in [False,True]:self.check_system(s,np.array([-.008,.012]),rv=keplerian)


if __name__=='__main__':unittest.main()

