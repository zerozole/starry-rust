import unittest
import numpy as np
from starry_rust import Map,Primary,Secondary,System


class SystemSurface(unittest.TestCase):
    def verify(self,s,t,rv=False):
        evaluate=(lambda:s.rv(t,total=False)) if rv else (lambda:s.flux(t,total=False))
        jac=s.rv_jacobian(t,total=False,surface=True) if rv else s.flux_jacobian(t,total=False,surface=True)
        np.testing.assert_allclose(jac['rv' if rv else 'flux'],evaluate(),atol=1e-7)
        for i,b in enumerate(s.bodies):
            m=b.map;keys=['inc','obl','amp']+(['roughness'] if m.reflected else [])+(['f','omega','beta','tpole','wav'] if m.oblate else [])
            for key in keys:
                old=getattr(m,key);h=abs(old)*1e-5 if key in ('tpole','wav') else 2e-6
                setattr(m,key,old+h);a=evaluate();setattr(m,key,old-h);c=evaluate();setattr(m,key,old)
                np.testing.assert_allclose(jac[f'body{i}.map.{key}'],(a-c)/(2*h),rtol=4e-4,atol=2e-3 if rv else 3e-7,err_msg=f'{i}.{key}')
            for key,offset in [('y',0),('u',1)]:
                vector=getattr(m,key)
                for j in range(offset,len(vector)):
                    old=vector[j].copy();h=2e-6
                    vector[j]=old+h;a=evaluate();vector[j]=old-h;c=evaluate();vector[j]=old
                    np.testing.assert_allclose(jac[f'body{i}.map.{key}'][...,j-offset],(a-c)/(2*h),rtol=4e-4,atol=2e-3 if rv else 3e-7,err_msg=f'{i}.{key}{j}')

    def test_reflected_oblate_overlap_and_gravity(self):
        m=Map(1,udeg=1,oblate=True,f=.15,fdeg=2,omega=.3,inc=1.1,obl=.3);m.y[1:]=[.03,.02,-.04];m.u[1]=.2
        p=Primary(m,prot=.8,theta0=.3)
        r=Map(1,reflected=True,udeg=1,roughness=.2,source_npts=4);r.y[1:]=[.01,.02,.03];r.u[1]=.1
        a=Secondary(r,porb=2.,r=.2,m=.01,ecc=.05,inc=1.54)
        b=Secondary(Map(amp=.01),a=8.,ecc=.05,r=.15,m=.005,inc=1.55,t0=.006,Omega=.03)
        self.verify(System(p,a,b,light_delay=True,texp=.002,oversample=3),np.array([.012,.99]))

    def test_rv_surface_spectral_exposure(self):
        m=Map(1,nw=2,rv=True,angle_unit='deg',inc=70.,obl=20.,udeg=1,veq=12000.,alpha=.2)
        m.y[1:]=[[.02,.03],[.01,-.02],[.04,.01]];m.u[1]=.2
        p=Primary(m,theta0=20.)
        a=Secondary(Map(1,rv=True,veq=5000.,amp=.02),porb=2.,r=.2,m=.01,ecc=.05,inc=1.54)
        self.verify(System(p,a,light_delay=True,texp=.003,oversample=3),np.array([.01]))

    def test_zero_primary_amplitude_reflection_coupling(self):
        p=Primary(Map(1,amp=0.))
        a=Secondary(Map(1,reflected=True),porb=2.,r=.2,inc=1.5)
        s=System(p,a);t=np.array([.3]);jac=s.flux_jacobian(t,total=False,surface=True)
        p.map.amp=1.;expected=s.flux(t,total=False)
        np.testing.assert_allclose(jac['body0.map.amp'],expected,atol=2e-12)
        np.testing.assert_array_equal(jac['body1.map.y'],0.)


if __name__=='__main__':unittest.main()
