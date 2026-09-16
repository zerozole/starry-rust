import unittest
import numpy as np
from starry_rust import Map,DopplerMap


class DopplerDerivatives(unittest.TestCase):
    def test_physical_kernel_and_spectrum_derivatives(self):
        rest=1.-.4*np.exp(-((np.arange(25)-12.)/2.)**2)
        for normalize in [False,True]:
            m=Map(3,inc=65.,angle_unit='deg',amp=1.3);m[1,1]=.2;m[2,-1]=-.1
            d=DopplerMap(m,rest,log_spacing=2e-5,half_width=4,veq=21500.,u=[.3,.1])
            phases=np.array([10.,50.]);jac=d.flux_jacobian(phases,normalize)
            np.testing.assert_allclose(jac['flux'],d.flux(phases,normalize),atol=2e-13)
            for key,h in [('theta',1e-3),('inc',1e-3),('veq',.1),('log_spacing',1e-10)]:
                if key=='theta':a=d.flux(phases+h,normalize);b=d.flux(phases-h,normalize)
                else:
                    owner=m if key=='inc' else d;old=getattr(owner,key)
                    setattr(owner,key,old+h);a=d.flux(phases,normalize)
                    setattr(owner,key,old-h);b=d.flux(phases,normalize);setattr(owner,key,old)
                np.testing.assert_allclose(jac[key],(a-b)/(2*h),rtol=3e-5,atol=2e-7)
            for i in range(2):
                old=d.u[i];d.u[i]=old+1e-5;a=d.flux(phases,normalize)
                d.u[i]=old-1e-5;b=d.flux(phases,normalize);d.u[i]=old
                np.testing.assert_allclose(jac['u'][...,i],(a-b)/2e-5,atol=3e-9)

    def test_multicomponent_resampling(self):
        m1=Map(1,inc=1.1);m2=Map(1,inc=1.1);m1[1,1]=.1;m2[1,-1]=.1
        wav=np.linspace(500.,500.2,21);wav0=np.linspace(499.8,500.4,31)
        rest=np.array([1.-.4*np.exp(-((wav0-500.1)/.03)**2),1.-.3*np.exp(-((wav0-500.15)/.02)**2)])
        d=DopplerMap.from_wavelengths([m1,m2],wav,wav0,rest,veq=20000.)
        jac=d.flux_jacobian(.4)
        np.testing.assert_allclose(jac['flux'],d.flux(.4),atol=1e-13)
        d.veq+=.1;a=d.flux(.4);d.veq-=.2;b=d.flux(.4)
        np.testing.assert_allclose(jac['veq'],(a-b)/.2,atol=2e-10)

    def test_high_degree_and_zero_rotation(self):
        m=Map(24,inc=1.1);m[24,3]=.1
        d=DopplerMap(m,np.ones(17),log_spacing=2e-5,half_width=3,veq=21000.,u=[.3])
        jac=d.flux_jacobian(.2)
        np.testing.assert_allclose(jac['flux'],1.,atol=1e-12)
        for key in ('theta','inc','veq','log_spacing'):np.testing.assert_allclose(jac[key],0.,atol=1e-9)
        d.veq=0.
        np.testing.assert_array_equal(d.flux_jacobian(.2)['veq'],0.)


if __name__=='__main__':unittest.main()
