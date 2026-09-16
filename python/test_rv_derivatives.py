import unittest
import numpy as np
from starry_rust import Map


class RVDerivatives(unittest.TestCase):
    def test_parameters_coefficients_and_limb(self):
        m=Map(2,udeg=2,rv=True,inc=1.1,obl=.3,veq=14000.,alpha=.2)
        m.y[1:]=np.arange(8)*.013;m.u[1:]=[.3,.1]
        kw=dict(theta=.4,xo=.45,yo=.2,ro=.25)
        jac=m.rv_jacobian(**kw)
        np.testing.assert_allclose(jac['rv'],m.rv(**kw),atol=1e-9)
        for key in ['theta','inc','obl','xo','yo','ro','amp','veq','alpha']:
            old=kw[key] if key in kw else getattr(m,key);h=1e-5 if key!='veq' else .1
            def assign(value):
                if key in kw:kw[key]=value
                else:setattr(m,key,value)
            assign(old+h);a=m.rv(**kw);assign(old-h);b=m.rv(**kw);assign(old)
            np.testing.assert_allclose(jac[key],(a-b)/(2*h),rtol=2e-5,atol=2e-5,err_msg=key)
        for key,offset in [('y',0),('u',1)]:
            vector=getattr(m,key)
            for i in range(offset,len(vector)):
                old=vector[i];h=1e-5;vector[i]=old+h;a=m.rv(**kw);vector[i]=old-h;b=m.rv(**kw);vector[i]=old
                np.testing.assert_allclose(jac[key][i-offset],(a-b)/(2*h),rtol=2e-5,atol=2e-5,err_msg=f'{key}{i}')

    def test_spectral_units_zero_amplitude_and_eclipse(self):
        m=Map(1,nw=2,rv=True,angle_unit='deg',inc=70.,obl=20.,veq=10000.)
        m.y[1:]=[[.02,.03],[.04,.01],[-.01,.03]];m.amp=[0.,1.]
        kw=dict(theta=[10.,30.],xo=.4,yo=.2,ro=.2)
        jac=m.rv_jacobian(**kw)
        np.testing.assert_allclose(jac['rv'],m.rv(**kw),atol=1e-9)
        a=m.rv(**dict(kw,theta=np.array(kw['theta'])+.001));b=m.rv(**dict(kw,theta=np.array(kw['theta'])-.001))
        np.testing.assert_allclose(jac['theta'],(a-b)/.002,atol=1e-7)
        self.assertTrue(np.all(m.rv_jacobian(ro=2.)['rv']==0))
        np.testing.assert_allclose(m.rv_jacobian(zo=-1.,ro=.2)['ro'],0.)

    def test_high_degree(self):
        m=Map(13,rv=True,veq=13000.,alpha=.2,inc=1.1)
        m.y[1]=.03;m.y[-1]=.002
        kw=dict(xo=.4,yo=.1,ro=.2,theta=.3)
        jac=m.rv_jacobian(**kw)
        np.testing.assert_allclose(jac['rv'],m.rv(**kw),atol=1e-7)
        np.testing.assert_allclose(jac['theta'],(m.rv(**dict(kw,theta=.30001))-m.rv(**dict(kw,theta=.29999)))/.00002,atol=1e-5)


if __name__=='__main__':unittest.main()
