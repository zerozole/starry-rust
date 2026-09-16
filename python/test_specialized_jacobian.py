import unittest
import numpy as np
from starry_rust import Map


class FullSpecializedJacobians(unittest.TestCase):
    def test_view_coefficients_limb_and_amplitude(self):
        for options in [dict(reflected=True,roughness=.3),dict(oblate=True,f=.2,fdeg=2,omega=.3)]:
            m=Map(2,udeg=2,inc=1.1,obl=.3,amp=1.4,**options)
            m[1],m[2]=.3,.1;m[1,1]=.2
            kw=dict(xo=.6,yo=.2,ro=.3,theta=.4,xs=.4,ys=.6,zs=1.3)
            jac=m.flux_jacobian(**kw)
            np.testing.assert_allclose(jac['flux'],m.flux(**kw),atol=2e-11)
            for key in ['theta','inc','obl','amp']:
                old=kw[key] if key=='theta' else getattr(m,key);h=1e-5
                if key=='theta':
                    kw[key]=old+h;a=m.flux(**kw);kw[key]=old-h;b=m.flux(**kw);kw[key]=old
                else:
                    setattr(m,key,old+h);a=m.flux(**kw);setattr(m,key,old-h);b=m.flux(**kw);setattr(m,key,old)
                np.testing.assert_allclose(jac[key],(a-b)/(2*h),atol=3e-8,rtol=2e-6,err_msg=key)
            for target,count,offset in [('y',m.Ny,0),('u',m.udeg,1)]:
                values=getattr(m,target)
                for j in range(count):
                    old=values[j+offset];values[j+offset]=old+1e-5;a=m.flux(**kw)
                    values[j+offset]=old-1e-5;b=m.flux(**kw);values[j+offset]=old
                    np.testing.assert_allclose(jac[target][j],(a-b)/2e-5,atol=3e-8,rtol=2e-6)


if __name__=='__main__':unittest.main()
