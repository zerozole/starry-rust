import unittest
import numpy as np
from starry_rust import Map,Primary,Secondary,System
import test_system_derivatives
import test_system_surface


class SourceSystem(unittest.TestCase):
    def test_source_convention_derivatives(self):
        p=Primary(Map(1,rv=True,veq=3500.,inc=1.2),m=1.1,prot=.8)
        p.map.y[1:]=[.03,.02,-.01]
        a=Secondary(Map(1,rv=True,veq=700.,amp=.02),porb=2.,m=.01,r=.15,ecc=.2,inc=1.54,w=1.1,t0=.001)
        b=Secondary(Map(rv=True,amp=.01),a=8.,m=.005,r=.1,ecc=.05,inc=1.55,t0=.005)
        s=System(p,a,b,light_delay=True,texp=.002,oversample=3,orbit_convention='starry')
        times=np.array([-.01,.014])
        helper=test_system_derivatives.SystemDerivatives()
        helper.check_system(s,times)
        helper.check_system(s,times,rv=True)
        test_system_surface.SystemSurface().verify(s,times,rv=True)
        # A source convention cannot be silently misspelled.
        with self.assertRaises(ValueError):System(p,a,orbit_convention='stellar')

if __name__=='__main__':unittest.main()
