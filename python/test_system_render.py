import unittest
import numpy as np
from starry_rust import Map,Primary,Secondary,System


class SystemRendering(unittest.TestCase):
    def test_native_images_positions_and_phases(self):
        primary=Primary(Map(2,angle_unit='deg'),prot=2.,theta0=20.)
        primary.map[1,1]=.2
        secondary=Secondary(Map(1,reflected=True,angle_unit='deg'),porb=3.,r=.1)
        system=System(primary,secondary)
        times=np.array([0.,.3])
        result=system.render(times,res=12)
        self.assertEqual(result['images'][0].shape,(2,12,12))
        np.testing.assert_allclose(result['positions'],np.stack(system.position(times),axis=-1).transpose(1,0,2))
        for i,t in enumerate(times):
            np.testing.assert_allclose(result['images'][0][i],primary.map.render(res=12,theta=20.+180.*t),equal_nan=True)
            source=(result['positions'][i,0]-result['positions'][i,1])/.1
            expected=secondary.map.render(res=12,xs=source[0],ys=source[1],zs=source[2],Rs=10.)*primary.map.amp
            np.testing.assert_allclose(result['images'][1][i],expected,equal_nan=True)


if __name__=='__main__':unittest.main()
