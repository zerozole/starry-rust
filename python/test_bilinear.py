import unittest
import numpy as np
from starry_rust import Map,DopplerMap

class BilinearDoppler(unittest.TestCase):
    def test_normalized_and_unnormalized_joint_fit(self):
        for normalize in [False,True]:
            m=Map(1,inc=1.1);m.y[1:]=[.07,-.04,.13]
            rest=1.-.5*np.exp(-((np.arange(19)-9.)/1.5)**2)
            d=DopplerMap(m,rest,log_spacing=2e-5,half_width=3,veq=15000.)
            phases=np.linspace(0.,5.,9)
            data=d.flux(phases,normalize=normalize)
            m.y[1:]*=.8;d.spectra[:]=1.-.8*(1.-rest)
            initial=np.sum((d.flux(phases,normalize=normalize)-data)**2)
            result=d.solve(data,theta=phases,solver='bilinear',normalize=normalize,
                           flux_err=.001,prior_cov={'map':1.,'spectrum':.2},iterations=20,tolerance=1e-8)
            final=np.sum((d.flux(phases,normalize=normalize)-data)**2)
            self.assertLess(final,initial*.01)
            scores=np.asarray(result['objective'])
            self.assertTrue(np.all(np.diff(scores)<=1e-7*(1+scores[:-1])))
            self.assertEqual(result['joint_covariance'].shape,(23,23))
            _,j=d.map_jacobian(phases,normalize)
            j=np.hstack((j,d.design_matrix(phases,fix_map=True,normalize=normalize)))
            precision=j.T@j/1e-6+np.diag(np.r_[np.ones(4),np.full(19,5.)])
            np.testing.assert_allclose(result['joint_covariance'],np.linalg.inv(precision),atol=1e-8,rtol=1e-7)
            self.assertGreater(np.max(np.abs(result['joint_covariance'][:4,4:])),1e-7)

if __name__=='__main__':unittest.main()
