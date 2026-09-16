import ctypes as ct
import unittest
import numpy as np
from starry_rust import Map,DopplerMap
from _starry_api import _dll,_ptr,array,ptr,check


class SparseInference(unittest.TestCase):
    def test_native_upstream_iteration_and_soft_threshold(self):
        rng=np.random.default_rng(192)
        a=rng.normal(size=(40,7));normal=array(a.T@a);rhs=array(rng.normal(size=7)*10)
        lam=.7;eps=1e-12;tol=1e-18
        w=np.ones(7)
        for iteration in range(1000):
            nxt=np.linalg.solve(normal+np.diag(lam/np.maximum(abs(w),lam*eps)),rhs)
            step=np.sum((w-nxt)**2);w=nxt
            if step<tol:break
        out=np.empty(9)
        _dll.starry_l1.argtypes=[_ptr,_ptr,ct.c_size_t,ct.c_double,ct.c_size_t,ct.c_double,ct.c_double,_ptr]
        check(_dll.starry_l1(ptr(normal),ptr(rhs),7,lam,1000,eps,tol,ptr(out)))
        np.testing.assert_allclose(out[:7],w,atol=1e-12)
        self.assertEqual(out[-2],iteration+1)
        normal=array(np.eye(7));rhs=array([3.,-.2,-2.,0.,.1,1.,-4.])
        check(_dll.starry_l1(ptr(normal),ptr(rhs),7,lam,1000,eps,tol,ptr(out)))
        np.testing.assert_allclose(out[:7],np.sign(rhs)*np.maximum(abs(rhs)-lam,0.),atol=1e-8)

    def test_sparse_spectrum_and_epoch_errors(self):
        surface=Map(1)
        rest=np.ones(21);rest[9:12]-=.3
        model=DopplerMap(surface,np.ones(21),log_spacing=2e-5,half_width=2,veq=18000.)
        model.spectra=rest[None,:]
        theta=np.array([0.,.5,1.])
        truth=model.flux(theta,normalize=False)
        model.spectra=np.ones((1,21))
        fit=model.solve(truth,theta=theta,solver='spectrum',flux_err=np.array([.01,.02,.01]),
            prior_mean=1.,spectral_method='L1',spectral_lambda=.01,iterations=1000,tolerance=1e-16)
        self.assertIsNone(fit['covariance'])
        np.testing.assert_allclose(model.flux(theta,normalize=False),truth,atol=2e-5)


if __name__=='__main__':unittest.main()
