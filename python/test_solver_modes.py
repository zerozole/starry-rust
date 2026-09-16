import unittest
import numpy as np
from starry_rust import Map,DopplerMap


class SolverModes(unittest.TestCase):
    def model(self):
        m=Map(1,inc=1.1);m.y[1:]=[.07,-.04,.13]
        rest=1.-.5*np.exp(-((np.arange(15)-7.)/1.5)**2)
        return DopplerMap(m,rest,log_spacing=2e-5,half_width=2,veq=15000.)

    def test_joint_nonlinear_objective(self):
        for normalize in [False,True]:
            d=self.model();theta=np.linspace(0.,5.,7);data=d.flux(theta,normalize=normalize)
            d.maps[0].y[1:]*=.8;d.spectra[:]=1.-.8*(1.-d.spectra)
            initial=np.sum((d.flux(theta,normalize=normalize)-data)**2)
            result=d.solve(data,theta=theta,solver='nonlinear',normalize=normalize,
                flux_err=.001,prior_cov={'map':1.,'spectrum':.2},iterations=40,tolerance=1e-8)
            self.assertLess(np.sum((d.flux(theta,normalize=normalize)-data)**2),initial*.01)
            self.assertTrue(np.all(np.diff(result['objective'])<=1e-8))
            _,j=d.map_jacobian(theta,normalize)
            j=np.hstack((j,d.design_matrix(theta,fix_map=True,normalize=normalize)))
            precision=j.T@j/1e-6+np.diag(np.r_[np.ones(4),np.full(15,5.)])
            np.testing.assert_allclose(result['joint_covariance'],np.linalg.inv(precision),atol=1e-8,rtol=1e-7)

    def test_tempered_source_equations_and_fixed_baseline(self):
        d=self.model();theta=np.array([0.,.8,1.6]);data=d.flux(theta)
        design=d.design_matrix(theta,fix_spectrum=True)
        continuum=d._native_design(theta,1);baseline=np.ones(data.size)
        mean=d._weights().copy();variance=1e-4*np.ones(data.size);nw=data.shape[1]
        for temperature in np.exp(np.linspace(2.,0.,4)):
            noise=np.diag(temperature*variance*baseline**2)
            for i in range(3):noise[i*nw:(i+1)*nw,i*nw:(i+1)*nw]+=.01
            precision=design.T@np.linalg.solve(noise,design)+np.eye(4)
            expected=np.linalg.solve(precision,design.T@np.linalg.solve(noise,data.ravel()*baseline)+mean)
            baseline=continuum@expected
        result=d.solve_tempered(data,theta=theta,flux_err=.01,steps=4,log_temperature=(2.,0.))
        np.testing.assert_allclose(result['map'],expected,atol=2e-11)
        np.testing.assert_allclose(result['cholesky']@result['cholesky'].T,np.linalg.inv(precision),atol=2e-11)
        d=self.model();base=d.baseline(theta);unscaled=d.flux(theta,normalize=False)
        a=d.solve(data,theta=theta,solver='map',normalize=True,baseline=base,flux_err=.01)
        d=self.model();b=d.solve(unscaled,theta=theta,solver='map',flux_err=.01*base[:,None])
        np.testing.assert_allclose(a[0],b[0],atol=1e-11)

    def test_bilinear_tempered_conditional_solutions(self):
        for normalized in [False,True]:
            d=self.model();theta=np.array([0.,.8,1.6]);data=d.flux(theta,normalize=normalized)
            mean=d._weights().copy();sm=d.spectra.ravel().copy();base=np.ones(3)
            for temperature in np.logspace(2.,0.,3):
                x=d.design_matrix(theta,fix_spectrum=True);nw=data.shape[1]
                noise=np.diag(np.repeat(temperature*1e-4*base**2,nw))
                if normalized:
                    for i in range(3):noise[i*nw:(i+1)*nw,i*nw:(i+1)*nw]+=.01
                precision=x.T@np.linalg.solve(noise,x)+np.eye(len(mean))
                weights=np.linalg.solve(precision,x.T@np.linalg.solve(noise,(data*base[:,None]).ravel())+mean)
                d._set_weights(weights)
                if normalized:base=d.baseline(theta)
                x=d.design_matrix(theta,fix_map=True);variance=np.repeat(1e-4*base**2,nw)
                spectral=np.linalg.solve(x.T@(x/variance[:,None])+np.eye(len(sm)),x.T@((data*base[:,None]).ravel()/variance)+sm)
                d.spectra=spectral.reshape(d.spectra.shape)
            fitted=self.model();result=fitted.solve_bilinear_tempered(data,theta=theta,flux_err=.01,normalize=normalized,steps=3)
            np.testing.assert_allclose(result['map'],weights,atol=1e-10)
            np.testing.assert_allclose(fitted.spectra.ravel(),spectral,atol=1e-10)
            np.testing.assert_allclose(result['map_cholesky']@result['map_cholesky'].T,np.linalg.inv(precision),atol=1e-10)

    def test_deconvolution_initialization(self):
        d=self.model();theta=np.array([0.,1.,2.]);data=d.flux(theta)
        mean=d.spectra.copy();uniform=DopplerMap(Map(inc=1.1),np.ones(15),log_spacing=2e-5,half_width=2,veq=15000.)
        k=uniform.design_matrix(0.,fix_map=True);f=data.mean(axis=0);residual=f/f[0]-k@mean[0]
        normal=k.T@k/.01**2;rhs=k.T@residual/.01**2;weights=np.ones(15)
        for _ in range(5):weights=np.linalg.solve(normal+np.diag(3./np.maximum(abs(weights),3e-12)),rhs)
        result=d.solve_bilinear_tempered(data,theta=theta,flux_err=.01,steps=1,initialization='deconvolve',spectral_lambda=3.,spectral_iterations=5,tolerance=1e-30)
        np.testing.assert_allclose(result['spectrum_guess'],mean+weights,atol=1e-11)


if __name__=='__main__':unittest.main()
