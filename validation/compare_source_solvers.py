"""Execute unmodified upstream solver method bodies with NumPy primitives.

Native forward design matrices are shared inputs; reference inference and the
schedule execute the pinned source code, not Rust inference. No Theano import.
"""
import ast
import json
import pathlib
import sys
import types
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map,DopplerMap

def block_diag(*blocks):
    n=sum(len(b) for b in blocks);out=np.zeros((n,n));i=0
    for b in blocks:out[i:i+len(b),i:i+len(b)]=b;i+=len(b)
    return out

def cho_solve(l,b):return np.linalg.solve(l.T,np.linalg.solve(l,b))
namespace=dict(np=np,tt=np,slinalg=np.linalg,_cho_solve=cho_solve,cho_solve=cho_solve,
               cho_factor=np.linalg.cholesky,block_diag=block_diag,tqdm=lambda x,**kw:x)
def extract(filename,classname,names):
    source=ROOT.parent/'starry-upstream/starry'/filename;tree=ast.parse(source.read_text())
    cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==classname)
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for method in methods:method.decorator_list=[]
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),namespace)
extract('_core/math.py','LinAlgType',['solve'])
namespace['greedy_linalg']=types.SimpleNamespace(solve=lambda *a:namespace['solve'](None,*a))
names=['solve_for_map_linear','solve_for_map_tempered','solve_for_spectrum_linear','solve_for_everything_bilinear']
extract('doppler_solve.py','Solve',names)

class Reference:
    def __init__(self,model,theta,data,normalized):
        self.model=model;self.theta=theta;self.flux=data;self.nt,self.nw=data.shape;self.nc=1;self.Ny=4;self.nw0_=model.spectra.shape[1]
        self.y=model._weights()[:,None];self.spectrum_=model.spectra.copy();self.spectral_guess=self.spectrum_.copy()
        self.spatial_mean=self.y.copy();self.spatial_inv_cov=np.eye(4)[:,:,None]
        self.spectral_mean=self.spectrum_.copy();self.spectral_inv_cov=np.eye(self.nw0_)[None,:,:]
        self.flux_err=np.array(.01);self.baseline=None;self.normalized=normalized;self.linear=not normalized
        self.baseline_var=.01;self.T=np.logspace(2,0,3);self.quiet=True;self.meta={};self._S=None;self._C=None
        self.continuum_idx=4;self.spectral_method='L2'
    def sync(self):self.model._set_weights(self.y.T.ravel());self.model.spectra=self.spectrum_.copy()
    @property
    def S(self):
        if self._S is None:self.sync();self._S=self.model.design_matrix(self.theta,fix_spectrum=True)
        return self._S
    @property
    def C(self):
        if self._C is None:self._C=self.S.reshape(self.nt,self.nw,-1)[:,self.continuum_idx,:]
        return self._C
    def dotMT(self,x):self.sync();return self.model.design_matrix(self.theta,fix_map=True).T@x
    def dotM(self,x):self.sync();return self.model.design_matrix(self.theta,fix_map=True)@x
for name in names:setattr(Reference,name,namespace[name])

def model():
    m=Map(1,inc=1.1);m.y[1:]=[.07,-.04,.13]
    rest=1-.4*np.exp(-((np.arange(17)-8)/2)**2)
    return DopplerMap(m,rest,log_spacing=2e-5,half_width=3,veq=15000.,continuum_index=4)

def main():
    errors=[];theta=np.array([0.,.8,1.6])
    for normalize in [False,True]:
        for mode in ['map','spectrum','combined']:
            d=model();data=d.flux(theta,normalize=normalize);data+=.0001*np.sin(np.arange(data.size)).reshape(data.shape)
            r=Reference(model(),theta,data,normalize)
            if mode=='map':
                if normalize:
                    r.solve_for_map_tempered();fitted=d.solve_tempered(data,theta=theta,flux_err=.01,steps=3,log_temperature=(2*np.log(10),0.))
                    actual=fitted['map'];chol=fitted['cholesky']
                else:r.solve_for_map_linear();actual,chol=d.solve(data,theta=theta,solver='map',flux_err=.01)
                expected=r.y.T.ravel();cov=r.cho_ycov@r.cho_ycov.T
            elif mode=='spectrum':
                r.solve_for_spectrum_linear();actual,chol=d.solve(data,theta=theta,solver='spectrum',flux_err=.01,normalize=normalize)
                expected=r.spectrum_.ravel();cov=r.cho_scov@r.cho_scov.T
            else:
                r.solve_for_everything_bilinear();fitted=d.solve_bilinear_tempered(data,theta=theta,flux_err=.01,normalize=normalize,steps=3)
                actual=np.r_[fitted['map'],d.spectra.ravel()];expected=np.r_[r.y.T.ravel(),r.spectrum_.ravel()]
                chol=fitted['map_cholesky'];cov=r.cho_ycov@r.cho_ycov.T
            np.testing.assert_allclose(actual,expected,atol=3e-10,rtol=2e-9,err_msg=f'{normalize} {mode}')
            np.testing.assert_allclose(chol@chol.T,cov,atol=3e-10,rtol=2e-9)
            errors.append(float(np.max(abs(actual-expected))))
    report=dict(cases=len(errors),max_absolute_error=max(errors),passed=True,
        reference='unmodified upstream map/spectrum/tempering method bodies and LinAlgType.solve, NumPy primitives, shared native design inputs')
    (ROOT/'validation/source_solvers_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
