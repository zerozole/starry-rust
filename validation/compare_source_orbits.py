"""Execute exoplanet 0.4.5 position/velocity/light-delay methods with NumPy.

The vendored source is the minimum dependency version in pinned starry setup.py.
Kepler's equation is independently solved here; no native orbital values are
used as reference inputs. Source method bodies are compiled unchanged.
"""
import ast
import hashlib
import json
import pathlib
import sys
import types
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map,Primary,Secondary,System

source=ROOT/'validation/exoplanet_keplerian_reference.py.txt'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f5ab60650ffbaffd7e8c61c35effc3954aa6bc5b831c048431d52b4871d1209a'
tree=ast.parse(source.read_text())
cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='KeplerianOrbit')
names=['_rotate_vector','_get_position','_get_retarded_position','get_relative_position','get_planet_position','get_star_position','_get_velocity','get_star_velocity']
methods=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in names]
tt=types.SimpleNamespace(**{n:getattr(np,n) for n in ['sqrt','squeeze','sin','cos','abs']},switch=np.where,lt=np.less,abs_=np.abs,shape_padright=lambda x:np.asarray(x))
namespace=dict(np=np,tt=tt,c_light=299792458.*86400/6.957e8)
for method in methods:method.decorator_list=[]
exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),namespace)

class Reference:
    def __init__(self,primary,body):
        self.period=body.porb;self.ecc=body.ecc;self.t0=body.t0
        self.a=(1.3271244e20*(primary.m+body.m)*(self.period*86400/(2*np.pi))**2)**(1/3)/6.957e8
        self.m_star=primary.m;self.m_planet=body.m;self.m_total=primary.m+body.m
        self.a_star=self.a*body.m/self.m_total;self.a_planet=-self.a*primary.m/self.m_total
        self.K0=2*np.pi/self.period*self.a/self.m_total/np.sqrt(1-self.ecc**2)
        self.Omega=body.Omega
        for name,angle in [('omega',body.w),('incl',body.inc),('Omega',body.Omega)]:
            setattr(self,'cos_'+name,np.cos(angle));setattr(self,'sin_'+name,np.sin(angle))
        e0=2*np.arctan2(np.sqrt(1-self.ecc)*np.cos(body.w),np.sqrt(1+self.ecc)*(1+np.sin(body.w)))
        self.m0=e0-self.ecc*np.sin(e0)
    def _get_true_anomaly(self,t,_pad=True):
        mean=self.m0+2*np.pi/self.period*(np.asarray(t)-self.t0)
        e=mean.copy()
        for _ in range(60):e-=(e-self.ecc*np.sin(e)-mean)/(1-self.ecc*np.cos(e))
        den=1-self.ecc*np.cos(e)
        return np.sqrt(1-self.ecc**2)*np.sin(e)/den,(np.cos(e)-self.ecc)/den
for name in names:setattr(Reference,name,namespace[name])

errors=[];cases=0
for delay in (False,True):
    p=Primary(Map(rv=True),m=1.2)
    a=Secondary(Map(rv=True,amp=.02),porb=2.,m=.1,r=.1,ecc=.4,inc=1.54,w=1.2,Omega=.2)
    b=Secondary(Map(rv=True,amp=.01),porb=3.,m=.2,r=.07,ecc=.7,inc=1.52,w=.7,Omega=-.3,t0=.04)
    s=System(p,a,b,light_delay=delay,orbit_convention='starry')
    t=np.array([-.03,.009,.21,.73,1.3])
    refs=[Reference(p,x) for x in (a,b)]
    expected=np.array([np.sum([r.get_star_position(t) for r in refs],axis=0),*[r.get_planet_position(t,light_delay=delay) for r in refs]]).transpose(1,0,2)
    error=np.max(np.abs(np.array(s.position(t))-expected));errors.append(float(error))
    np.testing.assert_allclose(s.position(t),expected,atol=3e-8,rtol=1e-9)
    orbital=np.array([np.zeros_like(t),*[-r.get_star_velocity(t)[2]*6.957e8/86400 for r in refs]])
    np.testing.assert_allclose(s.rv(t,total=False)-s.rv(t,total=False,keplerian=False),orbital,atol=3e-9,rtol=1e-12)
    # Reference relative geometry drives independent isolated-map occultations.
    xyz=np.array([np.zeros((3,len(t))),*[r.get_relative_position(t,light_delay=delay) for r in refs]])
    np.testing.assert_allclose(s.render(t,res=8)['positions'],xyz.transpose(2,0,1),atol=3e-8)
    for k in range(len(t)):
        flux=[]
        for i,body in enumerate(s.bodies):
            occ=[]
            for j,other in enumerate(s.bodies):
                d=xyz[j,:,k]-xyz[i,:,k]
                if d[2]>0 and np.hypot(d[0],d[1])<body.r+other.r:occ.append([d[0]/body.r,d[1]/body.r,other.r/body.r])
            assert len(occ)<=1, 'reference cases require at most one covering disk'
            flux.append(body.map.flux(*occ[0]) if occ else body.map.flux())
        np.testing.assert_allclose(s.flux([t[k]],total=False)[:,0],np.ravel(flux),atol=3e-8)
        cases+=len(flux)
report=dict(reference='exoplanet v0.4.5 unmodified method bodies and independent Newton solver',sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=cases+60,max_position_error=max(errors))
(ROOT/'validation/source_orbits_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
