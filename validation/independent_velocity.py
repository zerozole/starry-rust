"""Independent velocity references: analytic dipole chords and disk integration."""
import json
import pathlib
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map,DopplerMap

def vector(theta,inc,obl):
    v=np.array([.08,.05,-.03],dtype=np.result_type(theta,inc,obl))
    c,s=np.cos(theta),np.sin(theta);v=np.array([c*v[0]+s*v[2],v[1],-s*v[0]+c*v[2]])
    c,s=np.sin(inc),np.cos(inc);v=np.array([v[0],c*v[1]-s*v[2],s*v[1]+c*v[2]])
    c,s=np.cos(obl),np.sin(obl);return np.array([c*v[0]-s*v[1],s*v[0]+c*v[1],v[2]])

def spectrum(p,rest,normalize):
    theta,inc,veq,spacing,u=p;a=vector(theta,inc,0.)
    x=-299792458.*np.tanh(np.arange(-4,5)*spacing)/(veq*np.sin(inc))
    h=np.sqrt(np.where(abs(x.real)<1,1-x*x,0.))
    chord=(1-u)*2*h*(1+np.sqrt(3)*a[0]*x)+u*np.pi*h*h/2*(1+np.sqrt(3)*a[0]*x)
    chord+=np.sqrt(3)*a[2]*((1-u)*np.pi*h*h/2+u*4*h**3/3)
    kernel=1.2*chord/(np.sum(2*h)*(1-u/3))
    output=np.array([kernel@rest[i:i+9] for i in range(len(rest)-8)])
    return output/kernel.sum() if normalize else output

nodes,weights=np.polynomial.legendre.leggauss(96)
mu=(nodes+1)/2;az=np.arange(192)*2*np.pi/192
x=np.sqrt(1-mu[:,None]**2)*np.cos(az);y=np.sqrt(1-mu[:,None]**2)*np.sin(az);z=np.broadcast_to(mu[:,None],x.shape)
area=weights[:,None]*mu[:,None]*np.pi/192
def rv(p):
    theta,inc,obl,veq,alpha,u=p;a=vector(theta,inc,obl)
    intensity=(1+np.sqrt(3)*(a[0]*x+a[1]*y+a[2]*z))*(1-u*(1-z))
    axis=-np.sin(inc)*np.sin(obl)*x+np.sin(inc)*np.cos(obl)*y+np.cos(inc)*z
    velocity=veq*np.sin(inc)*(np.cos(obl)*x+np.sin(obl)*y)*(1-alpha*axis**2)
    return np.sum(area*intensity*velocity)/np.sum(area*intensity)

def main():
    errors=[];rv_errors=[];rest=1-.4*np.exp(-((np.arange(25)-12)/2)**2)
    for theta in [.2,1.3]:
        for inc in [.6,1.1]:
            m=Map(1,amp=1.2,inc=inc,obl=.3,rv=True,veq=21000.,alpha=.2,udeg=1);m.y[1:]=[.05,-.03,.08];m.u[1]=.3
            jac=m.rv_jacobian(theta=theta)
            rp=[theta,inc,.3,21000.,.2,.3]
            np.testing.assert_allclose(jac['rv'],rv(rp),atol=1e-9)
            for j,key in enumerate(['theta','inc','obl','veq','alpha','u']):
                shifted=np.array(rp,dtype=complex);shifted[j]+=1e-25j
                expected=rv(shifted).imag/1e-25;actual=jac[key][0] if key=='u' else jac[key]
                np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=1e-10);rv_errors.append(float(abs(actual-expected)))
            d=DopplerMap(Map(1,inc=inc,amp=1.2),rest,veq=21000.,log_spacing=2e-5,half_width=4,u=[.3]);d.maps[0].y[1:]=[.05,-.03,.08]
            p=[theta,inc,21000.,2e-5,.3]
            for normalize in [False,True]:
                jac=d.flux_jacobian(theta,normalize)
                np.testing.assert_allclose(jac['flux'],spectrum(p,rest,normalize),atol=2e-12)
                for j,key in enumerate(['theta','inc','veq','log_spacing','u']):
                    shifted=np.array(p,dtype=complex);shifted[j]+=1e-25j
                    expected=spectrum(shifted,rest,normalize).imag/1e-25;actual=jac[key][...,0] if key=='u' else jac[key]
                    np.testing.assert_allclose(actual,expected,atol=3e-8,rtol=1e-9);errors.append(float(np.max(abs(actual-expected))))
    report=dict(doppler_cases=len(errors),rv_cases=len(rv_errors),doppler_max_absolute_error=max(errors),rv_max_absolute_error=max(rv_errors),passed=True,
        reference='analytic dipole chord integrals and independent full-disk physical velocity quadrature; complex-step derivatives')
    (ROOT/'validation/independent_velocity_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
