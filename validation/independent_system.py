"""Independent circular two-body reference: analytic disk overlap and complex AD.

No native forward evaluations are used to construct reference derivatives.
Includes retarded barycentric positions, Kepler mass scaling and exposures.
"""
import json
import pathlib
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map,Primary,Secondary,System

def reference(parameters,delay):
    time,exposure,m0,m1,r0,r1,period,inc,epoch,amp0,amp1=parameters
    a=(1.3271244e20*(m0+m1)*(period*86400/(2*np.pi))**2)**(1/3)/6.957e8
    factors=np.array([-m1,m0])/(m0+m1);speed=299792458.*86400/6.957e8
    def apparent(t):
        ts=np.array([t,t]);n=2*np.pi/period
        for _ in range(20 if delay else 0):ts=t+factors*a*np.sin(inc)*np.cos(n*(ts-epoch))/speed
        phase=n*(ts-epoch)
        x=factors*a*np.sin(phase);y=-factors*a*np.cos(inc)*np.cos(phase)
        rv=factors*a*np.sin(inc)*n*np.sin(phase)*6.957e8/86400
        return x,y,rv
    flux=0.
    for offset,weight in [(-.5,1/6),(0.,4/6),(.5,1/6)]:
        x,y,_=apparent(time+exposure*offset)
        b=np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2)/r0;r=r1/r0
        assert abs(1-r.real)<b.real<1+r.real
        c0=(b*b+1-r*r)/(2*b);c1=(b*b+r*r-1)/(2*b*r)
        lens=np.arccos(c0)+r*r*np.arccos(c1)-.5*np.sqrt((-b+1+r)*(b+1-r)*(b-1+r)*(b+1+r))
        flux+=weight*(amp0*(1-lens/np.pi)+amp1)
    return np.r_[flux,apparent(time)[2]]

def main():
    keys=['time','texp','body0.m','body1.m','body0.r','body1.r','body1.porb','body1.inc','body1.t0','body0.map.amp','body1.map.amp']
    flux_errors=[];rv_errors=[]
    for delay in [False,True]:
        for exposure in [0.,.002]:
            for time in [.042,.048]:
                parameters=np.array([time,exposure,1.,.01,1.,.2,2.,1.54,0.,1.2,.03])
                p=Primary(Map(amp=parameters[-2]),m=1.)
                b=Secondary(Map(amp=parameters[-1]),porb=2.,m=.01,r=.2,inc=1.54)
                s=System(p,b,light_delay=delay,texp=exposure,oversample=3,order=2)
                fj=s.flux_jacobian([time],surface=True);rj=s.rv_jacobian([time],total=False,surface=True)
                expected=reference(parameters,delay)
                np.testing.assert_allclose(fj['flux'][0],expected[0],atol=2e-12)
                np.testing.assert_allclose(rj['rv'][:,0],expected[1:],atol=2e-7)
                for j,key in enumerate(keys):
                    shifted=parameters.astype(complex);shifted[j]+=1e-25j
                    derivative=reference(shifted,delay).imag/1e-25
                    fe=float(abs(fj[key][0]-derivative[0]));re=float(np.max(abs(rj[key][:,0]-derivative[1:])))
                    np.testing.assert_allclose(fj[key][0],derivative[0],atol=2e-9,rtol=2e-8,err_msg=key)
                    np.testing.assert_allclose(rj[key][:,0],derivative[1:],atol=2e-6,rtol=2e-8,err_msg=key)
                    flux_errors.append(fe);rv_errors.append(re)
    report=dict(cases=len(flux_errors),flux_max_absolute_error=max(flux_errors),rv_max_absolute_error=max(rv_errors),passed=True,
        reference='independent analytic disk overlap, circular Kepler orbit and retarded barycentric positions; complex-step derivatives')
    (ROOT/'validation/independent_system_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
