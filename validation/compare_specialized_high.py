"""Mixed high-degree maps against the pinned upstream specialized kernels."""
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map


def main():
    cases=[]
    suffix='.exe' if sys.platform=='win32' else ''
    for d in [13,16]:
        y=np.sin(np.arange((d+1)**2)*.79)*.01;y[0]=1.
        # The low-level unocculted phase oracle is illumination-aligned (theta=0).
        for b,theta,bo,ro,rough in [(-.4,.7,.5,.2,0.),(-.4,0.,0.,0.,.3)]:
            args=[d,b,theta,bo,ro,rough]
            row=np.array(list(map(float,subprocess.check_output([str(ROOT/'validation'/('reflected_oracle'+suffix)),*map(str,args)],text=True).split())))
            m=Map(d,reflected=True,roughness=rough);m.y=y
            value=m.flux(0.,bo,ro,xs=-np.sqrt(1-b*b)*np.sin(theta),ys=np.sqrt(1-b*b)*np.cos(theta),zs=-b)
            cases.append(dict(mode='reflected',degree=d,error=abs(value-row@y)))
        b,r,f,theta=.7,.2,.2,.4
        row=np.array(list(map(float,subprocess.check_output([str(ROOT/'validation'/('oblate_oracle'+suffix)),*map(str,[d,b,r,f,theta])],text=True).split())))
        m=Map(d,oblate=True,f=f,normalized=False);m.y=y
        value=m.flux(b*np.sin(theta),b*np.cos(theta),r)
        cases.append(dict(mode='oblate',degree=d,error=abs(value-row@y)))
    report=dict(cases=cases,passed=all(c['error']<2e-7 for c in cases))
    (ROOT/'validation'/'specialized_high_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
