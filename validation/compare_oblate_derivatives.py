"""Check oblate geometry/flattening derivatives against pinned upstream AD."""
import json
import pathlib
import subprocess
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map


def main():
    errors=[]
    exe=ROOT/'validation'/('oblate_oracle.exe' if sys.platform=='win32' else 'oblate_oracle')
    for d in [1,3]:
        for b,r,f,theta in [(.5,.2,.2,.4),(.9,.3,.3,.7)]:
            m=Map(d,oblate=True,f=f,normalized=False)
            reference=np.array(list(map(float,subprocess.check_output([str(exe),*map(str,[d,b,r,f,theta]),'gradients'],text=True).split()))).reshape(5,m.Ny)
            xo,yo=b*np.sin(theta),b*np.cos(theta)
            for j in range(m.Ny):
                m.y=np.zeros(m.Ny);m.y[j]=1.
                geometry=m.flux_gradient(xo,yo,r);jac=m.oblate_jacobian(xo,yo,r)
                actual=np.array([geometry[0]*np.sin(theta)+geometry[1]*np.cos(theta),geometry[2],jac['f'],
                    b*(geometry[0]*np.cos(theta)-geometry[1]*np.sin(theta))])
                errors.append(float(np.max(abs(actual-reference[1:,j]))))
    report=dict(cases=len(errors),max_absolute_error=max(errors),passed=max(errors)<2e-7,
        reference='pinned oblate/occultation.h automatic derivatives; separation, radius, flattening, angle')
    (ROOT/'validation'/'oblate_derivatives_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
