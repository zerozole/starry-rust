"""Compare native shape derivatives with upstream's unmodified AD kernel."""
import json
import pathlib
import subprocess
import sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from starry_rust import Map


def main():
    executable=ROOT/'validation'/('reflected_oracle.exe' if sys.platform=='win32' else 'reflected_oracle')
    errors=[];illumination_errors=[]
    for degree in [0,1,3]:
        m=Map(degree,reflected=True)
        for b,theta,bo,ro,rough in [(-.5,.7,.4,.2,0.),(-.5,1.1,.6,.4,.3),(.3,.5,.7,.5,0.)]:
            reference=np.array(list(map(float,subprocess.check_output([str(executable),*map(str,[degree,b,theta,bo,ro,rough]),'gradient'],text=True).split()))).reshape(6,m.Ny)
            m.roughness=rough
            source=dict(xs=-np.sqrt(1-b*b)*np.sin(theta),ys=np.sqrt(1-b*b)*np.cos(theta),zs=-b)
            for j in range(m.Ny):
                m.y=np.zeros(m.Ny);m.y[j]=1.
                gradient=m.flux_gradient(0.,bo,ro,**source)
                error=float(np.max(np.abs(gradient[1:]-reference[3:5,j])))
                errors.append(error)
                jac=m.illumination_jacobian(0.,bo,ro,**source)
                gradient=np.array([jac['xs'],jac['ys'],jac['zs']])
                bc=np.sqrt(1-b*b)
                db=np.array([b/bc*np.sin(theta),-b/bc*np.cos(theta),-1.])
                dt=np.array([-bc*np.cos(theta),-bc*np.sin(theta),0.])
                actual=np.array([gradient@db,gradient@dt,jac['roughness']])
                illumination_errors.append(float(np.max(abs(actual-reference[[1,2,5],j]))))
    report=dict(cases=len(errors),max_absolute_error=max(errors),illumination_cases=len(illumination_errors),
                illumination_max_absolute_error=max(illumination_errors),
                passed=max(errors)<2e-7 and max(illumination_errors)<2e-7,
                reference='pinned upstream reflected/occultation.h AD; separation, radius, phase geometry and roughness')
    (ROOT/'validation'/'specialized_gradients_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
