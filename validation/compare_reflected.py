import json
import math
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
def run(exe,args):
    return list(map(float,subprocess.check_output([str(exe),*map(str,args)],text=True).split()))

def main():
    cases=0
    maximum=0.
    failures=[]
    for d in [0,1,3]:
        for b in [-.99,-.5,0.,.5,.99]:
            for rough in [0.,.3]:
                for theta,bo,ro in [(0.,0.,0.),(0.,.4,.2),(.7,.4,.7),(1.8,.7,.5)]:
                    args=[d,b,theta,bo,ro,rough]
                    a=run(ROOT/'validation/reflected_oracle.exe',args)
                    v=run(ROOT/'target/release/probe.exe',['reflected',*args])
                    assert len(a)==len(v)
                    error=max(abs(x-y) for x,y in zip(a,v))
                    maximum=max(maximum,error)
                    if not all(math.isfinite(x) and math.isfinite(y) and abs(x-y)<2e-8 for x,y in zip(a,v)):
                        failures.append({'args':args,'error':error})
                        print('FAIL',args,error,flush=True)
                    cases+=1
    report={'cases':cases,'max_absolute_error':maximum,'failures':failures,'passed':not failures}
    (ROOT/'validation/reflected_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
