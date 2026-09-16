import json,math,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def values(exe,args):return list(map(float,subprocess.check_output([str(exe),*map(str,args)],text=True).split()))
def main():
    cases=0;worst=0.;failures=[];sphere_cases=0;sphere_worst=0.;source_floor_worst=0.
    # Upstream's low-level oblate routine accesses coefficient 2 even at degree 0.
    # Compare degree >=1, whose first column still tests the uniform-map response.
    for d in [1,3,5]:
        for f in [1e-6,.1,.3]:
            for b,r,theta in [(.5,.1,.5),(.85,.2,.8),(1.,.3,.4),(.1,.1,.7),(2.,.1,.3),(.2,2.,.3)]:
                args=[d,b,r,f,theta]
                a=values(ROOT/'validation/oblate_oracle.exe',args)
                b=values(ROOT/'target/release/probe.exe',['oblate',*args])
                err=max(abs(x-y) for x,y in zip(a,b));worst=max(worst,err);cases+=1
                if not all(math.isfinite(x) and math.isfinite(y) and abs(x-y)<2e-7 for x,y in zip(a,b)):
                    failures.append({'args':args,'error':err});print('FAIL',args,err,flush=True)
        for separation,radius in [(.5,.1),(.85,.2),(1.,.3),(.1,.1),(2.,.1),(.2,2.)]:
            args=[d,separation,radius,0.,0.]
            a=values(ROOT/'validation/oracle.exe',['flux',d,separation,radius])
            b=values(ROOT/'target/release/probe.exe',['oblate',*args])
            upstream_oblate=values(ROOT/'validation/oblate_oracle.exe',args)
            err=max(abs(x-y) for x,y in zip(a,b));sphere_worst=max(sphere_worst,err);sphere_cases+=1
            source_floor_worst=max(source_floor_worst,max(abs(x-y) for x,y in zip(upstream_oblate,b)))
            if len(a)!=len(b) or not all(math.isfinite(y) and abs(x-y)<2e-10 for x,y in zip(a,b)):
                failures.append({'spherical_args':args,'error':err})
    report={'cases':cases,'max_absolute_error':worst,'spherical_cases':sphere_cases,'spherical_max_absolute_error':sphere_worst,
            'upstream_zero_flattening_floor_max_difference':source_floor_worst,
            'note':'Positive flattening compared with upstream oblate kernel. Exact f=0 compared with upstream emitted sphere; oblate reference floors f to 1e-6 (macros.h STARRY_MIN_F).',
            'failures':failures,'passed':not failures}
    (ROOT/'validation/oblate_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
