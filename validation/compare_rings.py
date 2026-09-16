"""High-degree harmonic-ring flux against pinned upstream circular kernels."""
import json
import math
from decimal import localcontext
from decimal_rings import integral
from compare import ROOT,ORACLE,RUST,values

cases=[]
for degree in [5,10,15,20]:
    for b,r in [(0.,.4),(.3,.1),(.7,.4),(1.2,.5),(.5,.5),(.01,1.),(2.,.2),(1e-6,.1)]:
        oracle=values(ORACLE,'flux',degree,b,r)
        stable=values(RUST,'rings',degree,b,r)
        old=values(RUST,'analytic',degree,b,r)
        error=max(abs(a-b) for a,b in zip(oracle,stable))
        index=max(range(len(stable)),key=lambda i:abs(oracle[i]-stable[i]))
        l=math.isqrt(index);m=index-l*l-l
        checks=[]
        for i,(a,z) in enumerate(zip(oracle,stable)):
            if abs(a-z)<=2e-8:continue
            ll=math.isqrt(i);mm=i-ll*ll-ll
            with localcontext() as context:
                context.prec=50
                high=integral(ll,mm,b,r,128);low=integral(ll,mm,b,r,64)
            checks.append(dict(index=i,decimal_reference=str(high),
                rust_difference=abs(z-float(high)),upstream_difference=abs(a-float(high)),
                quadrature_difference=float(abs(high-low)),
                passed=abs(z-float(high))<1e-11 and abs(high-low)<1e-20))
        cases.append(dict(degree=degree,b=b,r=r,error=error,
            worst_degree=l,worst_order=m,
            old_green_difference=max(abs(a-b) for a,b in zip(old,stable)),
            upstream_agrees=error<=2e-8,high_precision_checks=checks,
            passed=all(math.isfinite(v) for v in stable) and (error<=2e-8 or all(c['passed'] for c in checks))))
report=dict(cases=cases,max_absolute_error=max(c['error'] for c in cases),
    note='Original upstream discrepancy retained. Every response outside the unchanged 2e-8 comparison tolerance is checked separately with 50-digit quadrature.',
    passed=all(c['passed'] for c in cases))
(ROOT/'validation'/'rings_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
