"""Unmodified upstream OpsRV filter, differentiated by complex-step evaluation.

The extracted method uses NumPy as the eager backend. This compares the native
filter and its four physical derivatives; full RV ratios are checked separately.
"""
import ast
import json
import pathlib
import subprocess
import sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
source=ROOT.parent/'starry-upstream/starry/_core/core.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OpsRV')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_rv_filter')
method.decorator_list=[]
namespace={'tt':np,'np':np}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
reference=namespace['compute_rv_filter']
probe=ROOT/'target/release'/('probe.exe' if sys.platform=='win32' else 'probe')
errors=[]
for inc in [0.,.01,1.1,np.pi/2,np.pi]:
    for obl in [0.,.3,2.]:
        for veq,alpha in [(0.,0.),(10000.,.2),(20000.,-.1),(3000.,1.)]:
            args=[inc,obl,veq,alpha]
            expected=[reference(None,*args)]
            for j in range(4):
                shifted=np.array(args,dtype=complex);shifted[j]+=1e-25j
                expected.append(reference(None,*shifted).imag/1e-25)
            actual=np.fromstring(subprocess.check_output([str(probe),'rv-filter-jacobian','3',*map(str,args)],text=True),sep=' ').reshape(5,16)
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=3e-13)
            errors.append(float(np.max(abs(actual-expected))))
report=dict(cases=len(errors),max_absolute_error=max(errors),passed=True,
    reference='upstream core.py OpsRV.compute_rv_filter, unmodified method body, complex-step derivatives')
(ROOT/'validation/rv_derivatives_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
