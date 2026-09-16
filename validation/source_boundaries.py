"""Record source-defined derivative/API boundaries without importing Theano."""
import ast
import json
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
UPSTREAM=ROOT.parent/'starry-upstream/starry'
entries=[]
for filename in ['integration.py','rotation.py','filter.py','polybasis.py','spot.py']:
    tree=ast.parse((UPSTREAM/'_core/ops'/filename).read_text())
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and 'GradientOp' in cls.name:
            methods=[n.name for n in cls.body if isinstance(n,ast.FunctionDef)]
            entries.append(dict(source=f'_core/ops/{filename}:{cls.lineno}',operator=cls.name,defines_further_grad='grad' in methods))
assert entries and not any(e['defines_further_grad'] for e in entries)
limb=(UPSTREAM/'_core/ops/limbdark/limbdark.py').read_text()
assert "can't propagate gradients wrt parameter" in limb
kepler=(UPSTREAM/'kepler.py').read_text()
assert 'for either all or none of the bodies.' in kepler
assert 'Oblate secondary bodies are not currently supported.' in kepler
maps=(UPSTREAM/'maps.py').read_text()
assert 'Combinations of `rv`, `reflected`, and `oblate` not yet implemented.' in maps
report=dict(gradient_operators=entries,limb_derivative_outputs_reject_further_gradients=True,
    all_body_rv_flags_must_match=True,specialized_map_combinations_rejected=True,oblate_secondaries_rejected=True,
    conclusion='No blanket arbitrary-flux-Hessian requirement; mixed oblate-primary/RV-secondary is excluded by upstream.',passed=True)
(ROOT/'validation/source_boundaries_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
