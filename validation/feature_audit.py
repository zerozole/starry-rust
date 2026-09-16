"""Inventory public upstream methods without importing upstream dependencies.

This is a coverage checklist, not evidence of numerical equivalence. A mapped
method still needs reference tests and its supported-domain limits checked.
"""
import ast
import json
import pathlib
from operator_evidence import operator_evidence,property_counterpart

ROOT=pathlib.Path(__file__).resolve().parents[1]
UPSTREAM=ROOT.parent/'starry-upstream'/'starry'
MAPPING={
    'flux':('implemented','map/reflected/oblate/system/doppler Rust modules'),
    'intensity':('implemented','map/imaging Rust modules; Doppler component Map access'),
    'render':('implemented','imaging Rust module; Doppler component Map access'),
    'rotate':('implemented','rotation.rs'),
    'spot':('implemented','surface::add_spot'),
    'design_matrix':('implemented','Map/System/Doppler designs; unrestricted Doppler matrix-free operator'),
    'intensity_design_matrix':('implemented','Map.intensity_design_matrix'),
    'set_data':('implemented','new Python inference inputs'),
    'set_prior':('implemented','native Gaussian inference'),
    'remove_prior':('implemented','new Python inference state'),
    'solve':('implemented','native Gaussian solves; Doppler conditional, bilinear and joint nonlinear fits'),
    'lnlike':('implemented','inference.rs marginal likelihood'),
    'draw':('implemented','posterior Gaussian draws'),
    'rv':('implemented','rv.rs/system.rs'),
    'position':('implemented','orbit.rs/system.rs'),
    'minimize':('implemented','bounded multistart surface search; scalar intrinsic maps'),
    'get_pixel_transforms':('implemented','pixels.rs compact forward/inverse/derivative operators'),
    'sht_matrix':('implemented','native pixel transforms and Gaussian smoothing weights'),
    'get_latlon_grid':('implemented','imaging::coordinate_grid and Map.get_latlon_grid'),
    'load':('implemented','native sample fitting, smoothing and cube SVD; numeric arrays with documented new grid convention'),
    'baseline':('implemented','Doppler baseline'),
    'dot':('implemented','Doppler fixed-map/fixed-spectrum and unrestricted forward/transpose'),
    'limbdark_is_physical':('implemented','surface::limb_is_physical'),
    'optimize':('out_of_api_scope','upstream PyMC3-context wrapper; numerical optimizer coverage reviewed separately'),
    'show':('out_of_api_scope','plotting convenience; numerical render outputs supplied'),
    'visualize':('out_of_api_scope','interactive plotting convenience'),
    'reset':('state_api','construct a new map or assign explicit coefficients'),
}

def main():
    entries=[]
    properties=[]
    for filename in ['maps.py','doppler.py','kepler.py']:
        tree=ast.parse((UPSTREAM/filename).read_text(encoding='utf-8'))
        for cls in tree.body:
            if not isinstance(cls,ast.ClassDef):continue
            for node in cls.body:
                if not isinstance(node,ast.FunctionDef) or node.name.startswith('_'):continue
                decorators=[ast.unparse(d) for d in node.decorator_list]
                if any(d=='property' for d in decorators):
                    properties.append(dict(source=f'{filename}:{node.lineno}',property=f'{cls.name}.{node.name}',counterpart=property_counterpart(filename,node.name)))
                if any(d=='property' or d.endswith('.setter') for d in decorators):continue
                status,equivalent=MAPPING.get(node.name,('unreviewed',''))
                if cls.name=='DopplerMap' and node.name=='solve':
                    status,equivalent=('implemented','L2/L1 spectra, conditional/bilinear/nonlinear fits; source temperature schedule and uniform-kernel L1 deconvolution initialization')
                if cls.name=='System' and node.name=='solve':
                    status,equivalent=('implemented','linear Gaussian map solves at fixed design; source does not supply a joint bilinear primary/reflected posterior')
                entries.append(dict(source=f'{filename}:{node.lineno}',method=f'{cls.name}.{node.name}',
                                    status=status,equivalent=equivalent))
    counts={s:sum(e['status']==s for e in entries) for s in sorted({e['status'] for e in entries})}
    internal=[]
    evidence=operator_evidence(ROOT)
    counterparts={
        'OpsYlm':'basis/map/rotation/surface/pixels/imaging native modules',
        'OpsLD':'map limb filters and emitted integration',
        'OpsRV':'rv native filters and velocity ratios',
        'OpsReflected':'reflected polynomial scattering and harmonic surface quadrature',
        'OpsOblate':'oblate gravity filters, conic geometry and harmonic surface quadrature',
        'OpsDoppler':'doppler native designs, matrix-free products, convolutions and inference::l1',
        'OpsSystem':'orbit/system native geometry; System.render composes native per-body images',
    }
    tree=ast.parse((UPSTREAM/'_core'/'core.py').read_text(encoding='utf-8'))
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef):continue
        for node in cls.body:
            if not isinstance(node,ast.FunctionDef) or node.name.startswith('_'):continue
            internal.append(dict(source=f'_core/core.py:{node.lineno}',operator=f'{cls.name}.{node.name}',
                status='numerical_counterpart' if cls.name in counterparts else 'unreviewed',
                equivalent=counterparts.get(cls.name,''),
                derivative_status='native first derivatives; see operator evidence and NUMERICAL_LIMITS.md'))
            key=f'{cls.name}.{node.name}'
            if key not in evidence:raise ValueError('missing operator evidence '+key)
            internal[-1].update(evidence[key])
            internal[-1]['derivative_status']='native first derivatives and System surface/RV composition; boundary contract in NUMERICAL_LIMITS.md'
    payload=dict(scope='Numerical Rust functionality plus new Python API; no Theano compatibility',
        evidence_warning='Implemented means a code path exists, not full validation or completeness.',
        counts=counts,methods=entries,properties=properties,internal_operators=internal,
        cross_cutting=['degree-domain parity and conditioning','native parameter derivatives',
                       'independent numerical references','Python API documentation',
                       'runtime platform validation'])
    (ROOT/'validation'/'feature_audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts))
    for entry in entries:
        if entry['status'] in ('unreviewed','missing','partial'):print(entry['method'],entry['status'],entry['equivalent'])

if __name__=='__main__':main()
