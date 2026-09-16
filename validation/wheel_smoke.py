"""Run with python -I after installing the wheel into .wheel-smoke."""
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.wheel-smoke'))
import numpy as np
import starry_rust as starry

library=pathlib.Path(starry._dll._name).resolve()
assert library.is_relative_to(ROOT/'.wheel-smoke'/'_starry_native'),library
m=starry.Map(2,nw=2)
transforms=m.surface_transforms()
m.amp=2.
m.rotate([1,0,0],.4)
np.testing.assert_allclose(m.flux(),[2.,2.])
np.testing.assert_allclose(transforms['forward']@m.y[:,0],np.full(len(transforms['lat']),1/np.pi),atol=1e-12)
r=starry.Map(1,reflected=True)
assert np.isfinite(r.illumination_jacobian(xs=.3,ys=.4,zs=1.)['xs'])
o=starry.Map(1,oblate=True,f=.1)
assert np.isfinite(o.oblate_jacobian()['f'])
s=starry.System(starry.Primary(starry.Map(1)),starry.Secondary(starry.Map(amp=0.),porb=2.,r=.1))
np.testing.assert_allclose(s.flux_jacobian([.01])['flux'],s.flux([.01]))
assert np.isfinite(s.flux_jacobian([.01])['body0.m']).all()
v=starry.Map(1,rv=True,veq=10000.)
np.testing.assert_allclose(v.rv_jacobian(xo=.4,ro=.2)['rv'],v.rv(xo=.4,ro=.2),atol=1e-9)
d=starry.DopplerMap(starry.Map(1),np.ones(11),log_spacing=2e-5,half_width=2,veq=15000.)
assert np.all(np.isfinite(d.flux_jacobian()['veq']))
s.orbit_convention='starry'
np.testing.assert_allclose(s.rv_jacobian([.01],surface=True)['rv'],s.rv([.01]),atol=1e-7)
assert np.isfinite(s.flux_jacobian([.01],surface=True)['body0.map.y']).all()
assert s.render([.01],res=8)['positions'].shape==(1,2,3)
d.continuum_index=2
np.testing.assert_allclose(d.flux([0.,.2])[:,2],1.)
assert np.isfinite(d.spectrum_jacobian([0.,.2])[1]).all()
namespace={}
for block in (ROOT/'NUMERICAL_API.md').read_text(encoding='utf-8').split('```python\n')[1:]:
    exec(compile(block.split('```',1)[0],'NUMERICAL_API.md','exec'),namespace)
print(f'Installed wheel loaded {library}')
print('Spectral maps, native pixel transforms, rotation and all Python numerical guide examples passed.')
