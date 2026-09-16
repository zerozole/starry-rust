import ctypes as ct
import unittest
import numpy as np
from starry_rust import Map,DopplerMap
from _starry_api import _dll,_ptr,array,ptr,check


class NativeLoading(unittest.TestCase):
    def test_truncated_svd_against_numpy(self):
        rng=np.random.default_rng(44)
        _dll.starry_low_rank.argtypes=[_ptr,ct.c_size_t,ct.c_size_t,ct.c_size_t,_ptr]
        for shape in [(23,7),(7,23),(4,4)]:
            for rank in [1,3]:
                a=array(rng.normal(size=shape));out=np.empty(rank*sum(shape))
                check(_dll.starry_low_rank(ptr(a),*shape,rank,ptr(out)))
                scores=out[:shape[0]*rank].reshape(shape[0],rank);spectra=out[shape[0]*rank:].reshape(rank,shape[1])
                u,s,v=np.linalg.svd(a,full_matrices=False)
                np.testing.assert_allclose(scores@spectra,(u[:,:rank]*s[:rank])@v[:rank],atol=2e-11)
                np.testing.assert_allclose(spectra@spectra.T,np.eye(rank),atol=1e-12)

    def test_cube_reconstructs_surface_spectra(self):
        h,w,n=10,20,15
        m1=Map(2);m1[1,1]=.2
        m2=Map(2);m2[2,-1]=.3
        images=np.stack([m1.render(res=h,projection='rect'),m2.render(res=h,projection='rect')])
        spectra=np.stack([np.linspace(.8,1.2,n),1.-.4*np.exp(-np.linspace(-3,3,n)**2)])
        cube=np.einsum('chw,cn->hwn',images,spectra)
        model=DopplerMap([Map(2),Map(2)],np.ones((2,n)),log_spacing=2e-5,half_width=2,veq=20000.)
        model.load(cube=cube,ridge=1e-12)
        restored=np.einsum('chw,cn->hwn',np.stack([m.render(res=h,projection='rect') for m in model.maps]),model.spectra)
        np.testing.assert_allclose(restored,cube,atol=3e-11)
        reference=DopplerMap([m1,m2],spectra,log_spacing=2e-5,half_width=2,veq=20000.)
        np.testing.assert_allclose(model.flux([0.,.5],normalize=False),reference.flux([0.,.5],normalize=False),atol=3e-11)

    def test_smoothing_and_partial_extent(self):
        m=Map(2,angle_unit='deg');m[2,1]=.2
        lat=np.linspace(60,-60,12);lon=np.linspace(-110,110,23)
        image=m.intensity(lat[:,None],lon[None,:])
        result=Map(2,angle_unit='deg')
        result.load(image,extent=(-115,115,-720/11,720/11),smoothing=.3,ridge=1e-12)
        expected=m.amp*m.y.copy();expected[4:]*=np.exp(-3*.3**2)
        np.testing.assert_allclose(result.amp*result.y,expected,atol=1e-11)


if __name__=='__main__':unittest.main()
