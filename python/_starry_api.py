"""Array-oriented Python interface. Numerical map operations execute in Rust."""
import ctypes as ct
import math
import weakref
import numpy as np
from starry_rust import _dll, _ptr

_u64=ct.c_uint64
_size=ct.c_size_t
_dll.starry_map_create.argtypes=[ct.c_uint32]
_dll.starry_map_create.restype=_u64
_dll.starry_map_destroy.argtypes=[_u64]
_dll.starry_map_destroy.restype=None
_dll.starry_last_error.argtypes=[ct.POINTER(ct.c_uint8),_size]
_dll.starry_last_error.restype=_size
_dll.starry_map_update.argtypes=[_u64,_ptr,_size,ct.c_double]
_dll.starry_map_batch.argtypes=[_u64,ct.c_uint32,_ptr,_size,_size,_ptr]
_dll.starry_map_gradients.argtypes=[_u64,_ptr,_size,_size,_ptr]
_dll.starry_map_jacobian.argtypes=[_u64,_ptr,_size,_ptr,_size]
_dll.starry_map_transform.argtypes=[_u64,ct.c_uint32,_ptr,_size,_ptr,_size]
_dll.starry_map_intensity_batch.argtypes=[_u64,_ptr,_size,ct.c_uint32,_ptr,ct.c_double,_ptr]
_dll.starry_system_batch.argtypes=[ct.POINTER(_u64),_size,_ptr,_ptr,_ptr,_size,ct.c_uint32,ct.c_double,_size,ct.c_uint32,_ptr]
_dll.starry_system_batch_v2.argtypes=[ct.POINTER(_u64),_size,_ptr,_ptr,_ptr,_size,ct.c_uint32,_ptr,_size,ct.c_uint32,ct.c_uint32,_ptr]
_dll.starry_linear_solve.argtypes=[_ptr,_size,_size,_ptr,_ptr,_ptr,_ptr,_ptr]
_dll.starry_map_render.argtypes=[_u64,_size,_size,ct.c_uint32,ct.c_double,ct.c_double,ct.c_double,_ptr]
_dll.starry_map_image.argtypes=[_u64,ct.c_uint32,_ptr,_size,_size,_size,ct.c_uint32,ct.c_uint32,_ptr]
_dll.starry_map_surface.argtypes=[_u64,ct.c_uint32,_ptr,_size,_ptr,_size,ct.c_uint32,_ptr]
_dll.starry_surface_operation.argtypes=[_u64,ct.c_uint32,_ptr,_size,_ptr,_size]
_dll.starry_pixel_grid.argtypes=[ct.c_uint32,_size,_ptr,_size]
_dll.starry_pixel_grid.restype=_size
_dll.starry_pixel_transforms.argtypes=[_u64,_ptr,_size,ct.c_double,_ptr,_size]
_dll.starry_render_grid.argtypes=[_ptr,_size,_size,ct.c_uint32,ct.c_uint32,_ptr]
_dll.starry_map_fit.argtypes=[_u64,_ptr,_size,ct.c_double,_ptr]
_dll.starry_doppler.argtypes=[_u64,ct.c_double,ct.c_double,ct.c_double,_ptr,_size,ct.c_double,_size,_ptr,_size,ct.c_uint32,_ptr]

def array(a):
    value=np.asarray(a,dtype=np.float64)
    return value if value.flags.c_contiguous else np.ascontiguousarray(value)
def ptr(a):return a.ctypes.data_as(_ptr)
def check(status):
    if status:
        buffer=(ct.c_uint8*4096)()
        _dll.starry_last_error(buffer,len(buffer))
        message=bytes(buffer).split(b'\0')[0].decode('utf-8',errors='replace')
        raise ValueError(message or 'invalid native arguments')

def covariance(value,n):
    a=array(value)
    if a.ndim==0:return np.eye(n)*float(a)
    if a.shape==(n,):return np.diag(a)
    if a.shape!=(n,n):raise ValueError('covariance has wrong shape')
    return a

class Map:
    """Harmonic map with native cached transforms. Angles default to radians.

    Set angle_unit='deg' for degree-valued rotation and viewing parameters.
    This interface is explicit about unsupported keyword arguments.
    """
    def __init__(self,ydeg=0,udeg=0,nw=None,*,amp=1.,reflected=False,rv=False,oblate=False,
                 inc=None,obl=0.,angle_unit='rad',roughness=0.,veq=0.,alpha=0.,
                 f=0.,omega=0.,beta=.23,tpole=6000.,wav=550e-9,fdeg=0,normalized=True,source_npts=1):
        if isinstance(ydeg,bool) or not isinstance(ydeg,int) or not 0<=ydeg<=32:raise ValueError('ydeg must be an integer in [0,32]')
        if not isinstance(udeg,int) or not 0<=udeg<=20 or ydeg+udeg>32:raise ValueError('invalid limb degree')
        if nw is not None and (not isinstance(nw,int) or nw<1):raise ValueError('invalid wavelength count')
        if sum(map(bool,[reflected,rv,oblate]))>1:raise ValueError('choose one specialized map mode')
        self.ydeg=ydeg;self.udeg=udeg;self.nw=nw;self.Ny=(ydeg+1)**2
        self.angle_unit=angle_unit;self.inc=(math.pi/2/self._angle) if inc is None else inc;self.obl=obl
        self.reflected=bool(reflected);self._rv=bool(rv);self.oblate=bool(oblate)
        self.roughness=roughness;self.veq=veq;self.alpha=alpha;self.f=f;self.omega=omega
        self.beta=beta;self.tpole=tpole;self.wav=wav;self.fdeg=fdeg;self.normalized=normalized
        self.source_npts=source_npts
        self.y=np.zeros((self.Ny,) if nw is None else (self.Ny,nw));self.y[0]=1.
        self.amp=float(amp) if nw is None else np.broadcast_to(amp,(nw,)).copy()
        self.u=np.zeros(udeg+1);self.u[0]=-1.
        self._handles=[int(_dll.starry_map_create(ydeg)) for _ in range(nw or 1)]
        if not all(self._handles):
            for h in self._handles:_dll.starry_map_destroy(h)
            raise ValueError('failed to create native map')
        self._finalizer=weakref.finalize(self,lambda hs:[_dll.starry_map_destroy(h) for h in hs],self._handles.copy())
        self._prior=None;self._data=None;self._solution=None

    @property
    def amp(self):return self._amp
    @amp.setter
    def amp(self,value):
        amplitude=array(value)
        if np.any(~np.isfinite(amplitude)):raise ValueError('amplitude must be finite')
        self._amp=float(amplitude) if self.nw is None else np.broadcast_to(amplitude,(self.nw,)).copy()

    @property
    def angle_unit(self):return self._angle_unit
    @angle_unit.setter
    def angle_unit(self,value):
        name=str(value).lower()
        if name in ('rad','radian','radians'):self._angle=1.
        elif name in ('deg','degree','degrees'):self._angle=math.pi/180.
        else:raise ValueError('angle_unit must be rad or deg')
        self._angle_unit=value

    def _sync(self):
        y=array(self.y);expected=(self.Ny,) if self.nw is None else (self.Ny,self.nw)
        if y.shape!=expected:raise ValueError('invalid map coefficient shape')
        amps=np.broadcast_to(self.amp,(self.nw or 1,))
        for i,h in enumerate(self._handles):
            col=array(y if self.nw is None else y[:,i]);check(_dll.starry_map_update(h,ptr(col),self.Ny,float(amps[i])))
        return y

    def __getitem__(self,key):
        if isinstance(key,tuple):
            index=self._indices(key)
            return self.y[index] if len(key)==2 else self.y[index,key[2]]
        return self.u[key]

    def _indices(self,key):
        if len(key) not in (2,3):raise IndexError('use degree, order, and optional wavelength')
        if len(key)==3 and self.nw is None:raise IndexError('map is not spectral')
        l,m=key[:2]
        if isinstance(l,(int,np.integer)) and isinstance(m,(int,np.integer)):
            if not 0<=l<=self.ydeg or abs(m)>l:raise IndexError('invalid harmonic index')
            return l*l+l+m
        degrees=list(range(self.ydeg+1))[l] if isinstance(l,slice) else [l]
        indices=[]
        for degree in degrees:
            if not 0<=degree<=self.ydeg:raise IndexError('invalid harmonic degree')
            orders=list(range(-degree,degree+1))[m] if isinstance(m,slice) else [m]
            indices.extend(degree*degree+degree+order for order in orders if abs(order)<=degree)
        return np.array(indices,dtype=int)

    def __setitem__(self,key,value):
        if isinstance(key,tuple):
            index=self._indices(key)
            if len(key)==2:self.y[index]=value
            else:self.y[index,key[2]]=value
        else:
            indices=np.arange(len(self.u))[key]
            if np.any(indices==0):raise ValueError('u[0] is fixed at -1')
            self.u[key]=value

    def _parameters(self,theta,xo,yo,zo,ro,xs,ys,zs):
        arrays=np.broadcast_arrays(*[array(v) for v in (theta,xo,yo,zo,ro,xs,ys,zs)])
        shape=arrays[0].shape;n=arrays[0].size
        p=np.zeros((n,22+self.udeg));p[:,21]=self.source_npts
        for column,value in zip([0,3,4,5,6,7,8,9],arrays):p[:,column]=value.ravel()
        p[:,0]*=self._angle;p[:,1]=self.inc*self._angle;p[:,2]=self.obl*self._angle
        p[:,10]=self.roughness*self._angle;p[:,11]=self.veq;p[:,12]=self.alpha;p[:,13]=self.f
        p[:,14]=self.omega;p[:,15]=self.beta;p[:,16]=self.tpole;p[:,18]=self.fdeg;p[:,19]=self.normalized
        if self.udeg:p[:,22:]=self.u[1:]
        return p,shape

    def flux(self,xo=0.,yo=0.,ro=0.,*,theta=0.,zo=1.,xs=0.,ys=0.,zs=1.,Rs=0.,integrated=False):
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,xs,ys,zs)
        p[:,20]=np.broadcast_to(Rs,shape).ravel()
        mode=1 if self.reflected else 3 if self.oblate else 0
        values=[]
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        for i,h in enumerate(self._handles):
            p[:,17]=wavelengths[i];out=np.empty(len(p));check(_dll.starry_map_batch(h,mode,ptr(p),len(p),p.shape[1],ptr(out)));values.append(out)
        result=np.stack(values,axis=-1)
        if self.nw is None:result=result[:,0].reshape(shape)
        else:
            result=result.reshape((*shape,self.nw))
            if integrated:result=result.sum(axis=-1)
        return result.item() if np.ndim(result)==0 else result

    def rv(self,*,theta=0.,xo=0.,yo=0.,zo=1.,ro=0.):
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,0.,0.,1.);channels=[]
        for h in self._handles:
            out=np.empty(len(p));check(_dll.starry_map_batch(h,2,ptr(p),len(p),p.shape[1],ptr(out)));channels.append(out)
        result=channels[0].reshape(shape) if self.nw is None else np.stack(channels,axis=-1).reshape((*shape,self.nw))
        return result.item() if result.ndim==0 else result

    def rv_jacobian(self,*,theta=0.,xo=0.,yo=0.,zo=1.,ro=0.):
        """Native RV derivatives at fixed occultor foreground ordering.

        Amplitude cancels in RV. Zero visible flux follows rv's zero branch;
        no derivative across a total-eclipse boundary is implied.
        """
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,0.,0.,1.)
        count=10+self.Ny+self.udeg;channels=[]
        _dll.starry_specialized_jacobian.argtypes=[_u64,ct.c_uint32,_ptr,_size,_ptr,_size]
        for h in self._handles:
            values=[]
            for row in p:
                out=np.empty(count)
                check(_dll.starry_specialized_jacobian(h,2,ptr(row),len(row),ptr(out),count));values.append(out)
            channels.append(np.array(values))
        result=channels[0].reshape((*shape,count)) if self.nw is None else np.stack(channels,axis=-2).reshape((*shape,self.nw,count))
        result[...,1:4]*=self._angle
        output={name:result[...,i] for i,name in enumerate(('rv','theta','inc','obl','xo','yo','ro','amp'))}
        output['y']=result[...,8:8+self.Ny];output['u']=result[...,8+self.Ny:8+self.Ny+self.udeg]
        output['veq']=result[...,-2];output['alpha']=result[...,-1]
        return output

    def intensity(self,lat=0.,lon=0.,*,xs=0.,ys=0.,zs=1.,Rs=0.,illuminate=True,mu=None,limbdarken=True,rv=True,on94_exact=False,theta=0.):
        if mu is not None:
            if self.ydeg!=0:raise ValueError('mu-only intensity requires a limb-only map')
            mu=array(mu)
            if np.any(~np.isfinite(mu)) or np.any((mu<0)|(mu>1)):raise ValueError('invalid mu')
            return 1.-sum(self.u[n]*(1.-mu)**n for n in range(1,self.udeg+1))
        self._sync();lat,lon=np.broadcast_arrays(array(lat)*self._angle,array(lon)*self._angle);shape=lat.shape
        if np.any(np.abs(lat)>math.pi/2):raise ValueError('invalid latitude')
        xyz=array(np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],axis=-1).reshape(-1,3))
        p,_=self._parameters(float(theta),0.,0.,1.,0.,float(xs),float(ys),float(zs));p[:,20]=Rs
        mode=1 if self.reflected else 3 if self.oblate else 2 if self._rv else 0;values=[]
        flags=int(illuminate)+2*int(on94_exact)+4*int(limbdarken)+8*int(rv)
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        for i,h in enumerate(self._handles):
            p[:,17]=wavelengths[i]
            out=np.empty(len(xyz));check(_dll.starry_map_surface(h,mode,ptr(p),p.shape[1],ptr(xyz),len(xyz),flags,ptr(out)));values.append(out)
        result=np.stack(values,axis=-1)
        if self.nw is None:result=result[:,0].reshape(shape)
        else:result=result.reshape((*shape,self.nw))
        return result.item() if result.ndim==0 else result

    def _transform(self,operation,args):
        self._sync();args=array(args);out=np.empty(self.Ny+1)
        for i,h in enumerate(self._handles):
            check(_dll.starry_map_transform(h,operation,ptr(args),len(args),ptr(out),len(out)))
            if self.nw is None:self.amp=out[0];self.y=out[1:].copy()
            else:self.amp[i]=out[0];self.y[:,i]=out[1:]

    def rotate(self,axis,theta):
        if len(axis)!=3:raise ValueError('axis needs three components')
        self._transform(0,[*axis,theta*self._angle])
    def spot(self,contrast=1.,radius=None,lat=0.,lon=0.,*,spot_smoothing=None):
        radius=math.pi/9 if radius is None else radius*self._angle
        self._transform(1,[contrast,radius,lat*self._angle,lon*self._angle,-1. if spot_smoothing is None else spot_smoothing])

    def render(self,res=300,projection='ortho',theta=0.,*,xs=0.,ys=0.,zs=1.,Rs=0.,illuminate=True,rv=True,on94_exact=False,_upstream_grid=False):
        projections={'ortho':0,'orthographic':0,'rect':1,'rectangular':1,'moll':2,'mollweide':2}
        if projection not in projections:raise ValueError('unknown projection')
        if not isinstance(res,int) or not 2<=res<=4000:raise ValueError('invalid resolution')
        self._sync();phases=np.atleast_1d(theta);frames=[]
        if phases.ndim!=1:raise ValueError('theta must be scalar or one dimensional')
        width=res if projections[projection]==0 or _upstream_grid else 2*res
        mode=1 if self.reflected else 3 if self.oblate else 2 if self._rv else 0
        flags=int(illuminate)+2*int(on94_exact)+4+8*int(rv)+16*int(_upstream_grid)
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        for phase in phases:
            channels=[]
            p,_=self._parameters(phase,0.,0.,1.,0.,float(xs),float(ys),float(zs));p[:,20]=Rs
            for i,h in enumerate(self._handles):
                p[:,17]=wavelengths[i]
                out=np.empty((res,width));check(_dll.starry_map_image(h,mode,ptr(p),p.shape[1],width,res,projections[projection],flags,ptr(out)));channels.append(out)
            frames.append(channels[0] if self.nw is None else np.stack(channels,axis=-1))
        return frames[0] if np.ndim(theta)==0 else np.stack(frames)

    def load_samples(self,lat,lon,intensity,*,weights=1.,ridge=1e-9):
        lat,lon,weights=np.broadcast_arrays(array(lat),array(lon),array(weights));values=array(intensity)
        expected=lat.shape if self.nw is None else (*lat.shape,self.nw)
        if values.shape!=expected:raise ValueError('sample intensity shape mismatch')
        self._sync();samples=np.empty((lat.size,4));samples[:,0]=lat.ravel()*self._angle;samples[:,1]=lon.ravel()*self._angle;samples[:,3]=weights.ravel()
        for i,h in enumerate(self._handles):
            samples[:,2]=(values if self.nw is None else values[...,i]).ravel();out=np.empty(self.Ny)
            check(_dll.starry_map_fit(h,ptr(samples),len(samples),ridge,ptr(out)))
            amplitude=out[0] if out[0]!=0 else 1.
            if self.nw is None:self.amp=amplitude;self.y=out/amplitude
            else:self.amp[i]=amplitude;self.y[:,i]=out/amplitude

    def get_latlon_grid(self,res=300,projection='ortho',theta=0.,*,_upstream_grid=False):
        """Intrinsic coordinates of render pixels, in angle_unit; NaN off disk."""
        projections={'ortho':0,'orthographic':0,'rect':1,'rectangular':1,'moll':2,'mollweide':2}
        if projection not in projections:raise ValueError('unknown projection')
        if not isinstance(res,int) or not 2<=res<=4000:raise ValueError('invalid resolution')
        width=res if projections[projection]==0 or _upstream_grid else 2*res
        p=array([self.inc*self._angle,self.obl*self._angle,theta*self._angle,self.f if self.oblate else 0.])
        out=np.empty((res,width,2))
        check(_dll.starry_render_grid(ptr(p),width,res,projections[projection],_upstream_grid,ptr(out)))
        return out[:,:,0]/self._angle,out[:,:,1]/self._angle

    def sht_matrix(self,*,inverse=False,return_grid=False,smoothing=None,oversample=2,lam=1e-6):
        """Smoothed pixel-to-harmonic transform, or harmonic-to-pixel if inverse."""
        t=self.surface_transforms(oversample=oversample,ridge=lam)
        result=t['forward'].copy() if inverse else t['inverse'].copy()
        if not inverse:
            strength=0. if smoothing is None and self.ydeg==0 else 2./self.ydeg if smoothing is None else smoothing
            parameters=array([strength]);weights=np.empty(self.Ny)
            check(_dll.starry_surface_operation(self._handles[0],3,ptr(parameters),1,ptr(weights),self.Ny))
            result*=weights[:,None]
        return (result,np.column_stack((t['lat'],t['lon']))) if return_grid else result

    def pixel_grid(self,oversample=2):
        """Equal-area Mollweide sample coordinates in angle_unit."""
        if not isinstance(oversample,int) or not 1<=oversample<=10:
            raise ValueError('oversample must be an integer from 1 to 10')
        n=_dll.starry_pixel_grid(self.ydeg,oversample,None,0)
        if not n:raise ValueError('invalid pixel grid')
        points=np.empty((n,2))
        if _dll.starry_pixel_grid(self.ydeg,oversample,ptr(points),points.size)!=n:
            raise ValueError('pixel grid generation failed')
        return points[:,0]/self._angle,points[:,1]/self._angle

    def surface_transforms(self,lat=None,lon=None,*,oversample=2,ridge=1e-6):
        """Native harmonic/pixel transforms of the intrinsic map.

        forward maps absolute harmonic weights (amp*y) to intensity; inverse
        is its ridge-regularized inverse. dlat/dlon act on harmonic weights,
        in angle_unit. For pixel derivatives use dlat @ (inverse @ pixels).
        Reflected maps return albedo units. Viewing and surface filters are
        excluded, as these operators describe the intrinsic coefficient field.
        """
        if (lat is None)!=(lon is None):raise ValueError('supply both lat and lon')
        if lat is None:lat,lon=self.pixel_grid(oversample)
        lat,lon=np.broadcast_arrays(array(lat),array(lon))
        points=array(np.column_stack((lat.ravel(),lon.ravel()))*self._angle)
        n=len(points);size=n*self.Ny;out=np.empty(4*size)
        check(_dll.starry_pixel_transforms(self._handles[0],ptr(points),n,ridge,ptr(out),out.size))
        forward=out[:size].reshape(n,self.Ny)
        inverse=out[size:2*size].reshape(self.Ny,n)
        dlat=out[2*size:3*size].reshape(n,self.Ny)*self._angle
        dlon=out[3*size:].reshape(n,self.Ny)*self._angle
        if self.reflected:
            forward=forward*math.pi;inverse=inverse/math.pi;dlat=dlat*math.pi;dlon=dlon*math.pi
        return dict(lat=lat.ravel().copy(),lon=lon.ravel().copy(),forward=forward,
                    inverse=inverse,dlat=dlat,dlon=dlon)

    def load(self,image,*,ridge=1e-9,smoothing=0.,extent=None,force_positive=False):
        """Fit a north-at-top equirectangular intensity array (radians internally)."""
        image=array(image)
        if image.ndim!=(2 if self.nw is None else 3):raise ValueError('expected equirectangular intensity array')
        h,w=image.shape[:2]
        if not h or not w or np.any(~np.isfinite(image)):raise ValueError('invalid image')
        bounds=np.array([-math.pi,math.pi,-math.pi/2,math.pi/2]) if extent is None else array(extent)*self._angle
        if bounds.shape!=(4,) or not np.all(np.isfinite(bounds)) or not (-math.pi/2<=bounds[2]<bounds[3]<=math.pi/2) or not bounds[0]<bounds[1]:raise ValueError('invalid image extent')
        if not np.isfinite(smoothing) or smoothing<0:raise ValueError('invalid smoothing')
        if force_positive and self.nw is not None:raise ValueError('positivity adjustment requires a scalar map')
        lat=bounds[3]-(bounds[3]-bounds[2])*(np.arange(h)+.5)/h;lon=bounds[0]+(bounds[1]-bounds[0])*(np.arange(w)+.5)/w
        lon,lat=np.meshgrid(lon,lat)
        self.load_samples(lat/self._angle,lon/self._angle,image,weights=np.cos(lat),ridge=ridge)
        parameters=array([smoothing]);weights=np.empty(self.Ny)
        check(_dll.starry_surface_operation(self._handles[0],3,ptr(parameters),1,ptr(weights),self.Ny))
        self.y*=weights if self.nw is None else weights[:,None]
        if force_positive:
            minimum=self.minimize()[2]
            if minimum<0:
                monopole=float(self.amp*self.y[0])*(1. if self.reflected else 1./math.pi)
                if monopole<=0:raise ValueError('positivity adjustment requires a positive mean')
                self.y[1:]*=monopole/(monopole-minimum)

    def minimize(self,*,bounds=None,oversample=2,ntries=5,return_info=False):
        if self.nw is not None:raise ValueError('minimization requires a scalar map')
        self._sync();bounds=[[-math.pi/2,math.pi/2],[-math.pi,math.pi]] if bounds is None else array(bounds)*self._angle
        parameters=array([*np.asarray(bounds).ravel(),oversample,ntries]);out=np.empty(5)
        check(_dll.starry_surface_operation(self._handles[0],0,ptr(parameters),len(parameters),ptr(out),len(out)))
        result=(out[0]/self._angle,out[1]/self._angle,out[2]*(math.pi if self.reflected else 1.))
        return (*result,{'evaluations':int(out[3]),'converged':bool(out[4])}) if return_info else result

    def limbdark_is_physical(self):
        u=array(self.u[1:]);out=np.empty(1)
        check(_dll.starry_surface_operation(self._handles[0],1,ptr(u),len(u),ptr(out),1))
        return bool(out[0])

    def flux_gradient(self,xo=0.,yo=0.,ro=0.,*,theta=0.,zo=1.,xs=0.,ys=0.,zs=1.,Rs=0.):
        """Native shape derivatives (xo, yo, ro), including specialized fields."""
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,xs,ys,zs);values=[]
        p[:,20]=np.broadcast_to(Rs,shape).ravel()
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        for i,h in enumerate(self._handles):
            p[:,17]=wavelengths[i];out=np.empty((len(p),3))
            if self.reflected or self.oblate:
                _dll.starry_specialized_gradients.argtypes=[_u64,ct.c_uint32,_ptr,_size,_size,_ptr]
                check(_dll.starry_specialized_gradients(h,1 if self.reflected else 3,ptr(p),len(p),p.shape[1],ptr(out)))
            else:check(_dll.starry_map_gradients(h,ptr(p),len(p),p.shape[1],ptr(out)))
            values.append(out)
        if self.nw is None:return values[0].reshape((*shape,3))
        return np.stack(values,axis=-2).reshape((*shape,self.nw,3))

    def flux_jacobian(self,xo=0.,yo=0.,ro=0.,*,theta=0.,zo=1.,xs=0.,ys=0.,zs=1.,Rs=0.):
        """Native analytic derivatives; angle derivatives use angle_unit."""
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,xs,ys,zs)
        p[:,20]=np.broadcast_to(Rs,shape).ravel()
        specialized=self.reflected or self.oblate
        count=8+self.Ny+self.udeg+(5 if specialized else 0);channels=[]
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        for i,h in enumerate(self._handles):
            p[:,17]=wavelengths[i]
            values=[]
            for row in p:
                out=np.empty(count)
                if specialized:
                    _dll.starry_specialized_jacobian.argtypes=[_u64,ct.c_uint32,_ptr,_size,_ptr,_size]
                    check(_dll.starry_specialized_jacobian(h,1 if self.reflected else 3,ptr(row),len(row),ptr(out),count))
                else:check(_dll.starry_map_jacobian(h,ptr(row),len(row),ptr(out),count))
                values.append(out)
            channels.append(np.array(values))
        result=channels[0].reshape((*shape,count)) if self.nw is None else np.stack(channels,axis=-2).reshape((*shape,self.nw,count))
        result[...,1:4]*=self._angle
        output={name:result[...,i] for i,name in enumerate(['flux','theta','inc','obl','xo','yo','ro','amp'])}
        output['y']=result[...,8:8+self.Ny];output['u']=result[...,8+self.Ny:8+self.Ny+self.udeg]
        if specialized:
            extra=result[...,8+self.Ny+self.udeg:]
            names=('xs','ys','zs','roughness','Rs') if self.reflected else ('f','omega','beta','tpole','wav')
            if self.reflected:extra[...,3]*=self._angle
            output.update({name:extra[...,i] for i,name in enumerate(names)})
        return output

    def illumination_jacobian(self,xo=0.,yo=0.,ro=0.,*,theta=0.,zo=1.,xs=0.,ys=0.,zs=1.,Rs=0.):
        """Native reflected source/scattering derivatives, including finite sources.

        Returns flux, xs, ys, zs, roughness, Rs. Roughness uses angle_unit.
        Exact rough-scattering phase poles and zero-radius extended sources
        need directional/one-sided limits and currently raise ValueError.
        """
        if not self.reflected:raise ValueError('illumination derivatives require a reflected map')
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,xs,ys,zs)
        p[:,20]=np.broadcast_to(Rs,shape).ravel();channels=[]
        _dll.starry_reflection_jacobian.argtypes=[_u64,_ptr,_size,_ptr]
        for h in self._handles:
            values=[]
            for row in p:
                out=np.empty(6);check(_dll.starry_reflection_jacobian(h,ptr(row),len(row),ptr(out)));values.append(out)
            channels.append(np.array(values))
        result=channels[0].reshape((*shape,6)) if self.nw is None else np.stack(channels,axis=-2).reshape((*shape,self.nw,6))
        result[...,4]*=self._angle
        return {name:result[...,i] for i,name in enumerate(('flux','xs','ys','zs','roughness','Rs'))}

    def oblate_jacobian(self,xo=0.,yo=0.,ro=0.,*,theta=0.,zo=1.):
        """Native viewing, flattening and gravity-filter derivatives.

        Angle derivatives use angle_unit; wavelength is in meters and polar
        temperature in kelvin. Coefficient and limb designs are separate.
        Derivatives at zero effective gravity require a limiting prescription.
        """
        if not self.oblate:raise ValueError('oblate derivatives require an oblate map')
        self._sync();p,shape=self._parameters(theta,xo,yo,zo,ro,0.,0.,1.);channels=[]
        wavelengths=np.broadcast_to(self.wav,(self.nw or 1,))
        _dll.starry_oblate_jacobian.argtypes=[_u64,_ptr,_size,_ptr]
        for i,h in enumerate(self._handles):
            p[:,17]=wavelengths[i];values=[]
            for row in p:
                out=np.empty(10);check(_dll.starry_oblate_jacobian(h,ptr(row),len(row),ptr(out)));values.append(out)
            channels.append(np.array(values))
        result=channels[0].reshape((*shape,10)) if self.nw is None else np.stack(channels,axis=-2).reshape((*shape,self.nw,10))
        result[...,1:4]*=self._angle
        return {name:result[...,i] for i,name in enumerate(('flux','theta','inc','obl','f','omega','beta','tpole','wav','amp'))}

    def flux_limb_darkened(self,u,xo=0.,yo=0.,ro=0.):
        limb=array(u)
        if limb.ndim!=1 or len(limb)>20 or self.ydeg+len(limb)>32:raise ValueError('invalid limb coefficients')
        saved=(self.udeg,self.u)
        try:
            self.udeg=len(limb);self.u=np.r_[-1.,limb]
            return self.flux(xo,yo,ro)
        finally:self.udeg,self.u=saved

    def design_matrix(self,**kwargs):
        y=self.y.copy();amp=self.amp;columns=[]
        try:
            self.amp=1.
            for i in range(y.size):
                self.y=np.zeros_like(y);self.y.flat[i]=1.;columns.append(np.atleast_1d(self.flux(**kwargs)).ravel())
        finally:self.y=y;self.amp=amp;self._sync()
        return np.stack(columns,axis=-1)
    def flux_design(self,xo=0.,yo=0.,ro=0.):return self.design_matrix(xo=xo,yo=yo,ro=ro)[0]
    def intensity_design_matrix(self,lat=0.,lon=0.):
        y=self.y.copy();amp=self.amp;columns=[]
        try:
            self.amp=1.
            for i in range(y.size):
                self.y=np.zeros_like(y);self.y.flat[i]=1.;columns.append(np.atleast_1d(self.intensity(lat,lon,illuminate=False)).ravel())
        finally:self.y=y;self.amp=amp;self._sync()
        return np.stack(columns,axis=-1)

    def set_data(self,flux,C=1.,cho_C=None):
        data=array(flux).ravel();noise=array(C) if cho_C is None else array(cho_C)@array(cho_C).T
        if cho_C is None and np.ndim(flux)>1 and noise.shape==np.shape(flux):noise=noise.ravel()
        if noise.ndim<2:noise=array(np.broadcast_to(noise,(len(data),)))
        elif noise.shape!=(len(data),len(data)):raise ValueError('noise covariance has wrong shape')
        self._data=(data,array(noise))
    def set_prior(self,mu=None,L=1.,cho_L=None):
        count=self.y.size
        if mu is None:mean=np.zeros(count)
        elif self.nw is not None and np.shape(mu)==(self.Ny,):mean=np.repeat(mu,self.nw)
        elif np.shape(mu)==self.y.shape:mean=array(mu).ravel()
        else:mean=np.broadcast_to(mu,(count,)).copy()
        cov=array(L) if cho_L is None else array(cho_L)@array(cho_L).T
        if self.nw is not None and cov.shape==(self.Ny,self.Ny):cov=np.kron(cov,np.eye(self.nw))
        elif self.nw is not None and cov.shape==(self.Ny,):cov=np.repeat(cov,self.nw)
        cov=covariance(cov,count)
        self._prior=(array(mean),array(cov))
    def remove_prior(self):self._prior=None
    def _infer(self,design):
        if self._data is None:raise ValueError('set_data must be called first')
        x=array(design);data,noise=self._data
        if x.shape!=(len(data),self.y.size):raise ValueError('wrong design matrix shape')
        mean,cov=(None,None) if self._prior is None else self._prior
        from _inverse_api import infer
        return infer(x,data,noise,mean,cov)
    def solve(self,*,design_matrix=None,**kwargs):
        x=self.design_matrix(**kwargs) if design_matrix is None else design_matrix
        mean,cov,_,_=self._infer(x);self._solution=(mean,cov)
        weights=mean.reshape(self.y.shape);self.amp=np.where(weights[0]!=0,weights[0],1.);self.y=weights/self.amp
        return mean,np.linalg.cholesky(cov)
    def lnlike(self,*,design_matrix=None,woodbury=True,**kwargs):
        if self._prior is None:raise ValueError('marginal likelihood requires a Gaussian prior')
        return self._infer(self.design_matrix(**kwargs) if design_matrix is None else design_matrix)[3]
    @property
    def solution(self):return self._solution
    def draw(self,rng=None):
        if self._solution is None:raise ValueError('solve must be called first')
        rng=np.random.default_rng() if rng is None else rng;mean,cov=self._solution
        x=rng.multivariate_normal(mean,cov);weights=x.reshape(self.y.shape);self.amp=np.where(weights[0]!=0,weights[0],1.);self.y=weights/self.amp;return x

class Primary:
    def __init__(self,map,*,r=1.,m=1.,prot=1.,t0=0.,theta0=0.):
        self.map=map;self.r=r;self.m=m;self.prot=prot;self.t0=t0;self.theta0=theta0
class Secondary(Primary):
    def __init__(self,map,*,porb=None,a=None,ecc=0.,inc=None,w=None,Omega=0.,**kwargs):
        super().__init__(map,**kwargs)
        if porb is None and a is None:raise ValueError('provide porb or a')
        self.porb=porb;self.a=a;self.ecc=ecc;self.inc=math.pi/2/map._angle if inc is None else inc
        self.w=math.pi/2/map._angle if w is None else w;self.Omega=Omega
class System:
    def __init__(self,primary,*secondaries,light_delay=False,texp=0.,oversample=31,order=0,orbit_convention='barycentric'):
        if orbit_convention not in ('barycentric','starry'):raise ValueError('orbit_convention must be barycentric or starry')
        self.orbit_convention=orbit_convention
        self.primary=primary;self.secondaries=list(secondaries);self.light_delay=light_delay;self.texp=texp;self.oversample=oversample
        if not isinstance(order,int) or order not in (0,1,2):raise ValueError('order must be 0, 1 or 2')
        self.order=order
    @property
    def bodies(self):return [self.primary,*self.secondaries]
    def _evaluate(self,t,mode,channel=0):
        times=array(np.atleast_1d(t));bodies=self.bodies;n=len(bodies);ids=np.empty(n,dtype=np.uint64);parameters=np.zeros((n,42));orbits=np.zeros((n-1,7))
        if times.ndim!=1:raise ValueError('times must be one dimensional')
        for i,b in enumerate(bodies):
            if b.map.oblate and i:raise ValueError('upstream does not support oblate secondary bodies')
            b.map._sync();ids[i]=b.map._handles[channel if b.map.nw is not None else 0]
            parameters[i,:10]=[b.r,b.m,b.prot,b.t0,b.theta0*b.map._angle,b.map.inc*b.map._angle,b.map.obl*b.map._angle,b.map.reflected,b.map.roughness*b.map._angle,b.map.udeg]
            parameters[i,10:10+b.map.udeg]=b.map.u[1:]
            parameters[i,30:34]=[b.map._rv,b.map.veq,b.map.alpha,b.map.source_npts]
            wav=np.broadcast_to(b.map.wav,(b.map.nw or 1,))[channel if b.map.nw is not None else 0]
            parameters[i,34:]=[b.map.oblate,b.map.f,b.map.omega,b.map.beta,b.map.tpole,wav,b.map.fdeg,b.map.normalized]
            if i:
                gm=1.3271244e20*(self.primary.m+b.m)
                if gm<=0:raise ValueError('orbital total mass must be positive')
                if b.porb is not None:
                    period=b.porb
                    if not math.isfinite(period) or period<=0:raise ValueError('period must be positive')
                    a=(gm*(period*86400/(2*math.pi))**2)**(1/3)/6.957e8
                else:
                    a=b.a
                    if not math.isfinite(a) or a<=0:raise ValueError('semimajor axis must be positive')
                    period=2*math.pi*math.sqrt((a*6.957e8)**3/gm)/86400
                orbits[i-1]=[a,period,b.ecc,b.inc*b.map._angle,b.w*b.map._angle,b.Omega*b.map._angle,b.t0]
        columns=3+7*(n-1)+7*n+(sum(9+b.map.Ny+b.map.udeg for b in bodies) if mode>=8 else 0)
        out=np.empty((times.size,n,3) if mode in (1,14) else (times.size,n,columns) if mode>=5 else (times.size,n))
        exposures=array(np.broadcast_to(self.texp,times.shape))
        if self.orbit_convention not in ('barycentric','starry'):raise ValueError('invalid orbit_convention')
        flags=int(bool(self.light_delay))+2*(self.orbit_convention=='starry')
        check(_dll.starry_system_batch_v2(ids.ctypes.data_as(ct.POINTER(_u64)),n,ptr(parameters),ptr(orbits),ptr(times),times.size,flags,ptr(exposures),self.oversample,self.order,mode,ptr(out)))
        return out
    def flux(self,t,total=True):
        counts={b.map.nw for b in self.bodies if b.map.nw is not None}
        if len(counts)>1:raise ValueError('inconsistent wavelength counts')
        if counts:
            values=np.stack([self._evaluate(t,0,c) for c in range(next(iter(counts)))],axis=-1)
            return values.sum(axis=1) if total else values.transpose(1,0,2)
        values=self._evaluate(t,0);return values.sum(axis=1) if total else values.T
    def position(self,t):
        p=self._evaluate(t,1);return tuple(p[:,:,i].T for i in range(3))
    def rv(self,t,keplerian=True,total=True):
        counts={b.map.nw for b in self.bodies if b.map.nw is not None}
        if len(counts)>1:raise ValueError('inconsistent wavelength counts')
        if counts:
            values=np.stack([self._evaluate(t,3 if keplerian else 4,c) for c in range(next(iter(counts)))],axis=-1)
            return values.sum(axis=1) if total else values.transpose(1,0,2)
        values=self._evaluate(t,3 if keplerian else 4)
        return values.sum(axis=1) if total else values.T

class DopplerMap:
    """Native convolution on a uniformly spaced log-wavelength grid.

    Spectrum arrays include half_width padding samples at each end. Components
    may have distinct maps and rest spectra. Angles follow the component Maps.
    """
    def __init__(self,maps,spectra,*,log_spacing,half_width,veq,inc=None,u=(),continuum_index=None):
        self.maps=[maps] if isinstance(maps,Map) else list(maps)
        self.spectra=array(spectra)
        if self.spectra.ndim==1:self.spectra=np.atleast_2d(self.spectra)
        if self.spectra.ndim!=2 or len(self.spectra)!=len(self.maps) or not self.maps:raise ValueError('one spectrum is required per component')
        if not isinstance(half_width,int) or half_width<1 or self.spectra.shape[1]<=2*half_width+1:raise ValueError('invalid padded spectrum length')
        if any(m.nw is not None or m.reflected or m.oblate for m in self.maps):raise ValueError('Doppler components must be scalar emitted maps')
        self.log_spacing=log_spacing;self.half_width=half_width;self.veq=veq;self.inc=inc;self.u=array(u)
        self.continuum_index=continuum_index
        if self.u.ndim!=1:raise ValueError('limb coefficients must be one dimensional')
    def _compute(self,theta,normalize,kernel):
        phases=np.atleast_1d(theta)
        if phases.ndim!=1:raise ValueError('theta must be scalar or one dimensional')
        frames=[]
        for phase in phases:
            values=[];baseline=0.
            for m,spectrum in zip(self.maps,self.spectra):
                m._sync();inc=m.inc if self.inc is None else self.inc
                n=2*self.half_width+1 if kernel else len(spectrum)-2*self.half_width;out=np.empty(n)
                check(_dll.starry_doppler(m._handles[0],inc*m._angle,float(phase)*m._angle,self.veq,ptr(self.u),len(self.u),self.log_spacing,self.half_width,None if kernel else ptr(spectrum),0 if kernel else len(spectrum),0,ptr(out)))
                values.append(out)
                if normalize:
                    k=np.empty(2*self.half_width+1)
                    check(_dll.starry_doppler(m._handles[0],inc*m._angle,float(phase)*m._angle,self.veq,ptr(self.u),len(self.u),self.log_spacing,self.half_width,None,0,0,ptr(k)));baseline+=k.sum()
            result=np.sum(values,axis=0)
            if normalize:
                if baseline==0:raise ValueError('zero Doppler continuum')
                result/=baseline
            frames.append(result)
        return frames[0] if np.ndim(theta)==0 else np.stack(frames)
    def flux(self,theta=0.,normalize=True):return self._compute(theta,normalize,False)
    def kernel(self,theta=0.,normalize=False):return self._compute(theta,normalize,True)


from _inverse_api import DopplerMixin, SystemMixin


class DopplerMap(DopplerMixin, DopplerMap):
    pass


class System(SystemMixin, System):
    pass
