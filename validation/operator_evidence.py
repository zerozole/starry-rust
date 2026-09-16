"""Explicit numerical counterparts and evidence for the pinned core operators.

Evidence is component-level unless marked upstream. This inventory makes no
claim of universal numerical accuracy or Python/Theano signature compatibility.
"""
GROUPS=[
 ('OpsYlm','rT rTA1 A A1 A1Inv','src/basis.rs::a1,a2_inverse,disk_moments; src/matrix.rs::solve','validation/compare.py','upstream C++ basis/Green transforms'),
 ('OpsYlm','sT X flux','src/map.rs::flux,flux_design; src/solver.rs::polynomial_flux; src/rings.rs::flux_design','validation/compare.py validation/compare_rings.py validation/decimal_rings.py','upstream and high-precision reference'),
 ('OpsYlm','tensordotRz dotR right_project left_project RAxisAngle','src/rotation.rs::axis_angle,coefficients; src/map.rs::projected','validation/compare.py python/test_interface.py tests/harmonics.rs','upstream rotations plus harmonic identities'),
 ('OpsYlm','F','src/map.rs::limb_filtered,multiplied','tests/high_degree_filters.rs python/test_extended.py','filter algebra and upstream forward comparisons'),
 ('OpsYlm','spotYlm expand_spot spot','src/surface.rs::add_spot,axisymmetric_fit','python/test_pixels_spectral.py tests/pixels.rs','native harmonic/profile tests'),
 ('OpsYlm','pT P intensity latlon_to_xyz','src/basis.rs::polynomial,harmonics; src/map.rs::intensity','validation/compare.py python/test_interface.py','upstream basis and intensity'),
 ('OpsYlm','limbdark_is_physical get_minimum','src/surface.rs::limb_is_physical,minimize','python/test_pixels_spectral.py validation/upstream_python.py','upstream physicality/minimum tests'),
 ('OpsYlm','render compute_ortho_grid compute_ortho_grid_inc_obl compute_rect_grid compute_moll_grid','src/imaging.rs::render,coordinate_grid','python/test_imaging_system.py python/test_exposure.py','render/intensity and coordinate consistency'),
 ('OpsYlm','set_vector set_matrix','python/_starry_api.py::Map.__setitem__','python/test_interface.py validation/upstream_python.py','new state API; upstream indexing tests'),
 ('OpsLD','limbdark_is_physical','src/surface.rs::limb_is_physical','validation/upstream_python.py','upstream limb physicality'),
 ('OpsLD','intensity flux X','src/map.rs::limb_intensity,limb_flux; python/_starry_api.py::Map.design_matrix','validation/upstream_python.py tests/core.rs','upstream limb/intensity tests'),
 ('OpsLD','render render_ld','src/imaging.rs::render','python/test_imaging_system.py validation/upstream_python.py','native images and upstream limb tests'),
 ('OpsLD','set_vector','python/_starry_api.py::Map.__setitem__','validation/upstream_python.py','state indexing'),
 ('OpsRV','compute_rv_filter','src/rv.rs::velocity_polynomial; src/rv_derivatives.rs::filters','validation/compare_rv_derivatives.py','unmodified upstream method and complex-step derivatives'),
 ('OpsRV','rv','src/rv.rs::radial_velocity; src/rv_derivatives.rs::jacobian','validation/independent_velocity.py python/test_rv_derivatives.py','independent physical disk integration and perturbations'),
 ('OpsReflected','A1Big','src/basis.rs::a1','validation/compare.py','shared upstream basis transform'),
 ('OpsReflected','rT sT X_point_source X flux flux_point_source','src/reflected.rs::flux,flux_extended,flux_extended_ellipses','validation/compare_reflected.py validation/compare_specialized_gradients.py validation/compare_specialized_high.py','upstream C++ values and AD'),
 ('OpsReflected','intensity unweighted_intensity compute_illumination_point_source compute_illumination','src/reflected.rs::intensity,source_points,illumination_polynomial; src/imaging.rs::intensity','validation/upstream_python.py python/test_illumination_derivatives.py','upstream Oren-Nayar and normalization; finite-source chain checks'),
 ('OpsReflected','render','src/imaging.rs::render','python/test_imaging_system.py','reflected render/intensity consistency'),
 ('OpsOblate','tensordotRz dotR','src/rotation.rs::coefficients','validation/compare.py','shared upstream rotation evidence'),
 ('OpsOblate','render compute_ortho_grid_fproj','src/imaging.rs::render,coordinate_grid','python/test_imaging_system.py python/test_exposure.py','oblate render and projected grids'),
 ('OpsOblate','sT X flux','src/oblate.rs::flux,observed_map; src/oblate_derivatives.rs::flux_ellipses','validation/compare_oblate.py validation/compare_oblate_derivatives.py python/test_system_surface.py','upstream C++/AD and overlapping-silhouette derivatives'),
 ('OpsOblate','weight_ylms_by_grav_dark_filter grav_dark compute_grav_dark_filter','src/oblate.rs::GravityDarkening.profile,filter,observed_map','tests/oblate_system.rs python/test_oblate_derivatives.py tests/domain_stress.rs','gravity profiles, derivatives and upper-degree product tests'),
 ('OpsDoppler','enforce_shape enforce_bounds','src/doppler.rs::dimensions,kernel; python/_starry_api.py::DopplerMap','python/test_inverse.py python/test_doppler_derivatives.py','explicit shape/domain validation; new API rejects invalid inputs'),
 ('OpsDoppler','get_x get_rT get_kT0 get_kT0_matrix get_kT','src/doppler.rs::chord_moments,kernel; src/doppler_derivatives.rs::kernel','validation/independent_velocity.py tests/high_degree_filters.rs','analytic chord reference and high-degree checks'),
 ('OpsDoppler','get_D_data get_D get_D_fixed_spectrum get_D_fixed_map','src/doppler.rs::map_design,spectrum_design','tests/doppler_inverse.rs python/test_inverse.py','design versus convolution and conditional inference'),
 ('OpsDoppler','dot_design_matrix_fixed_map_into dot_design_matrix_fixed_map_transpose_into dot_design_matrix_into dot_design_matrix_transpose_into','src/doppler.rs::spectral_map_dot; python/_inverse_api.py::DopplerMixin.dot','tests/doppler_inverse.rs python/test_inverse.py','forward/transpose adjoint and dense operator checks'),
 ('OpsDoppler','get_flux_from_design get_flux_from_conv get_flux_from_dotconv get_flux_from_convdot','src/doppler.rs::spectrum,map_design,spectral_map_dot','tests/doppler_inverse.rs python/test_inverse.py validation/independent_velocity.py','convolution/design parity and analytic spectrum'),
 ('OpsDoppler','L1','src/inference.rs::l1','python/test_l1.py python/test_solver_modes.py','independent matrix solutions of source iteration'),
 ('OpsSystem','position','src/orbit.rs::state,state_derivatives; src/system.rs::states,apparent; src/source_orbits.rs::positions','tests/orbit_derivatives.rs validation/independent_system.py validation/compare_source_orbits.py','independent circular delayed orbit and elliptic perturbations'),
 ('OpsSystem','X','src/system.rs::flux; src/system_derivatives.rs::flux_surface; python/_inverse_api.py::SystemMixin.design_matrix','validation/independent_system.py python/test_system_surface.py validation/upstream_python.py','analytic eclipse derivatives and upstream system inference'),
 ('OpsSystem','rv','src/system.rs::radial_velocities; src/system_derivatives.rs::radial_velocity_surface','validation/independent_system.py validation/compare_source_orbits.py python/test_source_system.py python/test_system_derivatives.py python/test_system_surface.py','independent orbital RV, rotational/exposure quotient checks'),
 ('OpsSystem','render','python/_inverse_api.py::SystemMixin.render; src/imaging.rs::render','python/test_system_render.py','image, phase, radius and position consistency'),
]

def operator_evidence(root):
    result={}
    for cls,names,implementation,tests,reference in GROUPS:
        evidence=tests.split()
        for path in evidence:
            if not (root/path).is_file():raise FileNotFoundError(path)
        for name in names.split():
            key=cls+'.'+name
            if key in result:raise ValueError('duplicate evidence '+key)
            result[key]=dict(implementation=implementation,evidence=evidence,reference=reference)
    return result

def property_counterpart(filename,name):
    if name=='lazy':return 'excluded symbolic execution mode; eager native numerical API'
    if name.endswith('_unit'):
        return 'explicit units in NUMERICAL_API.md; angle_unit selects rad/deg, other unit conversion is caller-side'
    if filename=='maps.py':
        derived={'N':'(ydeg+udeg+fdeg+1)**2','Nf':'(fdeg+1)**2','Nu':'udeg+1',
                 'Ny':'Map.Ny','deg':'ydeg+udeg+fdeg','fproj':'1-sqrt(cos(inc)**2+(1-f)**2*sin(inc)**2)',
                 'solution':'solve() return (mean, lower Cholesky); stored solution state'}
        if name in derived:return derived[name]
        if name in ('ydeg','udeg','fdeg','nw','y','u','wav','inc','obl','alpha','veq','source_npts','roughness','f','omega','tpole','beta'):
            return 'Map.'+name+' in python/_starry_api.py; native synchronized parameters'
    if filename=='doppler.py':
        fields={'nc':'len(maps)','nt':'len(theta) per call; epochs are not constructor state',
          'nw':'len(wav) or native output length','nw0':'input rest wavelength length; inputs resampled by from_wavelengths',
          'nw_':'len(wav_internal)','nw0_':'spectra.shape[1]','oversample':'from_wavelengths(oversample=)',
          'ydeg':'maps[0].ydeg','Ny':'maps[0].Ny','udeg':'len(u)','Nu':'len(u)+1',
          'deg':'maps[0].ydeg+len(u)','N':'(maps[0].ydeg+len(u)+1)**2',
          'inc':'DopplerMap.inc or component Map.inc','obl':'source fixed zero; kernel uses Doppler inclination and phase',
          'veq':'DopplerMap.veq','vsini':'veq*sin(inc); native kernel applies source 1 m/s floor',
          'wav':'DopplerMap.wav','wav_':'DopplerMap.wav_internal',
          'wav0':'caller input rest_wavelengths; wav0 locally denotes padded grid',
          'wav0_':'DopplerMap.wav0 padded rest grid','wavc':'wav[continuum_index] for sampled normalization',
          'spectrum':'caller input resampled by constructor/load','spectrum_':'DopplerMap.spectra',
          'y':'absolute component weights maps[i].amp*maps[i].y','u':'DopplerMap.u (without fixed monopole)',
          'spectral_map':'sum of outer products of component weights and spectra; spectral_map_dot operator'}
        if name in fields:return fields[name]
    if filename=='kepler.py':
        fields={'omega':'Secondary.w alias in source; use w in new API',
                'a':'Secondary.a when supplied; otherwise derived from porb and masses during native synchronization',
                'porb':'Secondary.porb when supplied; otherwise derived from a and masses during native synchronization',
                'map_indices':'coefficient block offsets from each body.map.Ny (and spectral channels)',
                'solution':'System.solve() return (mean, lower Cholesky); stored solution state'}
        if name in fields:return fields[name]
        if name in ('map','r','m','prot','t0','theta0','ecc','w','Omega','inc','light_delay','texp','oversample','order','primary','secondaries','bodies'):
            return 'Primary/Secondary/System.'+name+' explicit state in python/_starry_api.py'
    raise ValueError('unreviewed property '+filename+':'+name)
