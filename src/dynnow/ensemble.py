import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
from numba import njit
from scipy.signal import convolve
from pysteps.utils.spectral import rapsd
from dynnow.surrogates import comp_random_members_maaft_one_lead_time, comp_random_members_spectral_one_lead_time, comp_surrogates_hist_one_lead_time, comp_random_members_spectral_one_lead_time_nowcasting
import pyshtools as pysh
from functools import partial
import dask

class Ensemble(xr.Dataset):

    # https://www.pythontutorial.net/python-oop/python-__slots__/
    __slots__ = ()
    
    def __init__(self, *args, ensemble_type = None, **kwargs):
        '''self.forecast has shape (lead time, members, y, x) and self.observation has shape (lead time, y, x), self.bias has shape (lead time, y, x)
        spectral_expansion is either 'fourier' or 'spherical_harmonics'
        '''

        super().__init__(*args, **kwargs)
        
        attrs = {'ensemble': 'True'}
        if ensemble_type is not None:
            attrs['ensemble_type'] = ensemble_type
        if 'forecast' in self:
            attrs['spatial_shape'] = self.forecast.shape[-2:]
        elif 'observation' in self:
            attrs['spatial_shape'] = self.observation.shape[-2:]
        
        self.attrs.update(attrs)

        if 'ensemble_mean' not in self and 'forecast' in self:            
            self['ensemble_mean'] = self.forecast.mean(axis = 1)
        if 'error' not in self and 'observation' in self and 'ensemble_mean' in self:
            self['error'] = self.observation - self.ensemble_mean
        if 'bias' in self and 'debiased_error' not in self and 'observation' in self and 'ensemble_mean' in self:
            self['debiased_error'] = xr.DataArray(self.observation - self.bias - self.ensemble_mean, dims = ('lead_time', 'y', 'x'), coords = {'lead_time': self.lead_time})
        if 'ensemble_mean' in self and 'forecast' in self:
            self['member_vec'] = self.forecast - self.ensemble_mean
    
    def comp_power_spectrum(self, field):
        if f'spectral_{field}' not in self:
            self.comp_spectral_field(field)

        input_core_dims = self.spectral_dims
        output_core_dims = [self.PS_dim]
        dask_gufunc_kwargs = {'output_sizes': {self.PS_dim: len(self[self.PS_dim])}}
        
        if 'member' in self[f'spectral_{field}'].dims:
            input_core_dims = ['member'] + input_core_dims
            output_core_dims = ['member'] + output_core_dims
            dask_gufunc_kwargs['output_sizes']['member'] = len(self.member)
            
        self[f'PS_{field}'] = xr.apply_ufunc(self.comp_power_spectrum_one_lead_time,
                       self[f'spectral_{field}'],
                       dask = 'parallelized',
                       output_dtypes = [self[f'spectral_{field}'].dtype],
                       input_core_dims = [input_core_dims],
                       output_core_dims = [output_core_dims],
                       dask_gufunc_kwargs = dask_gufunc_kwargs,
                       vectorize = False)

    def comp_spectral_field(self, field):
        input_core_dims = self.spatial_dims
        output_core_dims = self.spectral_dims
        dask_gufunc_kwargs = {'output_sizes': {self.spectral_dims[i]: self.spectral_shape[i] for i in range(len(self.spectral_dims))}}
        
        if 'member' in self[field].dims:
            input_core_dims = ['member'] + input_core_dims
            output_core_dims = ['member'] + output_core_dims
            dask_gufunc_kwargs['output_sizes']['member'] = len(self.member)
                                                   
        spectral_field = xr.apply_ufunc(self.comp_spectral_transform_one_lead_time,
                                        self[field],
                                        dask = 'parallelized',
                                        output_dtypes = [self[field].dtype],
                                        input_core_dims = [input_core_dims],
                                        output_core_dims = [output_core_dims],
                                        dask_gufunc_kwargs = dask_gufunc_kwargs,
                                        vectorize = False)
        self[f'spectral_{field}'] = spectral_field

    def comp_grid_field(self, field):
        '''computes a field in grid space from its spectral coefficients'''
        assert field not in self
        
        grid_field = xr.apply_ufunc(self.comp_inverse_spectral_transform_one_lead_time,
                                    self[f'spectral_{field}'],
                                    dask = 'parallelized',
                                    output_dtypes = [self[f'spectral_{field}'].dtype],
                                    output_core_dims = [self.spatial_dims],
                                    input_core_dims = [self.spectral_dims],
                                    dask_gufunc_kwargs = {'output_sizes': {self.spatial_dims[i]: self.spatial_shape[i] for i in range(len(self.spatial_shape))}},
                                    vectorize = False)
        self[field] = grid_field

    def comp_eigenval_eigenvec(self):

        space = self.eigenvec_space # spatial or spectral
        dims = self.attrs[f'{space}_dims']
        shape = self.attrs[f'{space}_shape']
        
        if space == 'spectral':
            if 'spectral_forecast' not in self:
                self.comp_spectral_field('forecast')
            f = self.spectral_forecast
            name = 'spectral_eigenvec'
        if space == 'spatial':
            if 'forecast' not in self:
                self.comp_grid_field('forecast')
            f = self.forecast
            name ='eigenvec'

        input_core_dims = [['member'] + dims]
        output_core_dims = [['order'], ['order'] + dims]
        dask_gufunc_kwargs = {'output_sizes': {dims[i]: shape[i] for i in range(len(dims))}}
        dask_gufunc_kwargs['output_sizes']['order'] = len(self.member) - 1
        
        eival, eivec = xr.apply_ufunc(self.comp_eigenval_eigenvec_one_lead_time,
               f,
               dask = 'parallelized',
               output_dtypes = [f.dtype, f.dtype],
               input_core_dims = input_core_dims,
               output_core_dims = output_core_dims,
               dask_gufunc_kwargs = dask_gufunc_kwargs,
                vectorize = False)
            
        self['eigenval'] = eival
        self[name] = eivec

    def comp_eigenval_eigenvec_one_lead_time(self, forecast):
        '''forecast has shape (1, member,) + forecast_shape'''
        forecast = forecast.squeeze(axis = 0)
        members_vec = forecast - np.mean(forecast, axis = 0, keepdims = True)
        members_vec = members_vec.reshape(len(self.member), np.prod(self.spectral_shape))
        _, sqrtval, vec = np.linalg.svd(members_vec/np.sqrt(len(self.member) - 1), full_matrices = False)
        vec = vec.reshape((len(self.member),) + self.spectral_shape)
        return sqrtval[None, :-1]**2, vec[None, :-1] # remove the last zero eigenvalue and corresponding eigenvector

    def comp_cosine_error_members(self):
        if 'spectral_error' not in self:
            self.comp_spectral_field('error')
        if 'spectral_eigenvec' not in self:
            self.comp_eigenval_eigenvec()
        
        num = (self.spectral_error * da.conj(self.spectral_eigenvec)).sum(dim = self.spectral_dims)
        error_norm = da.sqrt((da.abs(self.spectral_error)**2).sum(dim = self.spectral_dims))
        cosine = num / error_norm
        self['cosine_error_members'] = cosine

    def comp_cosine_error_projection(self):
        '''
        one can show analytically that corr(e, u_i)^2 = |proj_{u_i}e|^2 / |e|^2, so that the fraction of e projected on the space spanned by u_i's is sum_i corr(e, u_i)^2
        one can also show that sqrt(sum(corr(u_i, e)**2)) is the correlation (=cos of angle) between the e and its projection onto the subspace spanned by the u_i
        '''
        if 'cosine_error_members' not in self:
            self.comp_cosine_error_members()
        
        self['cosine_error_projection'] = da.sqrt((self.cosine_error_members**2).sum(dim = 'order'))

    def comp_d_spectrum(self):
        if 'spectral_error' not in self:
            self.comp_spectral_field('error')
        if 'spectral_eigenvec' not in self:
            self.comp_eigenval_eigenvec()
        
        num = (self.spectral_error * da.conj(self.spectral_eigenvec)).sum(dim = self.spectral_dims)
        self['d_spectrum'] = num**2 / self.eigenval
    
    def comp_FSS(self, thresholds, scales, weights = None):
        
        FSS = xr.apply_ufunc(comp_FSS_one_lead_time,
               self.observation,
               self.forecast,
               xr.DataArray(thresholds, dims = ('threshold_FSS')),
               xr.DataArray(scales, dims = ('scale_FSS')),
               dask = 'parallelized',
               output_dtypes = [self.forecast.dtype],
               input_core_dims = [self.spatial_dims, ['member'] + self.spatial_dims, ['threshold_FSS'], ['scale_FSS']],
               output_core_dims = [['threshold_FSS', 'scale_FSS']],
               dask_gufunc_kwargs = {'output_sizes': {'threshold_FSS': len(thresholds), 'scale_FSS': len(scales)}},
               vectorize = False)
        FSS['threshold_FSS'] = thresholds
        FSS['scale_FSS'] = scales

        self['FSS'] = FSS
        
    def comp_histos_projection_eigenvec(self, bins = np.linspace(-1., 1., 101)):

        if 'spectral_eigenvec' not in self:
            self.comp_eigenval_eigenvec()
        if 'spectral_member_vec' not in self:
            self.comp_spectral_field('member_vec')
        
        histos = da.full((len(self.lead_time), len(self.order), len(bins) - 1), 0., chunks = (1, -1, -1))

        num = (self.spectral_member_vec * da.conj(self.spectral_eigenvec)).sum(dim = self.spectral_dims)                       # shape (lead_time, member, order)
        denom = da.sqrt((da.abs(self.spectral_member_vec)**2).sum(dim = self.spectral_dims))                                   # shape (lead time, member)
        proj = num/denom                                                                                                       # shape (lead time, member, order)

        for lt in range(len(self.lead_time)):
            for o in range(len(self.order)):
                histos[lt, o], _ = da.histogram(proj.data[lt, :, o], bins = bins, density = True)
        
        self['histos_projection_eigenvec'] = xr.DataArray(histos,
                                                          dims = ('lead_time', 'order', 'bins_center'),
                                                          coords = {'lead_time': self.lead_time, 'bins_center': (bins[1:] + bins[:-1])/2},
                                                          attrs = {'bin_width' : bins[1] - bins[0]}
                                                         )
    
    def generate_surrogates(self, ensemble_type, *args, **kwargs):

        if 'spectral' in ensemble_type:
            if 'nowcasting' in ensemble_type:
                f = comp_random_members_spectral_one_lead_time_nowcasting
            else:
                f = comp_random_members_spectral_one_lead_time
            generating_func_one_lead_time = partial(
                f,
                spectral_transform = self.comp_spectral_transform_one_lead_time,
                inverse_spectral_transform = self.comp_inverse_spectral_transform_one_lead_time
            )
        
        elif ensemble_type == 'maaft':
            generating_func_one_lead_time = partial(
                comp_random_members_maaft_one_lead_time,
                spectral_transform=self.comp_spectral_transform_one_lead_time,
                inverse_spectral_transform=self.comp_inverse_spectral_transform_one_lead_time
            )
        elif ensemble_type == 'hist':
            generating_func_one_lead_time = comp_surrogates_hist_one_lead_time
        else:
            print(f'Ensemble type {ensemble_type} not implemented')

        surrogates = xr.apply_ufunc(generating_func_one_lead_time,
                                    self.forecast,
                                    dask = 'parallelized',
                                    output_dtypes = [self.forecast.dtype],
                                    input_core_dims = [self.spatial_dims] + [[]] * len(args),
                                    output_core_dims = [self.spatial_dims],
                                    dask_gufunc_kwargs = {'output_sizes': {self.spatial_dims[i]: self.spatial_shape[i] for i in range(len(self.spatial_shape))}},
                                    vectorize = False
                                   )

        surrogates = type(self)(data_vars = {'forecast': surrogates, 'observation': self.observation})
        
        return surrogates
        
class SphericalHarmonicsEnsemble(Ensemble):

    # https://www.pythontutorial.net/python-oop/python-__slots__/
    __slots__ = ()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs.update({'spatial_dims': ['latitude', 'longitude'],
                           'spectral_dims': ['i', 'l', 'm'],
                           'PS_dim': 'l',
                           'spectral_shape': (2, min(len(self.latitude), len(self.longitude))//2, min(len(self.latitude), len(self.longitude))//2),
                           'eigenvec_space': 'spectral',
                          })
        lmax = min(self.spatial_shape)//2 - 1
        self['l'] = np.arange(0, lmax + 1)
        self['scale'] = ('l', self.comp_scales())

    def comp_scales(self):
        '''return the scales corresponding to the rapsd of spherical harmonics'''

        R = 6.371e3 # earth radius
        lmax = min(self.spatial_shape)//2 - 1
        l = np.arange(0, lmax + 1)
        return 2*np.pi * R/np.sqrt(l * (l+1))

    def comp_power_spectrum_one_lead_time(self, spectral_field):
        '''spectral_field has shape (1,) + spectral_shape or (1, member) + spectral_shape'''
        
        leading_shape = spectral_field.shape[:-3]
        field_flat = spectral_field.reshape(-1, *self.spectral_shape)

        res = []
        for i in range(len(field_flat)):
            res.append(pysh.spectralanalysis.spectrum(field_flat[i], unit = 'per_l'))
        res = np.stack(res)
        
        return res.reshape(*leading_shape, len(self.l))

    def comp_spectral_transform_one_lead_time(self, field):
        '''field has shape (1,) + spatial_shape or (1, member) + spatial_shape'''
        sampling = max(self.spatial_shape) // min(self.spatial_shape)
        
        leading_shape = field.shape[:-2]
        field_flat = field.reshape(-1, *self.spatial_shape)

        res = []
        for i in range(len(field_flat)):
            res.append(pysh.expand.SHExpandDH(field_flat[i], sampling=sampling))
        res = np.stack(res)
        
        return res.reshape(*leading_shape, *self.spectral_shape)

    def comp_inverse_spectral_transform_one_lead_time(self, field):
        '''field has shape (1,) + spectral_shape, or (1, member) + spectral_shape'''
        sampling = max(self.spatial_shape)//min(self.spatial_shape)

        leading_shape = field.shape[:-3]
        field_flat = field.reshape(-1, *self.spectral_shape)

        res = []
        for i in range(len(field_flat)):
            res.append(pysh.expand.MakeGridDH(field_flat[i], sampling=sampling))
        res = np.stack(res)
        
        return res.reshape(*leading_shape, *self.spatial_shape)
            

class FourierEnsemble(Ensemble):

    # https://www.pythontutorial.net/python-oop/python-__slots__/
    __slots__ = ()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['scale'] = self.comp_scales()
        self.attrs.update({'spatial_dims': ['y', 'x'],
                           'spectral_dims': ['ky', 'kx'],
                           'PS_dim': 'scale',
                           'spectral_shape': (len(self.y), len(self.x)),
                           'eigenvec_space': 'spectral'
                          })

    def comp_scales(self):
        '''return the scales corresponding the rapsd function of pysteps (assuming 1km resolution)'''
        
        l = max(self.spatial_shape)
        if l % 2 == 1:
            r_range = np.arange(0, int(l / 2) + 1)
        else:
            r_range = np.arange(0, int(l / 2))
        freq = np.fft.fftfreq(l)
        freq = freq[r_range]
        scales = 1/freq
        return scales
        
    def comp_spectral_transform_one_lead_time(self, field):
        # field is supposed to be real but np.fft.rfft2 is not used because the rapsd function used in comp_power_spectrum_one_lead_time expects Fourier coefficients to be those of np.fft.fft2
        return np.fft.fft2(field)
        
    def comp_inverse_spectral_transform_one_lead_time(self, field):
        # take the real part because some imaginary part can remain
        return np.real(np.fft.ifft2(field))
        
    def comp_power_spectrum_one_lead_time(self, spectral_field):
        '''spectral_field has shape (1,) + spectral_shape or (1, member) + spectral_shape'''
        
        leading_shape = spectral_field.shape[:-2]
        field_flat = spectral_field.reshape(-1, *self.spectral_shape)

        res = []
        for i in range(len(field_flat)):
            psd = np.fft.fftshift(field_flat[i])
            psd = np.abs(psd)**2 / psd.size
            res.append(rapsd(psd))
        res = np.stack(res)

        return res.reshape(*leading_shape, len(self.scale))

def comp_FSS_one_lead_time(O, P, thresholds, scales):

    O = O.squeeze(axis = 0)
    P = P.squeeze(axis = 0)
    
    FSS = np.full((len(thresholds), len(scales)), np.nan)
    
    for t, thresh in enumerate(thresholds):
        O_binary = O > thresh
        P_binary = P > thresh

        if len(P_binary.shape) == 3: # if P is an ensemble of forecast, its first axis should the ensemble dimension and we average over it
            P_binary = np.mean(P_binary, axis = 0)

        for s, sca in enumerate(scales):
            
            kernel = np.full((sca, sca), 1/sca**2)
            
            O_s = convolve(O_binary, kernel, mode = 'valid', method = 'fft')
            P_s = convolve(P_binary, kernel, mode = 'valid', method = 'fft')
            
            MSE_s = np.mean((O_s - P_s)**2)
            MSE_sref = np.mean((O_s**2 + P_s**2))

            FSS[t, s] = 1 - MSE_s / MSE_sref
    
    return FSS[None]
