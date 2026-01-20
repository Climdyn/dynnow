import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
from numba import njit
from scipy.signal import convolve
from pysteps.utils.spectral import rapsd
from dynnow.surrogates import comp_random_members_maaft_one_lead_time, comp_random_members_spectral_one_lead_time, comp_surrogates_hist_one_lead_time

#xr.register_dataarray_accessor('nowcasts')
class Ensemble(xr.Dataset):

    # https://www.pythontutorial.net/python-oop/python-__slots__/
    __slots__ = ()
    
    def __init__(self, nowcasts, observation, lead_times, bias = None, ensemble_type = None):
        '''members has shape (lead time, members, y, x) and observation has shape (lead time, y, x). Bias has shape (lead time, y, x)'''

        assert len(nowcasts.shape) == 4
        assert len(observation.shape) == 3
        assert nowcasts[:, 0].shape == observation.shape

        ensemble_mean = nowcasts.mean(axis = 1)
        attrs = {'spatial_shape': nowcasts.shape[-2:]}
        if ensemble_type is not None:
            attrs['ensemble_type'] = ensemble_type
        
        super().__init__(data_vars = {'nowcasts': (('lead_time', 'member', 'y', 'x'), nowcasts),
                                      'observation': (('lead_time', 'y', 'x'), observation),
                                      'ensemble_mean': (('lead_time', 'y', 'x'), ensemble_mean),
                                      'error': (('lead_time', 'y', 'x'), observation - ensemble_mean),
                                     },
                         coords = {'lead_time': lead_times},
                         attrs = attrs
                        )

        if bias is not None:
            self['debiased_error'] = xr.DataArray(observation - bias - ensemble_mean, dims = ('lead_time', 'y', 'x'), coords = {'lead_time': self.lead_time})
    
    def comp_scales(self):
        '''return the scales corresponding the rapsd function of pysteps'''
        l = max(self.spatial_shape)
        if l % 2 == 1:
            r_range = np.arange(0, int(l / 2) + 1)
        else:
            r_range = np.arange(0, int(l / 2))
        freq = np.fft.fftfreq(l)
        freq = freq[r_range]
        scales = 1/freq
        return scales

    def comp_spectral_field(self, field):
        '''field must be either observation, ensemble_mean, error or debiased_error '''
        scales = self.comp_scales()
        spectral_field = da.full((len(self.lead_time), len(scales)), np.nan, chunks = (1, -1))
        for lt in range(len(self.lead_time)):
            spectral_field[lt] = da.from_delayed(delayed_rapsd(self[field].data[lt]),
                                                shape = (len(scales),),
                                                dtype = np.float64)
    
        self[f'spectral_{field}'] = xr.DataArray(spectral_field, dims = ('lead_time', 'scale'), coords = {'lead_time': self.lead_time, 'scale': scales})

    def comp_spectral_variance(self):
        scales = self.comp_scales()
        spectral_variance = da.full((len(self.lead_time), len(scales)), np.nan, chunks = (1, -1))
        for lt in range(len(self.lead_time)):
            spectral_variance[lt] = da.from_delayed(comp_spectral_variance_one_lead_time(self.nowcasts.data[lt]),
                                                    shape = (len(scales),),
                                                    dtype = np.float64)
        
        self['spectral_variance']= xr.DataArray(spectral_variance, dims = ('lead_time', 'scale'), coords = {'lead_time': self.lead_time, 'scale': scales})

    def comp_spectral_eigenvec(self):

        if 'eigenvec' not in self:
            self.comp_eigenval_eigenvec()

        scales = self.comp_scales()
        
        spectral_eigenvec = da.full(self.eigenvec.shape[:2] + (len(scales),), np.nan, chunks = (1, -1, -1))
        for t in range(len(self.eigenvec)):
            spectral_eigenvec[t] = da.from_delayed(comp_spectral_eigenvec_one_lead_time(self.eigenvec.data[t]),
                                                   shape = (len(self.order), len(scales)),
                                                   dtype = np.float64)
        
        self['spectral_eigenvec'] = xr.DataArray(spectral_eigenvec, dims = ('lead_time', 'order', 'scale'), coords = {'lead_time': self.lead_time, 'scale': scales})
    
    def comp_eigenval_eigenvec(self):
        eigenval = da.full((len(self.lead_time), self.nowcasts.data.shape[1] - 1), np.nan, chunks = (1, -1))
        eigenvec = da.full((len(self.lead_time), self.nowcasts.data.shape[1] - 1,) + self.spatial_shape, np.nan, chunks = (1, -1, -1, -1))

        for lt in range(len(self.lead_time)):
            eival, eivec = comp_eigenval_eigenvec_one_lead_time(self.nowcasts[lt].data)
            eigenval[lt] = da.from_delayed(eival, shape = (len(self.member) - 1,), dtype = np.float64)
            eigenvec[lt] = da.from_delayed(eivec, shape = (len(self.member) - 1,) + self.spatial_shape, dtype = np.float64)

        self['eigenval'] = xr.DataArray(eigenval, dims = ('lead_time', 'order'), coords = {'lead_time': self.lead_time})
        self['eigenvec'] = xr.DataArray(eigenvec, dims = ('lead_time', 'order', 'y', 'x'), coords = {'lead_time': self.lead_time})

    def comp_FSS(self, thresholds = 10 * np.log10(np.array([0.25, 0.5, 1, 2, 4, 8, 16])), scales = 2**np.arange(9) + 1):
    
        FSS = da.full((len(self.lead_time), len(thresholds), len(scales)), np.nan, chunks = (1, -1, -1))
    
        for lt in range(len(self.lead_time)):
            FSS[lt] = da.from_delayed(comp_FSS_one_lead_time(self.observation.data[lt],
                                                             self.nowcasts.data[lt],
                                                             thresholds = thresholds,
                                                             scales = scales)[2],
                                      shape = (len(thresholds), len(scales)),
                                      dtype = np.float64)

        self['FSS'] = xr.DataArray(FSS,
                                   dims = ('lead_time', 'threshold_FSS', 'scale_FSS'),
                                   coords = {'lead_time': self.lead_time, 'threshold_FSS': thresholds, 'scale_FSS': scales})

    def comp_cosine_error_members(self):

        if 'eigenvec' not in self:
            self.comp_eigenval_eigenvec()
            
        num = da.sum(self.error.data[:, None] * self.eigenvec.data, axis = (2, 3))      # sum over the y and x dimensions
        error_norm = da.linalg.norm(self.error.data, axis = (1, 2))                     # norm of the error over y and x dimensions
        eigenvec_norm = da.linalg.norm(self.eigenvec.data, axis = (2, 3))               # norm over the y and x dimensions
        cosine = num/(error_norm[:, None] * eigenvec_norm)                              # cosine is scalar product divided by the norms (broadcasting error_norm over the member dimension)

        self['cosine_error_members'] = xr.DataArray(cosine, dims = ('lead_time', 'order'), coords = {'lead_time': self.lead_time})

    def comp_cosine_error_projection(self):
        '''
        one can show analytically that corr(e, u_i)^2 = |proj_{u_i}e|^2 / |e|^2, so that the fraction of e projected on the space spanned by u_i's is sum_i corr(e, u_i)^2
        one can also show that sqrt(sum(corr(u_i, e)**2)) is the correlation (=cos of angle) between the e and its projection onto the subspace spanned by the u_i
        '''

        if 'cosine_error_members' not in self:
            self.comp_cosine_error_members()
        
        self['cosine_error_projection'] = xr.DataArray(da.sqrt(da.sum(self.cosine_error_members.data**2, axis = 1)), dims = ('lead_time',), coords = {'lead_time' : self.lead_time})

    def comp_histos_projection_eigenvec(self, bins = np.linspace(-1., 1., 101)):

        if 'eigenvec' not in self:
            self.comp_eigenval_eigenvec()
        
        histos = da.full((len(self.lead_time), len(self.order), len(bins) - 1), 0., chunks = (1, -1, -1))

        residual_vec = self.nowcasts.data - da.expand_dims(self.ensemble_mean.data, axis = 1)                                  # shape (lead time, member, y, x)
        num = da.sum(da.expand_dims(residual_vec, axis = 2) * da.expand_dims(self.eigenvec.data, axis = 1), axis = (3, 4))     # shape (lead_time, member, order)
        denom = da.linalg.norm(residual_vec, axis = (2, 3))                                                                    # shape (lead time, member)
        proj = num/denom[:, :, None]                                                                                           # shape (lead time, member, order)

        for lt in range(len(self.lead_time)):
            for o in range(len(self.order)):
                histos[lt, o], _ = da.histogram(proj[lt, :, o], bins = bins, density = True)
        
        self['histos_projection_eigenvec'] = xr.DataArray(histos,
                                                          dims = ('lead_time', 'order', 'bins_center'),
                                                          coords = {'lead_time': self.lead_time, 'bins_center': (bins[1:] + bins[:-1])/2},
                                                          attrs = {'bin_width' : bins[1] - bins[0]}
                                                         )
    
    def spectral_analyze(self):
        
        self.comp_spectral_variance()
        self.comp_spectral_field('error')
        self.comp_spectral_field('ensemble_mean')
        self.comp_spectral_field('observation')
        
        if 'debiased_error' in self:
            self.comp_spectral_field('debiased_error')

    def ensemble_scores(self, thresholds_FSS = 10 * np.log10(np.array([0.25, 0.5, 1, 2, 4, 8, 16])), scales_FSS = 2**np.arange(9) + 1):
        '''computes for the ensemble: eigenvalues, eigenvectors, the correlation of error with members_vec, the projection of the error on ensemble members and the FSS'''

        self.comp_eigenval_eigenvec()
        self.comp_spectral_eigenvec()
        self.comp_cosine_error_members()
        self.comp_cosine_error_projection()                 
        self.comp_FSS(thresholds = thresholds_FSS, scales = scales_FSS)
    
    def generate_surrogates(self, ensemble_type, **kwargs):

        surrogates = da.full(self.nowcasts.shape, np.nan, chunks = (1, -1, -1, -1))
    
        if ensemble_type == 'maaft':
            generating_func_one_lead_time = comp_random_members_maaft_one_lead_time
        elif ensemble_type == 'spectral':
            generating_func_one_lead_time = comp_random_members_spectral_one_lead_time
        elif ensemble_type == 'hist':
            generating_func_one_lead_time = comp_surrogates_hist_one_lead_time
        else:
            print(f'Ensemble type {ensemble_type} not implemented')
        
        for lt in range(len(self.lead_time)):
            surrogates[lt] = da.from_delayed(generating_func_one_lead_time(self.nowcasts.data[lt], **kwargs),
                                                                           shape = self.nowcasts.shape[1:],
                                                                           dtype = np.float64)
    
        surrogates = Ensemble(surrogates, self.observation.data, lead_times = self.lead_time.data, ensemble_type = ensemble_type)
    
        return surrogates

@delayed
def comp_spectral_variance_one_lead_time(x):
    '''x has shape (members,) + spatial_shape'''
    spectral_variance = 0
    xmean = np.mean(x, axis = 0) # ensemble mean
    for m in range(len(x)):
        member_vec = x[m] - xmean
        sp = rapsd(member_vec, fft_method = np.fft)
        spectral_variance += sp
    spectral_variance /= (len(x) - 1)
    
    return spectral_variance

@delayed
def delayed_rapsd(field):
    return rapsd(field, fft_method = np.fft)

@delayed
def comp_spectral_error_one_lead_time(error):
    return rapsd(error, fft_method = np.fft)

@delayed
def comp_spectral_mean_one_lead_time(mean):
    return rapsd(mean, fft_method = np.fft)

@delayed(nout = 2)
@njit
def comp_eigenval_eigenvec_one_lead_time(pred):
    '''pred has shape (member,) + spatial_shape'''
    members_vec = pred - np.expand_dims(np.sum(pred, axis = 0)/len(pred), axis = 0)
    members_vec = members_vec.reshape(len(pred), pred.shape[1] * pred.shape[2])
    _, sqrtval, vec = np.linalg.svd(members_vec/np.sqrt(len(members_vec) - 1), full_matrices = False)
    return sqrtval[:-1]**2, vec.reshape(pred.shape)[:-1] # remove the last zero eigenvalue and corresponding eigenvector

@delayed(nout = 3)
def comp_FSS_one_lead_time(O, P, thresholds = 10 * np.log10(np.array([0.5, 1, 2, 4, 8])), scales = 2**np.arange(9) + 1):

    FSS = np.full((len(thresholds), len(scales)), np.nan)
    
    for t, thresh in enumerate(thresholds):
        O_binary = O > thresh
        P_binary = P > thresh

        if len(P_binary.shape) == 3:
            P_binary = np.mean(P_binary, axis = 0)

        for s, sca in enumerate(scales):
            
            kernel = np.full((sca, sca), 1/sca**2)
            
            O_s = convolve(O_binary, kernel, mode = 'valid', method = 'fft')
            P_s = convolve(P_binary, kernel, mode = 'valid', method = 'fft')

            MSE_s = np.mean((O_s - P_s)**2)
            MSE_sref = np.mean(O_s**2 + P_s**2)

            FSS[t, s] = 1 - MSE_s / MSE_sref
    
    return scales, thresholds, FSS

@delayed
def comp_spectral_eigenvec_one_lead_time(eigenvec):
    '''eigenvec has shape (member,) + spatial_shape'''
    spectral_eigenvec = np.full((len(eigenvec), eigenvec.shape[-1]//2), np.nan)
    
    for m in range(len(eigenvec)):
        spectral_eigenvec[m] = rapsd(eigenvec[m], fft_method = np.fft)
    
    return spectral_eigenvec