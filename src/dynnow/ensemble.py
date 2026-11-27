'''
Diagnostics of one ensemble forecast.

Most diagnostics only depend on the inner products between the member vectors x_i = forecast_i - ensemble mean
and the error e = observation - ensemble mean. They are thus computed in two steps:
1. comp_gram() computes the (small) Gram matrices of these vectors for every lead time, in pixel space or in
   spectral space (same inner product, see dynnow.geometry). In spectral space, the Gram matrices are also
   split by scale, which gives the power spectra.
2. the diagnostics are linear algebra on these Gram matrices (see dynnow.linalg); they do not touch the fields.

All the results are stored in self.diagnostics (an xr.Dataset)

Example
-------
geometry = SphereGeometry(forecast.latitude, forecast.longitude)
ensemble = Ensemble(forecast, geometry, observation=observation, space='spectral')
ensemble.comp_gram()
ensemble.comp_cosine_error_projection()
ensemble.comp_power_spectrum('error')
ensemble.diagnostics
'''

import numpy as np
import xarray as xr
from scipy.signal import convolve

from dynnow import linalg
from dynnow import surrogates as sur

from numba import njit, prange


class Ensemble:

    def __init__(self, forecast, geometry, observation=None, space='pixel'):
        '''
        forecast: DataArray (lead_time, member, *geometry.spatial_dims)
        geometry: SphereGeometry or PlaneGeometry
        observation: DataArray (lead_time, *geometry.spatial_dims), or None (then the diagnostics involving the error are not available)
        space: 'pixel' or 'spectral', where the inner products are computed.
               'spectral' also gives the Gram matrices split by scale (needed for the power spectra); it is
               only available on the whole domain (no mask, Driscoll-Healy grid for the sphere).
        '''
        assert space in ('pixel', 'spectral')
        if space == 'spectral':
            geometry.check_spectral()

        self.forecast = forecast
        self.observation = observation
        self.geometry = geometry
        self.space = space
        self.diagnostics = xr.Dataset()  # to store diagnostics
        self.intermediates = xr.Dataset() # to store intermediates quantities that are not diagnostics

    @property
    def n_member(self):
        return len(self.forecast.member)

    @property
    def ensemble_mean(self):
        return self.forecast.mean('member')

    @property
    def member_vec(self):
        return self.forecast - self.ensemble_mean

    @property
    def error(self):
        return self.observation - self.ensemble_mean

    # --- Gram matrices --------------------------------------------------------------------------------------

    def get_vectors(self):
        '''member vectors (and the error as last vector if there is an observation), dims (..., vec, *spatial_dims)'''
        vectors = self.member_vec.drop_vars('member').rename(member='vec')
        if self.observation is not None:
            vectors = xr.concat([vectors, self.error.expand_dims('vec')], dim='vec')
        return vectors

    def comp_gram(self):
        '''
        Stores
        - gram_members (lead_time, member, member1): <x_i, x_j>
        - gram_members_error (lead_time, member): <x_i, e>
        - gram_error (lead_time): <e, e>
        and in spectral space the same matrices split by scale: gram_members_by_scale (lead_time, l or k, member, member1), ...
        '''
        vectors = self.get_vectors()
        spatial_dims = self.geometry.spatial_dims

        if self.space == 'pixel':
            gram = xr.apply_ufunc(self.geometry.comp_pixel_gram_one_lead_time,
                                  vectors,
                                  input_core_dims=[['vec'] + spatial_dims],
                                  output_core_dims=[['vec', 'vec1']],
                                  dask='parallelized',
                                  output_dtypes=[np.float64],
                                  dask_gufunc_kwargs={'output_sizes': {'vec1': len(vectors.vec)}, 'allow_rechunk': True})
            self.store_gram(gram, suffix='')

        if self.space == 'spectral':
            PS_dim = self.geometry.PS_dim
            gram_by_scale = xr.apply_ufunc(self.geometry.comp_scale_gram_one_lead_time,
                                           vectors,
                                           input_core_dims=[['vec'] + spatial_dims],
                                           output_core_dims=[[PS_dim, 'vec', 'vec1']],
                                           dask='parallelized',
                                           output_dtypes=[np.float64],
                                           dask_gufunc_kwargs={'output_sizes': {PS_dim: len(self.geometry.PS_coord), 'vec1': len(vectors.vec)},
                                                               'allow_rechunk': True})
            gram_by_scale = gram_by_scale.assign_coords({PS_dim: self.geometry.PS_coord, 'scale': (PS_dim, self.geometry.scales)})
            self.store_gram(gram_by_scale, suffix='_by_scale')
            self.store_gram(gram_by_scale.sum(PS_dim), suffix='')

    def store_gram(self, gram, suffix):
        '''splits the Gram matrix of the vectors (members, error) into its blocks'''
        M = self.n_member
        member = self.forecast.member.values
        self.intermediates['gram_members' + suffix] = gram.isel(vec=slice(0, M), vec1=slice(0, M))\
            .rename(vec='member', vec1='member1').assign_coords(member=member, member1=member)
        if self.observation is not None:
            self.intermediates['gram_members_error' + suffix] = gram.isel(vec=slice(0, M), vec1=M)\
                .rename(vec='member').assign_coords(member=member)
            self.intermediates['gram_error' + suffix] = gram.isel(vec=M, vec1=M)

    # --- diagnostics computed from the Gram matrices ----------------------------------------------------------

    def comp_eigenval_eigenvec(self):
        '''eigenvalues of the covariance (lead_time, order) and eigenvector coefficients (lead_time, member, order)'''
        eigenval, Q = linalg.comp_eigenval_eigenvec_coefficients(self.intermediates.gram_members)
        self.diagnostics['eigenval'] = eigenval
        self.intermediates['eigenvec_coefficients'] = Q

    def comp_error_dot_eigenvec(self):
        '''<e, v_k> = sum_i Q_ik <x_i, e>'''
        if 'eigenvec_coefficients' not in self.intermediates:
            self.comp_eigenval_eigenvec()
        return xr.dot(self.intermediates.eigenvec_coefficients, self.intermediates.gram_members_error, dim='member')

    def comp_cosine_error_members(self):
        '''cosine between the error and each eigenvector (lead_time, order)'''
        self.diagnostics['cosine_error_members'] = self.comp_error_dot_eigenvec() / np.sqrt(self.intermediates.gram_error)

    def comp_cosine_error_projection(self):
        '''
        one can show analytically that corr(e, u_i)^2 = |proj_{u_i}e|^2 / |e|^2, so that the fraction of e projected on the space spanned by u_i's is sum_i corr(e, u_i)^2
        one can also show that sqrt(sum(corr(u_i, e)**2)) is the correlation (=cos of angle) between the e and its projection onto the subspace spanned by the u_i
        '''
        if 'cosine_error_members' not in self.diagnostics:
            self.comp_cosine_error_members()
        self.diagnostics['cosine_error_projection'] = np.sqrt((self.diagnostics.cosine_error_members**2).sum('order'))

    def comp_d_spectrum(self):
        '''<e, v_k>^2 / lambda_k (lead_time, order)'''
        self.diagnostics['d_spectrum'] = self.comp_error_dot_eigenvec()**2 / self.diagnostics.eigenval

    def comp_multivariate_energy_score(self, fair=True):
        '''
        1/M sum_i |x_i - y| - 1/(2 n) sum_{i != j} |x_i - x_j|, with x_i - y = v_i - e and x_i - x_j = v_i - v_j
        fair=True: n = M (M - 1) (fair energy score), fair=False: n = M^2 (standard energy score)
        (the squared distances are clipped at 0 because of the rounding errors, and the diagonal i = j is set to 0)
        '''
        M = self.n_member
        gram = self.intermediates.gram_members
        norm2 = linalg.diagonal(gram)                                                              # |x_i|^2
        dist2_obs = norm2 + self.intermediates.gram_error - 2 * self.intermediates.gram_members_error  # |x_i - e|^2
        dist2_members = norm2 + norm2.rename(member='member1') - 2 * gram                          # |x_i - x_j|^2
        dist_members = np.sqrt(dist2_members.clip(min=0)).where(gram.member != gram.member1, 0.)
        n_pairs = M * (M - 1) if fair else M**2
        mean_dist_obs = np.sqrt(dist2_obs.clip(min=0)).mean('member')
        mean_dist_members = dist_members.sum(['member', 'member1']) / n_pairs
        self.diagnostics['multivariate_energy_score'] = mean_dist_obs - mean_dist_members / 2

    def comp_projection_members_eigenvec(self):
        '''cosine between each member vector and each eigenvector: <x_i, v_k> / |x_i| = sum_j G_ij Q_jk / sqrt(G_ii), (lead_time, member, order)'''
        if 'eigenvec_coefficients' not in self.intermediates:
            self.comp_eigenval_eigenvec()
        gram = self.intermediates.gram_members
        Q = self.intermediates.eigenvec_coefficients.rename(member='member1')
        return xr.dot(gram, Q, dim='member1') / np.sqrt(linalg.diagonal(gram))

    def comp_histos_projection_eigenvec(self, bins=np.linspace(-1., 1., 101)):
        '''histogram over the members of the cosines between the member vectors and each eigenvector (lead_time, order, bins_center)'''
        projection = self.comp_projection_members_eigenvec()
        histos = xr.apply_ufunc(lambda x: np.histogram(x, bins=bins, density=True)[0],
                                projection,
                                input_core_dims=[['member']],
                                output_core_dims=[['bins_center']],
                                vectorize=True,
                                dask='parallelized',
                                output_dtypes=[np.float64],
                                dask_gufunc_kwargs={'output_sizes': {'bins_center': len(bins) - 1}})
        self.diagnostics['histos_projection_eigenvec'] = histos.assign_coords(bins_center=(bins[1:] + bins[:-1]) / 2)\
            .assign_attrs(bin_width=bins[1] - bins[0])

    def comp_power_spectrum(self, field):
        '''
        field: 'error', 'member_vec' or 'eigenvec' (spectral space only), stored as PS_{field}
        PS(e)_s = <e, e>_s, PS(x_i)_s = <x_i, x_i>_s, PS(v_k)_s = sum_ij Q_ik Q_jk <x_i, x_j>_s
        '''
        assert self.space == 'spectral', 'the power spectra need the Gram matrices split by scale (space="spectral")'
        if field == 'error':
            PS = self.intermediates.gram_error_by_scale
        elif field == 'member_vec':
            PS = linalg.diagonal(self.intermediates.gram_members_by_scale)
        elif field == 'eigenvec':
            if 'eigenvec_coefficients' not in self.intermediates:
                self.comp_eigenval_eigenvec()
            Q = self.intermediates.eigenvec_coefficients
            PS = xr.dot(Q, self.intermediates.gram_members_by_scale, Q.rename(member='member1'), dim=['member', 'member1'])
        else:
            raise ValueError(f'no power spectrum for {field}')
        self.diagnostics[f'PS_{field}'] = PS
        
    def comp_loo(self):
        '''setups the experiment where one member is used as the verifying observation (the real observation is discarded)'''
        idxs = np.arange(len(self.forecast.member))
        idx = np.random.choice(idxs)
        
        loo = Ensemble(self.forecast.isel(member = idxs[idxs != idx]), self.geometry, observation = self.forecast.isel(member = idx), space = self.space)
        loo.comp_gram()
        loo.comp_cosine_error_projection()
        loo.comp_cosine_error_members()

        return loo

    def comp_ensemble_convergence(self, smaller_sizes=[5, 10, 20, 50], agreement_order=4, seed=None):
        '''
        Computes diagnostics to assess the convergence of the ensemble in phase space
        Several smaller ensembles are defined (of sizes smaller_sizes) and the eigenvalues of the covariance matrices of these ensembles are computed
        Are also computed the principal angles between the first agreement_order eigenvectors of the smaller ensembles and those of the full ensemble
        The smaller ensembles are sub-blocks of the Gram matrix (re-centered on their own mean), no field is used.
        Returns a DataTree with one child per size.
        '''
        if 'eigenvec_coefficients' not in self.intermediates:
            self.comp_eigenval_eigenvec()
        rng = np.random.default_rng(seed)
        gram = self.intermediates.gram_members
        Q = self.intermediates.eigenvec_coefficients.isel(order=slice(0, agreement_order))

        ensemble_convergence = xr.DataTree(xr.Dataset(coords={'lead_time': self.forecast.lead_time}))
        for n_member in smaller_sizes:
            members = rng.choice(self.n_member, size=n_member, replace=False)
            smaller_gram = linalg.recenter(gram.isel(member=members, member1=members))
            smaller_eigenval, smaller_Q = linalg.comp_eigenval_eigenvec_coefficients(smaller_gram)
            smaller_Q = smaller_Q.isel(order=slice(0, agreement_order))
            # the eigenvectors of the smaller ensemble are combinations of the x_i (see dynnow.linalg), so the
            # inner products with those of the full ensemble only need <x_i, x_j> for j in the smaller ensemble
            principal_angles = linalg.comp_principal_angles(Q, gram.isel(member1=members), smaller_Q)
            ensemble_convergence[f'{n_member}members'] = xr.DataTree(xr.Dataset(data_vars={
                'eigenval': smaller_eigenval.rename(order='smaller_order'),
                'principal_angles': principal_angles.rename(order='agreement_order')}))
        return ensemble_convergence

    # --- diagnostics that need the fields ------------------------------------------------------------------

    def comp_eigenvec(self):
        '''eigenvectors in pixel space v_k = sum_i Q_ik x_i, (lead_time, order, *spatial_dims) (returned, not stored)'''
        if 'eigenvec_coefficients' not in self.intermediates:
            self.comp_eigenval_eigenvec()
        return xr.dot(self.intermediates.eigenvec_coefficients, self.member_vec, dim='member')

    def comp_FSS(self, thresholds, scales):
        '''fractions skill score of the ensemble (plane only: the smoothing kernel is a square of pixels)'''
        from geometry import SphereGeometry
        assert isinstance(geometry, SphereGeometry)
        spatial_dims = self.geometry.spatial_dims
        FSS = xr.apply_ufunc(comp_FSS_one_lead_time,
                             self.observation,
                             self.forecast,
                             kwargs={'thresholds': thresholds, 'scales': scales},
                             input_core_dims=[spatial_dims, ['member'] + spatial_dims],
                             output_core_dims=[['threshold_FSS', 'scale_FSS']],
                             vectorize=True,
                             dask='parallelized',
                             output_dtypes=[np.float64],
                             dask_gufunc_kwargs={'output_sizes': {'threshold_FSS': len(thresholds), 'scale_FSS': len(scales)}})
        self.diagnostics['FSS'] = FSS.assign_coords(threshold_FSS=thresholds, scale_FSS=scales)
        
    def comp_variogram_score(self, factor = 15, p = 0.5):
        """Variogram score (Scheuerer & Hamill 2015), weights cos(lat_i) cos(lat_j) / dist_ij"""

        forecast = self.geometry.downscale(self.forecast, factor = factor)
        observation = self.geometry.downscale(self.observation, factor = factor)
        
        variogram_score = xr.apply_ufunc(comp_variogram_score_one_lead_time,
                                         forecast,
                                         observation,
                                         forecast.latitude,
                                         forecast.longitude,
                                         p,
                                         input_core_dims = [['member', 'latitude', 'longitude'], ['latitude', 'longitude'], ['latitude'], ['longitude'], []],
                                         output_core_dims = [[]],
                                         dask = 'parallelized',
                                         output_dtypes = [forecast.dtype],
                                         vectorize = True)
        self.diagnostics['variogram_score'] = variogram_score
    

    def generate_surrogates(self, kind, **kwargs):
        '''
        kind: 'spectral' (sphere only), 'spectral_nowcasting', 'maaft' or 'hist' (see dynnow.surrogates)
        returns a new Ensemble with the surrogate members, the same observation, geometry and space
        '''
        functions = {'spectral': sur.comp_surrogates_spectral,
                     'spectral_nowcasting': sur.comp_surrogates_spectral_nowcasting,
                     'maaft': sur.comp_surrogates_maaft,
                     'hist': sur.comp_surrogates_hist}
        if kind != 'hist':
            kwargs['spectral_transform'] = self.geometry.to_spectral
            kwargs['inverse_spectral_transform'] = self.geometry.to_pixel

        core_dims = ['member'] + self.geometry.spatial_dims
        surrogates = xr.apply_ufunc(functions[kind],
                                    self.forecast,
                                    kwargs=kwargs,
                                    input_core_dims=[core_dims],
                                    output_core_dims=[core_dims],
                                    vectorize=True,
                                    dask='parallelized',
                                    output_dtypes=[self.forecast.dtype])  # all the surrogate functions keep the dtype of the forecast
        return Ensemble(surrogates.transpose(*self.forecast.dims), self.geometry, observation=self.observation, space=self.space)


def comp_FSS_one_lead_time(O, P, thresholds, scales):
    '''O: observation (y, x), P: forecast (member, y, x) or (y, x)'''
    FSS = np.full((len(thresholds), len(scales)), np.nan)

    for t, thresh in enumerate(thresholds):
        O_binary = O > thresh
        P_binary = P > thresh

        if len(P_binary.shape) == 3:  # if P is an ensemble of forecast, its first axis should the ensemble dimension and we average over it
            P_binary = np.mean(P_binary, axis=0)

        for s, sca in enumerate(scales):
            kernel = np.full((sca, sca), 1 / sca**2)

            O_s = convolve(O_binary, kernel, mode='valid', method='fft')
            P_s = convolve(P_binary, kernel, mode='valid', method='fft')

            MSE_s = np.mean((O_s - P_s)**2)
            MSE_sref = np.mean((O_s**2 + P_s**2))

            FSS[t, s] = 1 - MSE_s / MSE_sref

    return FSS

@njit(parallel = True, fastmath = True, cache = True)
def comp_variogram_score_one_lead_time(forecast, observation, lat, lon, p):
    """forecast: (member, latitude, longitude), observation: (latitude, longitude)
    Returns (sum_ij w_ij a_ij^2, sum_ij w_ij) over pairs i != j, w_ij = w_i w_j / great-circle distance."""

    lat, lon = np.broadcast_arrays(np.deg2rad(lat)[:, None], np.deg2rad(lon)[None])
    lat, lon = lat.flatten(), lon.flatten()
    d = len(lat) # d = nlat * nlon
    xyz = np.zeros((len(lat), 3))
    xyz[:, 0] = np.cos(lat)*np.cos(lon)
    xyz[:, 1] = np.cos(lat)*np.sin(lon)
    xyz[:, 2] = np.sin(lat)
    w = np.cos(lat)
    
    
    N = len(forecast)
    x = np.ascontiguousarray(forecast.transpose((1, 2, 0))).reshape(d, -1) # shape (nlat * nlon, nmembers)
    y = observation.flatten()
    
    num = 0.; den = 0.
    for i in prange(d):
        for j in range(d):
            if j == i:
                continue

            # great-circle distance (radians)
            c = np.sum(xyz[i, :] * xyz[j, :])
            dist = np.arccos(min(1.0, max(-1.0, c)))  
            if dist < 1e-12:
                continue
                
            wij = w[i] * w[j] / dist

            # ensemble term
            ev = 0.
            for k in range(N):
                ev += np.abs(x[i, k] - x[j, k]) ** p
            ev /= N
            
            # observation term
            o = np.abs(y[i] - y[j]) ** p
            
            num += wij * (o - ev)**2
            den += wij
    return num/den
