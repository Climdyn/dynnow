import xarray as xr
import numpy as np
import dask.array as da

def comp_principal_angles_one_lead_time(spectral_eigenvec1, spectral_eigenvec2):
    scalar_products = np.tensordot(spectral_eigenvec1, spectral_eigenvec2, axes = [[1, 2, 3], [1, 2, 3]])
    _, sigma, _ = np.linalg.svd(scalar_products)
    return np.arccos(np.clip(sigma, -1, 1))

def comp_principal_angles(ensemble1, ensemble2):
    # do not ask here to compute eigenvectors, they need to be computed outside because I want sometimes to compute principal angles between the 4 first eigenvectors of ensemble
    
    # rename the order dimension, otherwise it fails if 'order' is of different length for ensemble1 and ensemble2
    ensemble1 = ensemble1.rename({'order': 'order1'})
    ensemble2 = ensemble2.rename({'order': 'order2'})
    
    principal_angles = xr.apply_ufunc(comp_principal_angles_one_lead_time,
                               ensemble1.spectral_eigenvec,
                               ensemble2.spectral_eigenvec,
                               dask = 'parallelized',
                               output_dtypes = [ensemble1.spectral_eigenvec.dtype],
                               input_core_dims = [['order1', 'i', 'l', 'm'], ['order2', 'i', 'l', 'm']],
                               output_core_dims = [['order']],
                               dask_gufunc_kwargs = {'output_sizes': {'order': min(len(ensemble1.order1), len(ensemble2.order2))}},
                               vectorize = True)
    return principal_angles

def comp_cosine_member_member(ensemble1, ensemble2):
    if 'spectral_ensemble_mean' not in ensemble1:
        ensemble1.comp_spectral_field('ensemble_mean')
    if 'spectral_ensemble_mean' not in ensemble2:
        ensemble2.comp_spectral_field('ensemble_mean')

    spectral_member_vec1 = ensemble1.spectral_forecast - ensemble1.spectral_ensemble_mean
    spectral_member_vec2 = ensemble2.spectral_forecast - ensemble2.spectral_ensemble_mean
    num = (spectral_member_vec1 * spectral_member_vec2).sum(dim = ('i', 'l', 'm'))
    norm1 = da.sqrt((spectral_member_vec1**2).sum(dim = ('i', 'l', 'm')))
    norm2 = da.sqrt((spectral_member_vec2**2).sum(dim = ('i', 'l', 'm')))
    cosine = num / (norm1 * norm2)
    
    return cosine