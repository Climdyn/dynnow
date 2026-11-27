'''
Comparison of two ensembles (e.g. two models, or an ensemble and its surrogates).

comp_cross_gram computes the inner products <x1_i, x2_j> between the member vectors of the two ensembles
(each centered on its own ensemble mean); the comparisons are then linear algebra on this matrix and on
the eigenvector coefficients of each ensemble.
The two ensembles must have the same geometry and the same space.
'''

import numpy as np
import xarray as xr

from dynnow import linalg


def comp_cross_gram(ensemble1, ensemble2):
    '''
    returns a Dataset with
    - cross_gram (lead_time, member, member1): <x1_i, x2_j>, member indexes ensemble1 and member1 ensemble2
    - cross_gram_by_scale (lead_time, l or k, member, member1) in spectral space
    '''
    assert ensemble1.geometry is ensemble2.geometry, 'the two ensembles must share the same geometry object'
    assert ensemble1.space == ensemble2.space
    geometry = ensemble1.geometry
    spatial_dims = geometry.spatial_dims

    vectors1 = ensemble1.member_vec
    vectors2 = ensemble2.member_vec.rename(member='member1')
    input_core_dims = [['member'] + spatial_dims, ['member1'] + spatial_dims]

    if ensemble1.space == 'pixel':
        cross_gram = xr.apply_ufunc(geometry.comp_pixel_gram_one_lead_time,
                                    vectors1, vectors2,
                                    input_core_dims=input_core_dims,
                                    output_core_dims=[['member', 'member1']],
                                    dask='parallelized',
                                    output_dtypes=[np.float64])
        return xr.Dataset({'cross_gram': cross_gram})

    PS_dim = geometry.PS_dim
    cross_gram_by_scale = xr.apply_ufunc(geometry.comp_scale_gram_one_lead_time,
                                         vectors1, vectors2,
                                         input_core_dims=input_core_dims,
                                         output_core_dims=[[PS_dim, 'member', 'member1']],
                                         dask='parallelized',
                                         output_dtypes=[np.float64],
                                         dask_gufunc_kwargs={'output_sizes': {PS_dim: len(geometry.PS_coord)}})
    cross_gram_by_scale = cross_gram_by_scale.assign_coords({PS_dim: geometry.PS_coord, 'scale': (PS_dim, geometry.scales)})
    return xr.Dataset({'cross_gram': cross_gram_by_scale.sum(PS_dim), 'cross_gram_by_scale': cross_gram_by_scale})


def comp_principal_angles(ensemble1, ensemble2, cross_gram, n_order=None):
    '''
    principal angles between the subspaces spanned by the first n_order eigenvectors of each ensemble
    (all of them by default), (lead_time, order)
    cross_gram: the DataArray comp_cross_gram(ensemble1, ensemble2).cross_gram
    '''
    Q1 = ensemble1.intermediates.eigenvec_coefficients
    Q2 = ensemble2.intermediates.eigenvec_coefficients
    if n_order is not None:
        Q1 = Q1.isel(order=slice(0, n_order))
        Q2 = Q2.isel(order=slice(0, n_order))
    return linalg.comp_principal_angles(Q1, cross_gram, Q2)


def comp_cosine_member_member(ensemble1, ensemble2, cross_gram):
    '''
    cosine between member vector i of ensemble1 and member vector i of ensemble2 (same number of members), (lead_time, member)
    cross_gram: the DataArray comp_cross_gram(ensemble1, ensemble2).cross_gram
    '''
    norm2_1 = linalg.diagonal(ensemble1.intermediates.gram_members)
    norm2_2 = linalg.diagonal(ensemble2.intermediates.gram_members)
    norm2_2 = norm2_2.assign_coords(member=norm2_1.member.values)   # members are paired by position
    products = linalg.diagonal(cross_gram)
    return products / np.sqrt(norm2_1 * norm2_2)
