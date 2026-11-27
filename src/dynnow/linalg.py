'''
Linear algebra on (small) Gram matrices of member vectors.

Notation: x_i (i = 1..M) are the members centered on the ensemble mean, G_ij = <x_i, x_j> their Gram matrix.
The covariance operator C f = 1/(M-1) sum_i x_i <x_i, f> has the same non-zero eigenvalues as G / (M - 1)
(method of snapshots). If G u_k = (M - 1) lambda_k u_k, the normalised eigenvectors of C are
    v_k = sum_i Q_ik x_i,    with Q_ik = u_ik / sqrt((M - 1) lambda_k).
Q is called the eigenvector coefficients. Since G 1 = 0 (centered members), the u_k with lambda_k > 0 are
orthogonal to (1, ..., 1), so that v_k = sum_i Q_ik y_i for any other centering y_i = x_i - c.

All functions take and return DataArrays; the Gram matrices have dims (..., member, member1).
'''

import numpy as np
import xarray as xr


def eigh_one_lead_time(gram):
    '''
    gram: (..., M, M) Gram matrix of the centered members
    returns the eigenvalues (..., M - 1) of the covariance and the eigenvector coefficients Q (..., M, M - 1),
    in decreasing order of the eigenvalues, without the null eigenvalue due to the centering
    '''
    M = gram.shape[-1]
    val, U = np.linalg.eigh(gram / (M - 1))
    val = val[..., ::-1][..., :-1]
    U = U[..., ::-1][..., :-1]
    Q = U / np.sqrt((M - 1) * val)[..., None, :]
    return val, Q


def comp_eigenval_eigenvec_coefficients(gram):
    '''
    eigenvalues of the covariance, dims (..., order), and
    eigenvector coefficients Q, dims (..., member, order): v_k = sum_i Q_ik x_i
    '''
    M = len(gram.member)
    eigenval, Q = xr.apply_ufunc(eigh_one_lead_time, gram,
                                 input_core_dims=[['member', 'member1']],
                                 output_core_dims=[['order'], ['member', 'order']],
                                 dask='parallelized',
                                 output_dtypes=[np.float64, np.float64],
                                 dask_gufunc_kwargs={'output_sizes': {'order': M - 1}})
    return eigenval, Q.assign_coords(member=gram.member.values)


def diagonal(gram):
    '''diagonal of a Gram matrix (..., member, member1) -> (..., member), e.g. the squared norms of the members'''
    diag = xr.apply_ufunc(np.diagonal, gram,
                          kwargs={'axis1': -2, 'axis2': -1},
                          input_core_dims=[['member', 'member1']],
                          output_core_dims=[['member']],
                          dask='parallelized',
                          output_dtypes=[gram.dtype])
    return diag.assign_coords(member=gram.member.values)


def recenter(gram):
    '''
    Gram matrix of the members centered on their own mean (useful for a sub-ensemble):
    <x_i - mean_k x_k, x_j - mean_k x_k> = G_ij - mean_k G_kj - mean_k G_ik + mean_kl G_kl
    '''
    return gram - gram.mean('member') - gram.mean('member1') + gram.mean(['member', 'member1'])


def principal_angles_one_lead_time(products):
    '''products: (..., order1, order2) inner products between two orthonormal bases -> principal angles (..., order)'''
    sigma = np.linalg.svd(products, compute_uv=False)
    return np.arccos(np.clip(sigma, -1., 1.))


def comp_principal_angles(Q1, cross_gram, Q2):
    '''
    principal angles between the subspaces spanned by v1_k = sum_i Q1_ik x1_i and v2_k = sum_j Q2_jk x2_j
    Q1: (..., member, order), Q2: (..., member, order), cross_gram: (..., member, member1) with <x1_i, x2_j>
    returns (..., order) with len(order) = min(len(Q1.order), len(Q2.order))
    '''
    Q1 = Q1.rename(order='order1')
    Q2 = Q2.rename(member='member1', order='order2')
    products = xr.dot(Q1, cross_gram, Q2, dim=['member', 'member1'])                  # <v1_k, v2_l>, (..., order1, order2)
    return xr.apply_ufunc(principal_angles_one_lead_time, products,
                          input_core_dims=[['order1', 'order2']],
                          output_core_dims=[['order']],
                          dask='parallelized',
                          output_dtypes=[np.float64],
                          dask_gufunc_kwargs={'output_sizes': {'order': min(len(Q1.order1), len(Q2.order2))}})
