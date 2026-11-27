'''
Surrogate ensembles. Same algorithms as dynnow.surrogates, but every function takes the members of one
lead time, shape (member, *spatial_shape) (the loop over lead times is done by xr.apply_ufunc(vectorize=True)).
spectral_transform / inverse_spectral_transform transform one field (they are Geometry.to_spectral / to_pixel).
'''

import numpy as np
from numba import njit
import pyshtools as pysh


def comp_surrogates_hist(original):
    '''shuffles the pixels of each member (equivalent to np.random.permutation member by member, but faster)'''
    generator = np.random.default_rng()
    vectorized = original.reshape(original.shape[0], -1)
    surrogates = generator.permuted(vectorized, axis=1)
    return surrogates.reshape(original.shape)


def adjust_PS(surrogate, target, spectral_transform, inverse_spectral_transform):
    '''keeps the phases of surrogate and takes the amplitudes of target'''
    spectral_surrogate = spectral_transform(surrogate)
    phase_surrogate = spectral_surrogate / np.abs(spectral_surrogate)
    spectral_surrogate = np.abs(spectral_transform(target)) * phase_surrogate
    return np.real(inverse_spectral_transform(spectral_surrogate))


@njit
def adjust_hist(surrogate, target):
    out = np.full(surrogate.shape, np.nan).flatten()
    argsorting = np.argsort(surrogate.flatten())
    out[argsorting] = np.sort(target.flatten())
    return out.reshape(surrogate.shape)


def comp_single_random_member_maaft(mean, member, spectral_transform, inverse_spectral_transform, iterations=30):
    '''
    creates a random member using a modified version of the amplitude adjusted fourier transform (maaft).
    The original AAFT is described here https://npg.copernicus.org/articles/13/321/2006/, even though this is not the original publication
    mean is the ensemble mean
    '''
    surrogate = np.random.permutation(member)

    for i in range(iterations):
        # adjust the surrogate_vec so that it has the same power spectrum than member_vec (which is member - mean)
        surrogate_vec = adjust_PS(surrogate - mean, member - mean, spectral_transform, inverse_spectral_transform)
        surrogate = mean + surrogate_vec
        # match the distribution of values of the surrogate with that of the member
        surrogate = adjust_hist(surrogate, member)

    return surrogate


def comp_surrogates_maaft(original, spectral_transform, inverse_spectral_transform, iterations=30):
    surrogates = np.full(original.shape, np.nan, dtype = original.dtype)
    mean = np.mean(original, axis=0)
    for m in range(len(original)):
        surrogates[m] = comp_single_random_member_maaft(mean, original[m], spectral_transform, inverse_spectral_transform, iterations=iterations)
    return surrogates


def comp_surrogates_spectral(original, spectral_transform, inverse_spectral_transform, exact_power=False):
    '''
    (sphere only) each member vector is replaced by a random field with the same power per degree l
    (random 4pi-normalised spherical harmonic coefficients), added to the ensemble mean
    '''
    surrogates = np.full(original.shape, np.nan, dtype = original.dtype)
    mean = np.mean(original, axis=0)
    for m in range(len(original)):
        spectral_member = spectral_transform(original[m] - mean)
        clm = pysh.shclasses.SHCoeffs.from_random(np.sum(spectral_member**2, axis=(0, 2)), exact_power=exact_power).coeffs
        surrogates[m] = mean + inverse_spectral_transform(clm)
    return surrogates


def comp_surrogates_spectral_nowcasting(original, spectral_transform, inverse_spectral_transform):
    '''each member vector is replaced by a shuffled member vector whose power spectrum is adjusted to the original one'''
    surrogates = np.full(original.shape, np.nan, dtype = original.dtype)
    mean = np.mean(original, axis=0)
    for m in range(len(original)):
        surrogate = np.random.permutation(original[m])
        surrogate_vec = adjust_PS(surrogate - mean, original[m] - mean, spectral_transform, inverse_spectral_transform)
        surrogates[m] = mean + surrogate_vec
    return surrogates
