import numpy as np
from numba import njit
from dask import delayed
import dask.array as da
import pyshtools as pysh

def comp_surrogates_hist_one_lead_time(original):
    '''equivalent to shuffling member by member, faster than equivalent njitted function using a for loop with np.random.permutation'''
    original = original.squeeze(axis = 0)
    generator = np.random.default_rng()
    vectorized = original.reshape(original.shape[0], original.shape[1] * original.shape[2])
    surrogates = generator.permuted(vectorized, axis = 1)
    surrogates = surrogates.reshape(original.shape)
    return surrogates[None]

def adjust_PS(surrogate, target, spectral_transform, inverse_spectral_transform):
    FFT_surrogate = spectral_transform(surrogate[None]).squeeze()
    phase_surrogate = FFT_surrogate/np.abs(FFT_surrogate)
    FFT_surrogate = np.abs(spectral_transform(target[None]).squeeze()) * phase_surrogate
    surrogate = inverse_spectral_transform(FFT_surrogate[None]).squeeze().real
    return surrogate
    
@njit
def adjust_hist(surrogate, target):
    out = np.full(surrogate.shape, np.nan).flatten()
    argsorting = np.argsort(surrogate.flatten())
    out[argsorting] = np.sort(target.flatten())
    out = out.reshape(surrogate.shape)
    return out

def comp_single_random_member_maaft(mean, member, spectral_transform, inverse_spectral_transform, iterations = 30):
    '''creates a random member using a modified version of the amplitude adjusted fourier transform (maaft). The original AAFT is described here https://npg.copernicus.org/articles/13/321/2006/, even though this is not the original publication
    mean is the ensemble mean while member is the member and the mean
    '''
    surrogate = np.random.permutation(member)
    
    for i in range(iterations):
        # compute the surrogate_vec
        surrogate_vec = surrogate - mean

        # adjust the surrogate_vec so that it has the same power spectrum than member_vec (which is member - mean)
        surrogate_vec = adjust_PS(surrogate_vec, member - mean, spectral_transform, inverse_spectral_transform)
        
        # construct the surrogate
        surrogate = mean + surrogate_vec

        # match the distribution of values of the surrogate with that of the member
        surrogate = adjust_hist(surrogate, member)
    
    return surrogate

def comp_random_members_maaft_one_lead_time(original, spectral_transform = None, inverse_spectral_transform = None, iterations = 30):
    '''original has shape (member,) + spatial_shape'''
    original = original.squeeze(axis = 0)
    surrogates = np.full(original.shape, np.nan)
    mean = np.mean(original, axis = 0)

    for m in range(len(original)):
        surrogates[m] = comp_single_random_member_maaft(mean, original[m], spectral_transform, inverse_spectral_transform, iterations = iterations)

    return surrogates[None]

def comp_random_members_spectral_one_lead_time(original, spectral_transform = None, inverse_spectral_transform = None, exact_power = False):
    'original has shape (1, member,) + spatial_shape'
    original = original.squeeze(axis = 0)
    surrogates = np.full(original.shape, np.nan)
    mean = np.mean(original, axis = 0)
    
    for m in range(len(original)):
        spectral_member = spectral_transform(original[m] - mean)
        clm = pysh.shclasses.SHCoeffs.from_random(np.sum(spectral_member**2, axis = (0, 2)), exact_power = exact_power).coeffs
        surrogates[m] = mean + inverse_spectral_transform(clm)
        
    return surrogates[None]
    
def comp_random_members_spectral_one_lead_time_nowcasting(original, spectral_transform = None, inverse_spectral_transform = None):
    '''members has shape (1, member,) + spatial_shape'''
    original = original.squeeze(axis = 0)
    surrogates = np.full(original.shape, np.nan)
    mean = np.mean(original, axis = 0)
    
    for m in range(len(original)):
        surrogate = np.random.permutation(original[m])
        surrogate_vec = surrogate - mean
        surrogate_vec = adjust_PS(surrogate_vec, original[m] - mean, spectral_transform, inverse_spectral_transform)
        surrogates[m] = mean + surrogate_vec

    return surrogates[None]
