import numpy as np
from numba import njit
from dask import delayed
import dask.array as da

@delayed
def comp_surrogates_hist_one_lead_time(original):
    '''equivalent to shuffling member by member, faster than equivalent njitted function using a for loop with np.random.permutation'''
    generator = np.random.default_rng()
    vectorized = original.reshape(original.shape[0], original.shape[1] * original.shape[2])
    surrogates = generator.permuted(vectorized, axis = 1)
    surrogates = surrogates.reshape(original.shape)
    return surrogates

def adjust_PS(surrogate, target):
    FFT_surrogate = np.fft.rfft2(surrogate)
    phase_surrogate = FFT_surrogate/np.abs(FFT_surrogate)
    FFT_surrogate = np.abs(np.fft.rfft2(target)) * phase_surrogate
    surrogate = np.fft.irfft2(FFT_surrogate).real
    return surrogate
    
@njit
def adjust_hist(surrogate, target):
    out = np.full(surrogate.shape, np.nan).flatten()
    argsorting = np.argsort(surrogate.flatten())
    out[argsorting] = np.sort(target.flatten())
    out = out.reshape(surrogate.shape)
    return out

def comp_single_random_member_maaft(mean, member, iterations):
    '''creates a random member using a modified version of the amplitude adjusted fourier transform (maaft). The original AAFT is described here https://npg.copernicus.org/articles/13/321/2006/, even though this is not the original publication
    mean is the ensemble mean while member is the member and the mean
    '''
    surrogate = np.random.permutation(member)
    
    for i in range(iterations):
        # compute the surrogate_vec
        surrogate_vec = surrogate - mean

        # adjust the surrogate_vec so that it has the same power spectrum than member_vec (which is member - mean)
        surrogate_vec = adjust_PS(surrogate_vec, member - mean)
        
        # construct the surrogate
        surrogate = mean + surrogate_vec

        # match the distribution of values of the surrogate with that of the member
        surrogate = adjust_hist(surrogate, member)
    
    return surrogate

@delayed
def comp_random_members_maaft_one_lead_time(original, iterations = 30):
    '''original has shape (member,) + spatial_shape'''
    surrogates = np.full(original.shape, np.nan)
    mean = np.mean(original, axis = 0)

    for m in range(len(original)):
        surrogates[m] = comp_single_random_member_maaft(mean, original[m], iterations)

    return surrogates

@delayed
def comp_random_members_spectral_one_lead_time(original):
    '''members has shape (member,) + spatial_shape'''
    surrogates = np.full(original.shape, np.nan)
    mean = np.mean(original, axis = 0)
    
    for m in range(len(original)):
        surrogate = np.random.permutation(original[m])
        surrogate_vec = surrogate - mean
        surrogate_vec = adjust_PS(surrogate_vec, original[m] - mean)
        surrogates[m] = mean + surrogate_vec

    return surrogates
'''
@delayed
def comp_random_members_spectral_one_lead_time(members):
    'members has shape (member,) + spatial_shape'
    random_vec = np.full(members.shape, np.nan)
    mean = np.mean(members, axis = 0)
    
    for m in range(len(members)):
        random_vec[m] = generate_random(members[m] - mean)

    return mean + random_vec
'''