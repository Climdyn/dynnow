import xarray as xr
from dynnow.ensemble import Ensemble
import numpy as np
from tqdm import tqdm
import dask.array as da

def full_analysis(original, lightened = True):

    maaft = original.generate_surrogates('maaft')
    spectral = original.generate_surrogates('spectral')
    hist = original.generate_surrogates('hist')

    original.spectral_analyze()

    original.ensemble_scores()
    maaft.ensemble_scores()
    spectral.ensemble_scores()
    hist.ensemble_scores()

    original.comp_FSS()
    maaft.comp_FSS()
    spectral.comp_FSS()
    hist.comp_FSS()

    original.comp_histos_projection_eigenvec()
    maaft.comp_histos_projection_eigenvec()
    spectral.comp_histos_projection_eigenvec()
    hist.comp_histos_projection_eigenvec()

    original = lighten(original)
    maaft = lighten(maaft)
    spectral = lighten(spectral)
    hist = lighten(hist)

    dt = xr.DataTree(xr.Dataset(original.coords), children = {'original': xr.DataTree(original),
                                                              'maaft': xr.DataTree(maaft),
                                                              'spectral': xr.DataTree(spectral),
                                                              'hist': xr.DataTree(hist)
                                                             })
    
    return dt

def lighten(ensemble, to_drop = ('nowcasts', 'observation', 'ensemble_mean', 'error', 'eigenvec')):
    return ensemble.drop_vars(to_drop)

def compute_bias(predictions, observations):
    '''predictions has shape (central_time, lead_time, member, y, x) and events has shape (central_time, lead time, y, x)'''
    
    mean = da.mean(predictions.data, axis = 2)
    bias = da.mean(observations.data - mean, axis = 0)
    return bias # shape (lead_time, y, x)

def events_loop(predictions, observations):

    print('Computing bias')
    bias = compute_bias(predictions, observations).compute()
    
    dt = []

    for ct in range(len(observations.central_time)):
        original = Ensemble(predictions.data[ct], observations.data[ct], np.arange(1, 21) * 5, bias = bias)
        dt.append(full_analysis(original).compute())
        
    return xr.concat(dt, dim = observations.central_time)

    