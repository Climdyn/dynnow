import xarray as xr
from dynnow.ensemble import Ensemble
import numpy as np
from tqdm import tqdm
import dask.array as da

class Analysis(xr.DataTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        # get the child node
        child = super().__getitem__(key)

        # convert it to an Ensemble object if the 'ensemble' flag is True
        if hasattr(child, 'ensemble') and child.ensemble == 'True':
            child = Ensemble(child)
            
        return child

    def __getattr__(self, name):
        # get the child node
        child = super().__getattr__(name)
        
        # convert it to an Ensemble object if the 'ensemble' flag is True
        if hasattr(child, 'ensemble') and child.ensemble == 'True':
            child = Ensemble(child)
            
        return child

    @classmethod
    def from_netcdf(cls, filename):
        dt = xr.open_datatree(filename)
        children = {}
        for key in dt.keys():
            children[key] = dt[key]
        return cls(dt.coords, children = children)

def full_analysis(original, lightened = True):

    maaft = original.generate_surrogates('maaft')
    spectral = original.generate_surrogates('spectral')
    hist = original.generate_surrogates('hist')

    original.spectral_analyze()

    original.ensemble_scores()
    maaft.ensemble_scores()
    spectral.ensemble_scores()
    hist.ensemble_scores()

    if lightened:
        original = lighten(original)
        maaft = lighten(maaft)
        spectral = lighten(spectral)
        hist = lighten(hist)

    dt = Analysis(xr.Dataset(original.coords), children = {'original': xr.DataTree(original),
                                                            'maaft': xr.DataTree(maaft),
                                                            'spectral': xr.DataTree(spectral),
                                                            'hist': xr.DataTree(hist)
                                                          })
    
    return dt

def lighten(ensemble, to_drop = ('nowcasts', 'observation', 'ensemble_mean', 'error', 'eigenvec', 'bias')):
    return ensemble.drop_vars(to_drop, errors = 'ignore')

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
        ds = xr.Dataset(data_vars = {'nowcasts': (('lead_time', 'member', 'y', 'x'),predictions.isel(central_time = ct).data),
                                     'observation': (('lead_time', 'y', 'x'), observations.isel(central_time = ct).data),
                                     'bias': (('lead_time', 'y', 'x'), bias.data)
                                    },
                        coords = {'lead_time': predictions.lead_time.data})
        original = Ensemble(ds)
        dt.append(full_analysis(original).compute())
        
    return xr.concat(dt, dim = observations.central_time)

    