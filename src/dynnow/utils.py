import xarray as xr
import numpy as np
import dask.array as da
from dask import delayed
import os

def dB_transform_dask(precip):
    '''dB_transform leaving nan's as nan's'''
    res = xr.where(precip > 0.1, 10*np.log10(precip), precip)
    res = xr.where(precip <= 0.1, -15, precip)
    return res

def predict_loop(store, events, forecast_function, ensemble_members = 50, encoding = {}):
    '''events should be already in dBR units if they need to'''
    #predictions = []
    
    for i, ct in enumerate(events.central_time):
        print(f'{i + 1}th event over {len(events.central_time)}')
        
        ev = events.sel(central_time = ct)
        pred = predict_event(ev, forecast_function, ensemble_members)
        predictions = pred
        #predictions.append(pred)

        save_to_zarr(store, predictions, ct.data, encoding = encoding)
    
    #predictions = xr.concat(predictions, dim = events.central_time)

    #return predictions

def save_to_zarr(store, predictions, central_time, encoding = {}):
    predictions = predictions.expand_dims(dim = {'central_time': [central_time]}, axis = 0)

    if os.path.isdir(store):
        predictions.to_zarr(store, mode = 'a', append_dim = 'central_time')
    else:
        predictions.to_zarr(store, mode = 'w-', encoding = encoding)

def predict_event(event, forecast_function, ensemble_members):
    '''events should be already in dBR units if they need to'''
    
    # select inputs
    inputs = event.isel(lead_time = np.arange(4))

    # compute the ensemble nowcast
    '''
    pred = da.from_delayed(delayed(forecast_function)(inputs.data, ensemble_members = ensemble_members), # (lead_time, member, y, x)
                           shape = (20, ensemble_members) + event.shape[-2:],
                           dtype = np.float64)
    '''
    pred = forecast_function(inputs.data, ensemble_members = ensemble_members) # (lead_time, member, y, x)

    pred = xr.DataArray(pred,
                        dims = ('lead_time', 'member', 'y', 'x'),
                        coords = {'lead_time': event.lead_time.isel(lead_time = np.arange(4, 24)).data,
                                  'member': np.arange(1, ensemble_members + 1),
                                  'y': inputs.y.data,
                                  'x': inputs.x.data},
                        name = 'predictions'
                       )
                        
    return pred