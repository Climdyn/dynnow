from dynnow.analysis import events_loop
import xarray as xr
import numpy as np
from dask.distributed import Client

if __name__ == '__main__':

    client = Client(n_workers = 3, threads_per_worker = 1, memory_limit = '15GiB')
    print(client.dashboard_link)

    evaluation_mask = xr.open_dataarray('../data_events/evaluation_mask.nc')
    
    ######### Stratiform events ############
    predictions = xr.open_dataarray('data/predictions_stratiform1.zarr',
                                    chunks = {'central_time': 1, 'lead_time': 1, 'member': -1, 'y': -1, 'x': -1}).where(evaluation_mask, drop = True)
    events = xr.open_dataset('../data_events/stratiform_events.nc',
                             chunks = {'central_time': 1, 'lead_time': 1, 'y': -1, 'x': -1}).precip_intensity_EDK.where(evaluation_mask, drop = True)
    observations = events.isel(lead_time = np.arange(4, 24))
    
    dt = events_loop(predictions, observations)
    dt.to_netcdf('data/results_stratiform_events1.nc')

    ######### Convective events ############
    predictions = xr.open_dataarray('data/predictions_convective1.zarr',
                                    chunks = {'central_time': 1, 'lead_time': 1, 'member': -1, 'y': -1, 'x': -1}).where(evaluation_mask, drop = True)
    events = xr.open_dataset('../data_events/convective_events.nc',
                             chunks = {'central_time': 1, 'lead_time': 1, 'y': -1, 'x': -1}).precip_intensity_EDK.where(evaluation_mask, drop = True)
    observations = events.isel(lead_time = np.arange(4, 24))
    
    dt = events_loop(predictions, observations)
    dt.to_netcdf('data/results_convective_events1.nc')