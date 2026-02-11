from dynnow.utils import predict_loop
import xarray as xr
from forecast import ldcast_forecast
from dask.distributed import Client

if __name__ == '__main__':
    
    LDCast_mask = xr.open_dataarray('../data_events/LDCast_mask.nc')
    
    ######### Convective events #############
    convective_events = xr.open_dataarray('../data_events/convective_events.nc').where(LDCast_mask, drop = True)

    ensemble_members = 50
    encoding = {'predictions': {'chunks': (1, 1, ensemble_members) + convective_events.shape[-2:]},
                'central_time': {'units': "minutes since 2017-01-01 01:00:00"}
               }
    predictions = predict_loop('data/predictions_convective1.zarr', convective_events, ldcast_forecast, ensemble_members = ensemble_members, encoding = encoding)

    ######### Stratiform events #############
    stratiform_events = xr.open_dataarray('../data_events/stratiform_events.nc').where(LDCast_mask, drop = True)

    ensemble_members = 50
    encoding = {'predictions': {'chunks': (1, 1, ensemble_members) + stratiform_events.shape[-2:]},
                'central_time': {'units': "minutes since 2017-01-01 01:00:00"}
               }
    predictions = predict_loop('data/predictions_stratiform1.zarr', stratiform_events, ldcast_forecast, ensemble_members = ensemble_members, encoding = encoding)