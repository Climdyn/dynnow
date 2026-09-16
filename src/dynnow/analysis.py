import xarray as xr
import numpy as np
import dask.array as da
import dask

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
        
    def compute(self, client = None):
        '''xr.DataTree objects do not implement well the dask methods (https://github.com/pydata/xarray/issues/9355), so .compute() does not handle well the task dependencies'''
        
        d = self.to_dict()
        if client:
            computed = client.compute(list(d.values())) # not working
        else:
            computed = dask.compute(*d.values())
        computed = dict(zip(d.keys(), computed))
        return type(self).from_dict(computed)

    @classmethod
    def from_netcdf(cls, filename):
        dt = xr.open_datatree(filename)
        children = {}
        for key in dt.keys():
            children[key] = dt[key]
        return cls(dt.coords, children = children)

    
