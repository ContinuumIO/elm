

import xarray as xr

def load_meta(*args, **kwargs):
    return {}

def load_array(filenames, **kwargs):

    return xr.open_mfdata