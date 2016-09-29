from collections import OrderedDict
import os
import re

from elm.readers.netcdf import load_netcdf_array, load_netcdf_meta
from elm.readers.hdf4 import load_hdf4_array, load_hdf4_meta
from elm.readers.hdf5 import load_hdf5_array, load_hdf5_meta
from elm.readers.tif import load_dir_of_tifs_meta,load_dir_of_tifs_array

__all__ = ['load_array', 'load_meta']

EXT = OrderedDict([
    ('netcdf', ('nc', 'nc\d',)),
    ('hdf5', ('h5', 'hdf5', 'hd5',)),
    ('hdf4', ('hdf4', 'h4', 'hd4',)),
])

def _find_file_type(filename):
    if os.path.isdir(filename):
        ftype = 'tif'
    else:
        ext = filename.split('.')[-1]
        for ftype, exts in EXT.items():
            if any(re.search(ext, filename, re.IGNORECASE) for ext in exts):
                break
            else:
                ftype = 'netcdf'
    return ftype


def load_array(filename, meta=None, band_specs=None):
    ftype = _find_file_type(filename)
    if meta is None:
        if ftype == 'tif':
            meta = load_meta(filename, band_specs=band_specs)
        else:
            meta = load_meta(filename)
    if ftype == 'netcdf':
        return load_netcdf_array(filename, meta, band_specs=band_specs)
    elif ftype == 'hdf5':
        return load_hdf5_array(filename, meta, band_specs=band_specs)
    elif ftype == 'hdf4':
        return load_hdf4_array(filename, meta, band_specs=band_specs)
    elif ftype == 'tif':
        return load_dir_of_tifs_array(filename, meta, band_specs=band_specs)


def load_meta(filename, **kwargs):
    ftype = _find_file_type(filename)
    if ftype == 'netcdf':
        return load_netcdf_meta(filename, **kwargs)
    elif ftype == 'hdf5':
        return load_hdf5_meta(filename, **kwargs)
    elif ftype == 'hdf4':
        return load_hdf4_meta(filename, **kwargs)
    elif ftype == 'tif':
        return load_dir_of_tifs_meta(filename, **kwargs)
