from collections import OrderedDict
import logging
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
    ('hdf', ('hdf',))
])

logger = logging.getLogger(__name__)

def _find_file_type(filename):
    '''Guess file type on extension or "tif" if
    filename is directory, default: netcdf'''
    if os.path.isdir(filename):
        ftype = 'tif'
    else:
        this_ext = filename.split('.')[-1]
        for ftype, exts in EXT.items():
            if any(re.search(ext, this_ext, re.IGNORECASE) for ext in exts):
                break
            else:

                ftype = 'netcdf'
    return ftype


def load_array(filename, meta=None, band_specs=None, reader=None):
    '''Create ElmStore from HDF4 / 5 or NetCDF files or TIF directories

    Parameters:

        filename:   filename (HDF4 / 5 or NetCDF) or
                    directory name (TIF)
        meta:       meta data from "filename" already loaded
        band_specs: list of strings or elm.readers.BandSpec objects
        reader:     named reader from elm.readers - one of:
                     ('tif', 'hdf4', 'hdf5', 'netcdf')

    Returns:
        es:         ElmStore (xarray.Dataset) with bands specified
                    by band_specs as DataArrays in "data_vars" attribute
    '''
    ftype = reader or _find_file_type(filename)
    if meta is None:
        if ftype == 'tif':
            meta = _load_meta(filename, ftype, band_specs=band_specs)
        else:
            meta = _load_meta(filename, ftype)
    if ftype == 'netcdf':
        return load_netcdf_array(filename, meta, band_specs=band_specs)
    elif ftype == 'hdf5':
        return load_hdf5_array(filename, meta, band_specs=band_specs)
    elif ftype == 'hdf4':
        return load_hdf4_array(filename, meta, band_specs=band_specs)
    elif ftype == 'tif':
        return load_dir_of_tifs_array(filename, meta, band_specs=band_specs)
    elif ftype == 'hdf':
        try:
            es = load_hdf4_array(filename, meta, band_specs=band_specs)
        except Exception as e:
            logger.info('NOTE: guessed HDF4 type. Failed: {}. \nTrying HDF5'.format(repr(e)))
            es = load_hdf5_array(filename, meta, band_specs=band_specs)
        return es


def _load_meta(filename, ftype, **kwargs):

    if ftype == 'netcdf':
        return load_netcdf_meta(filename, **kwargs)
    elif ftype == 'hdf5':
        return load_hdf5_meta(filename, **kwargs)
    elif ftype == 'hdf4':
        return load_hdf4_meta(filename, **kwargs)
    elif ftype == 'tif':
        return load_dir_of_tifs_meta(filename, **kwargs)
    elif ftype == 'hdf':
        try:
            return load_hdf4_meta(filename, **kwargs)
        except Exception as e:
            logger.info('NOTE: guessed HDF4 type. Failed: {}. \nTrying HDF5'.format(repr(e)))
            return load_hdf5_meta(filename, **kwargs)


def load_meta(filename, **kwargs):
    '''Load metadata for a HDF4 / HDF5 or NetCDF file or TIF directory

    Parameters:
        filename:       filename (HDF4 / 5 and NetCDF) or directory (TIF)
        kwargs:         keyword args that may include "band_specs",
                        a list of string band names or elm.readers.BandSpec
                        objects

    Returns:
        meta:           dict with the following keys
    '''

    reader = kwargs.get('reader')
    kw = {k: v for k, v in reader.items() if k != 'reader'}
    return _load_meta(filename, ftype, **kw)

