from __future__ import print_function

import logging

from affine import Affine
import netCDF4 as nc
import xarray as xr

from elm.readers.util import (geotransform_to_bounds, add_es_meta,
                              VALID_X_NAMES, VALID_Y_NAMES)
from elm.readers import ElmStore
from elm.sample_util.band_selection import match_meta

__all__ = ['load_netcdf_meta', 'load_netcdf_array']

logger = logging.getLogger(__name__)

def _nc_str_to_dict(nc_str):
    str_list = [g.split('=') for g in nc_str.split(';\n')]
    return dict([g for g in str_list if len(g) == 2])


def _assert_nc_attr(nc_dataset, attr_name):
    assert attr_name in nc_dataset.ncattrs(), 'NetCDF Header {} not found'.format(attr_name)


def _get_grid_headers(nc_dataset):
    _assert_nc_attr(nc_dataset, 'Grid.GridHeader')
    return _nc_str_to_dict(nc_dataset.getncattr('Grid.GridHeader'))


def _get_nc_attrs(nc_dataset):
    _assert_nc_attr(nc_dataset, 'Grid.GridHeader')
    _assert_nc_attr(nc_dataset, 'HDF5_GLOBAL.FileHeader')
    _assert_nc_attr(nc_dataset, 'HDF5_GLOBAL.FileInfo')

    grid_header = _nc_str_to_dict(nc_dataset.getncattr('Grid.GridHeader'))
    file_header = _nc_str_to_dict(nc_dataset.getncattr('HDF5_GLOBAL.FileHeader'))
    file_info = _nc_str_to_dict(nc_dataset.getncattr('HDF5_GLOBAL.FileInfo'))

    return {**grid_header, **file_header, **file_info}  # PY3 specific - should this change?


def _get_bandmeta(nc_dataset):
    _assert_nc_attr(nc_dataset, 'Grid.GridHeader')
    return _nc_str_to_dict(nc_dataset.getncattr('Grid.GridHeader'))


def _get_geotransform(nc_info):

    # rotation not taken into account
    x_range = (float(nc_info['WestBoundingCoordinate']), float(nc_info['EastBoundingCoordinate']))
    y_range = (float(nc_info['SouthBoundingCoordinate']), float(nc_info['NorthBoundingCoordinate']))

    aform = Affine(float(nc_info['LongitudeResolution']), 0.0, x_range[0],
                   0.0, -float(nc_info['LatitudeResolution']), y_range[1])

    return aform.to_gdal()


def _get_subdatasets(nc_dataset):
    sds = []
    for k in nc_dataset.variables.keys():
        var_obj = nc_dataset.variables[k]
        obj = {d: var_obj.getncattr(d) for d in var_obj.ncattrs()}
        sds.append(obj)
    return sds


def _normalize_coords(ds):
    '''makes sure that output dataset has `x` and `y` coordinates.
    '''

    coord_names = [k for k in ds.coords.keys()]

    x_coord = next((c for c in coord_names if c.lower() in VALID_X_NAMES), None)
    y_coord = next((c for c in coord_names if c.lower() in VALID_Y_NAMES), None)

    if x_coord is None:
        raise ValueError('x coordinate not found within input dataset')
    if y_coord is None:
        raise ValueError('y coordinate not found within input dataset')

    coords = dict(x=ds[x_coord], y=ds[y_coord])
    return coords


def load_netcdf_meta(datafile):
    '''
    loads metadata for NetCDF

    Parameters
    ----------
    datafile - str: Path on disk to NetCDF file

    Returns
    -------
    Dictionary of metadata
    '''
    ras = nc.Dataset(datafile)
    attrs = _get_nc_attrs(ras)
    geotrans = _get_geotransform(attrs)
    x_size = ras.dimensions['lon'].size  # TODO: remove hardcoded lon variable name
    y_size = ras.dimensions['lat'].size  # TODO: remove hardcoded lat variable name

    meta = {'meta': attrs,
            'band_meta': _get_bandmeta(ras),
            'geo_transform': geotrans,
            'sub_datasets': _get_subdatasets(ras),
            'height': y_size,
            'width': x_size,
            'name': datafile,
            }
    return meta


def load_netcdf_array(datafile, meta, variables):
    '''
    loads metadata for NetCDF

    Parameters
    ----------
    datafile - str: Path on disk to NetCDF file
    meta - dict: netcdf metadata object
    variables - dict<str:str>, list<str>: list of variables to load

    Returns
    -------
    ElmStore xarray.Dataset
    '''
    logger.debug('load_netcdf_array: {}'.format(datafile))
    ds = xr.open_dataset(datafile)

    if isinstance(variables, dict):
        data = { k: ds[v] for k, v in variables.items() }

    if isinstance(variables, (list, tuple)):
        data = { v: ds[v] for v in variables }
    new_es = ElmStore(data,
                    coords=_normalize_coords(ds),
                    attrs=meta)
    for band in new_es.data_vars:
        getattr(new_es, band).attrs['geo_transform'] = meta['geo_transform']
    return new_es