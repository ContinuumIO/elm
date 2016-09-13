from collections import OrderedDict
import copy
import gc
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr

from elm.sample_util.band_selection import match_meta
from elm.readers.util import (geotransform_to_coords,
                              geotransform_to_bounds,
                              SPATIAL_KEYS,
                              raster_as_2d)
from elm.readers import ElmStore
logger = logging.getLogger(__name__)


__all__ = ['load_tif_meta',
           'load_dir_of_tifs_meta',
           'load_dir_of_tifs_array',]


def load_tif_meta(filename):
    r = rio.open(filename)
    meta = {'meta': r.meta}
    meta['geo_transform'] = r.get_transform()
    meta['bounds'] = r.bounds
    meta['height'] = r.height
    meta['width'] = r.width
    meta['name'] = meta['sub_dataset_name'] = filename
    return r, meta

def ls_tif_files(dir_of_tiffs):
    tifs = os.listdir(dir_of_tiffs)
    tifs = [f for f in tifs if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    return [os.path.join(dir_of_tiffs, t) for t in tifs]

def load_dir_of_tifs_meta(dir_of_tiffs, band_specs=None, **meta):
    tifs = ls_tif_files(dir_of_tiffs)
    meta = copy.deepcopy(meta)
    band_order_info = []
    band_metas = []

    for band_idx, tif in enumerate(tifs):
        raster, band_meta = load_tif_meta(tif)

        if band_specs:
            for idx, band_spec in enumerate(band_specs):
                if match_meta(band_meta, band_spec):
                    band_order_info.append((idx, tif, band_spec.name))
                    band_metas.append((idx, band_meta))
                    break
        else:
            band_name = 'band_{}'.format(band_idx)
            band_order_info.append((band_idx, tif, band_name))
            band_metas.append((band_idx, band_meta))

    if not band_order_info or (band_specs and (len(band_order_info) != len(band_specs))):
        raise ValueError('Failure to find all bands specified by '
                         'band_specs with length {}.\n'
                         'Found only {} of '
                         'them.'.format(len(band_specs), len(band_order_info)))
    # error if they do not share coords at this point
    band_order_info.sort(key=lambda x:x[0])
    band_metas.sort(key=lambda x:x[0])
    band_metas = [b[1] for b in band_metas]
    meta['band_meta'] = band_metas
    meta['band_order_info'] = band_order_info
    return meta

def open_prefilter(filename):
    '''Placeholder for future operations on open file rasterio
    handle like resample / aggregate or setting width, height, etc
    on load.  TODO see optional kwargs to rasterio.open'''
    try:
        r = rio.open(filename)
        return r, r.read()
    except Exception as e:
        logger.info('Failed to rasterio.open {}'.format(filename))
        raise

def load_dir_of_tifs_array(dir_of_tiffs, meta, band_specs=None):
    logger.debug('load_dir_of_tifs_array: {}'.format(dir_of_tiffs))
    band_order_info = meta['band_order_info']
    tifs = ls_tif_files(dir_of_tiffs)
    logger.info('Load tif files from {}'.format(dir_of_tiffs))

    if not len(band_order_info):
        raise ValueError('No matching bands with '
                         'band_specs {}'.format(band_specs))
    native_dims = ('y', 'x')
    elm_store_dict = OrderedDict()
    attrs = {'meta': meta}
    attrs['band_order'] = []
    for idx, filename, band_name in band_order_info:
        band_meta = copy.deepcopy({k: v for k, v in meta.items()
                                   if k not in ('band_order_info', 'band_order')})
        handle, raster = open_prefilter(filename)
        raster = raster_as_2d(raster)
        band_meta['geo_transform'] = handle.get_transform()
        coords_x, coords_y = geotransform_to_coords(handle.width, handle.height, band_meta['geo_transform'])
        elm_store_dict[band_name] = xr.DataArray(raster,
                                                 coords=[('y', coords_y),
                                                         ('x', coords_x),],
                                                 dims=native_dims,
                                                 attrs=band_meta)

        attrs['band_order'].append(band_name)
    gc.collect()
    return ElmStore(elm_store_dict, attrs=attrs)
