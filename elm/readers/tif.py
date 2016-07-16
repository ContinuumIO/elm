import gc
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr

from elm.sample_util.band_selection import match_meta
from elm.readers.util import (geotransform_to_dims,
                              bands_share_coords,
                              SPATIAL_KEYS)
from elm.preproc.elm_store import ElmStore
logger = logging.getLogger(__name__)

def load_tif_meta(filename):
    r = rio.open(filename)
    meta = {'MetaData': r.meta}
    meta['GeoTransform'] = r.get_transform()
    meta['Bounds'] = r.bounds
    meta['Height'] = r.height
    meta['Width'] = r.width
    meta['Name'] = filename
    return r, meta

def ls_tif_files(dir_of_tiffs):
    tifs = os.listdir(dir_of_tiffs)
    tifs = [f for f in tifs if f.lower().endswith('.tif') or f.lower.endswith('.tiff')]
    return [os.path.join(dir_of_tiffs, t) for t in tifs]

def load_dir_of_tifs_meta(dir_of_tiffs, band_specs, **meta):
    tifs = ls_tif_files(dir_of_tiffs)
    meta = {'MetaData': meta}
    band_order_info = []
    band_metas = []
    for tif in tifs:
        raster, band_meta = load_tif_meta(tif)
        for idx, band_spec in enumerate(band_specs):
            band_name = match_meta(band_meta, band_spec)
            if band_name:
                band_order_info.append((idx, tif, band_name))
                band_metas.append((idx, band_meta))
                break
    if not band_order_info or len(band_order_info) != len(band_specs):
        raise ValueError('Failure to find all bands specified by '
                         'band_specs with length {}.\n'
                         'Found only {} of '
                         'them.'.format(len(band_specs), len(band_order_info)))
    # error if they do not share coords at this point
    band_order_info.sort(key=lambda x:x[0])
    band_metas.sort(key=lambda x:x[0])
    band_metas = [b[1] for b in band_metas]
    meta['BandMetaData'] = band_metas
    meta['BandOrderInfo'] = band_order_info
    if bands_share_coords(band_metas, raise_error=False):
        meta['MetaData'].update(band_metas[0]['MetaData'])
        for key in SPATIAL_KEYS:
            meta[key] = band_metas[0][key]
    return meta

def open_prefilter(filename):
    '''Placeholder for future operations on open file handle
    like resample / aggregate '''
    try:
        r = rio.open(filename)
        return r, r.read()
    except Exception as e:
        logger.info('Failed to rasterio.open {}'.format(filename))
        raise

def load_dir_of_tifs_array(dir_of_tiffs, meta, band_specs):
    band_order_info = meta['BandOrderInfo']
    tifs = ls_tif_files(dir_of_tiffs)
    logger.info('Load tif files from {}'.format(dir_of_tiffs))
    bands_share_coords(meta['BandMetaData'], raise_error=True)

    if not len(band_order_info):
        raise ValueError('No matching bands with band_specs {}'.format(band_specs))

    idx, filename, band_name = band_order_info[0]
    _, arr = open_prefilter(filename)
    if len(arr.shape) == 3:
        if arr.shape[0] == 1:
            yx_shape = arr.shape[1:]
        else:
            raise ValueError('Did not expect 3-d TIF unless singleton in 0 or 2 dimension')
    shp = (len(band_order_info),) + yx_shape
    store = np.empty(shp, dtype=arr.dtype)
    store[0, :, :] = arr
    if len(band_order_info) > 1:
        for idx, filename, band_name in band_order_info[1:]:
            handle, raster = open_prefilter(filename)
            store[idx, :, :] = raster
            del raster
            gc.collect()
    band_labels = [_[-1] for _ in band_specs]
    coords_x, coords_y = geotransform_to_dims(handle.width, handle.height, meta['GeoTransform'])
    band_data = xr.DataArray(store,
                           coords=[('band', band_labels),
                                   ('y', coords_y),
                                   ('x', coords_x),
                                   ],
                           dims=['band','y','x',],
                           attrs=meta)

    return ElmStore({'sample': band_data})