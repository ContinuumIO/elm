import gc
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr

from elm.sample_util.band_selection import match_meta
from elm.readers.util import (geotransform_to_dims,
                              geotransform_to_bounds)
from elm.preproc.elm_store import ElmStore
logger = logging.getLogger(__name__)

def load_tif_meta(filename):
    r = rio.open(filename)
    meta = {'MetaData': {}}
    meta['GeoTransform'] = r.get_transform()
    meta['Bounds'] = r.bounds
    return r, meta, r.meta

def ls_tif_files(dir_of_tiffs):
    tifs = os.listdir(dir_of_tiffs)
    tifs = [f for f in tifs if f.lower().endswith('.tif') or f.lower.endswith('.tiff')]
    return [os.path.join(dir_of_tiffs, t) for t in tifs]

def load_dir_of_tifs_meta(dir_of_tiffs):
    tifs = ls_tif_files(dir_of_tiffs)
    meta = {'Metadata': {}}
    band_metas = []
    for tif in tifs:
        raster, m, band_meta = load_tif_meta(tif)
        meta.update(m)
        band_metas.append(band_meta)
        band_metas[-1]['name'] = tif
    meta['BandMetaData'] = band_metas
    meta['Height'] = raster.height
    meta['Width'] = raster.width

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
    keeping = []
    tifs = ls_tif_files(dir_of_tiffs)
    logger.info('Load tif files from {}'.format(dir_of_tiffs))
    for band_meta in meta['BandMetaData']:
        filename = band_meta['name']
        for idx, band_spec in enumerate(band_specs):
            band_name = match_meta(band_meta, band_spec)
            if band_name:
                keeping.append((idx, band_meta, filename, band_name))
                break
    if len(keeping) != len(band_specs):
        raise ValueError('Failure to find all bands specified by '
                         'band_specs with length {}.\n'
                         'Found only {} of '
                         'them.'.format(len(band_specs), len(keeping)))
    keeping.sort(key=lambda x:x[0])
    if not len(keeping):
        raise ValueError('No matching bands with band_specs {}'.format(band_specs))

    idx, _, filename, band_name = keeping[0]
    _, arr = open_prefilter(filename)
    if len(arr.shape) == 3:
        if arr.shape[0] == 1:
            xy_shape = arr.shape[1:]
        else:
            raise ValueError('Did not expect 3-d TIF unless singleton in 0 or 2 dimension')
    shp = (len(keeping),) + xy_shape
    store = np.empty(shp, dtype=arr.dtype)
    store[0, :, :] = arr
    if len(keeping) > 1:
        for idx, _, filename, band_name in keeping[1:]:
            handle, raster = open_prefilter(filename)
            store[idx, :, :] = raster
            del raster
            gc.collect()
    band_labels = [_[-1] for _ in band_specs]
    longitude, latitude = geotransform_to_dims(handle.width, handle.height, meta['GeoTransform'])
    band_data = xr.DataArray(store,
                           coords=[('band', band_labels),
                                   ('latitude', latitude),
                                   ('longitude', longitude),
                                   ],
                           dims=['band','lat','long',],
                           attrs=meta)

    return ElmStore({'sample': band_data})