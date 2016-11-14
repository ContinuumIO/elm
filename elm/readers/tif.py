'''
Tools for reading GeoTiff files.  Typically use the interface through

elm.readers.load_array
elm.readers.load_meta

'''
from collections import OrderedDict
import copy
import gc
import logging
import os

import numpy as np
import rasterio as rio
import xarray as xr

from elm.sample_util.metadata_selection import match_meta
from elm.readers.util import (geotransform_to_coords,
                              geotransform_to_bounds,
                              SPATIAL_KEYS,
                              raster_as_2d,
                              READ_ARRAY_KWARGS,
                              take_geo_transform_from_meta,
                              BandSpec)

from elm.readers import ElmStore
logger = logging.getLogger(__name__)


__all__ = ['load_tif_meta',
           'load_dir_of_tifs_meta',
           'load_dir_of_tifs_array',]


def load_tif_meta(filename):
    '''Read the metadata of one TIF file'''
    r = rio.open(filename)
    if r.count != 1:
        raise ValueError('elm.readers.tif only reads tif files with 1 band (shape of [1, y, x]). Found {} bands'.format(r.count))
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


def array_template(r, meta, **reader_kwargs):
    dtype = getattr(np, r.dtypes[0])

    if not 'window' in reader_kwargs:
        if 'height' in reader_kwargs:
            height = reader_kwargs['height']
        else:
            height = meta['height']
        if 'width' in reader_kwargs:
            width = reader_kwargs['width']
        else:
            width = meta['width']
    else:
        if 'height' in reader_kwargs:
            height = reader_kwargs['height']
        else:
            height = np.diff(reader_kwargs['window'][0])
        if 'width' in reader_kwargs:
            width = reader_kwargs['width']
        else:
            width = np.diff(reader_kwargs['window'][0])
    return np.empty((1, height, width), dtype=dtype)


def load_dir_of_tifs_meta(dir_of_tiffs, band_specs=None, **meta):
    '''Load metadata from same-directory GeoTiffs representing
    different bands of the same image.

    Parameters:
        dir_of_tiffs: Directory with GeoTiffs
        band_specs:   List of elm.readers.BandSpec objects
        meta:         included in returned metadata'''
    logger.debug('load_dir_of_tif_meta {}'.format(dir_of_tiffs))
    tifs = ls_tif_files(dir_of_tiffs)
    meta = copy.deepcopy(meta)
    band_order_info = []
    for band_idx, tif in enumerate(tifs):
        raster, band_meta = load_tif_meta(tif)

        if band_specs:
            for idx, band_spec in enumerate(band_specs):
                if (isinstance(band_spec, BandSpec) and match_meta(band_meta, band_spec)) or (isinstance(band_spec, str) and band_spec in tif):
                    band_order_info.append((idx, tif, band_spec, band_meta))
                    break

        else:
            band_name = 'band_{}'.format(band_idx)
            band_order_info.append((band_idx, tif, band_name, band_meta))

    if not band_order_info or (band_specs and (len(band_order_info) != len(band_specs))):
        logger.debug('len(band_order_info) {}'.format(len(band_order_info)))
        raise ValueError('Failure to find all bands specified by '
                         'band_specs with length {}.\n'
                         'Found only {} of '
                         'them.'.format(len(band_specs), len(band_order_info)))
    # error if they do not share coords at this point
    band_order_info.sort(key=lambda x:x[0])
    meta['band_meta'] = [b[-1] for b in band_order_info]
    meta['band_order_info'] = [b[:-1] for b in band_order_info]
    return meta

def open_prefilter(filename, meta, **reader_kwargs):
    '''Placeholder for future operations on open file rasterio
    handle like resample / aggregate or setting width, height, etc
    on load.  TODO see optional kwargs to rasterio.open'''
    try:
        r = rio.open(filename)
        raster = array_template(r, meta, **reader_kwargs)
        logger.debug('reader_kwargs {} raster template shape {}'.format(reader_kwargs, raster.shape))
        r.read(out=raster)
        return r, raster
    except Exception as e:
        logger.info('Failed to rasterio.open {}'.format(filename))
        raise

def load_dir_of_tifs_array(dir_of_tiffs, meta, band_specs=None):
    '''Return an ElmStore where each subdataset is a DataArray

    Parameters:

        dir_of_tiffs: directory of GeoTiff files where each is a
                      single band raster
        meta:     meta from elm.readers.load_dir_of_tifs_meta
        band_specs: list of elm.readers.BandSpec objects,
                    defaulting to reading all subdatasets
                    as bands
    Returns:
        X: ElmStore

    '''

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
    for (idx, filename, band_spec), band_meta in zip(band_order_info, meta['band_meta']):
        band_name = getattr(band_spec, 'name', band_spec)
        if not isinstance(band_spec, str):
            reader_kwargs = {k: getattr(band_spec, k)
                             for k in READ_ARRAY_KWARGS
                             if getattr(band_spec, k)}
        else:
            reader_kwargs = {}
        if 'buf_xsize' in reader_kwargs:
            reader_kwargs['width'] = reader_kwargs.pop('buf_xsize')
        if 'buf_ysize' in reader_kwargs:
            reader_kwargs['height'] = reader_kwargs.pop('buf_ysize')
        if 'window' in reader_kwargs:
            reader_kwargs['window'] = tuple(map(tuple, reader_kwargs['window']))
        multy = band_meta['height'] / reader_kwargs['height']
        multx = band_meta['width'] / reader_kwargs['width']
        band_meta.update(reader_kwargs)
        geo_transform = take_geo_transform_from_meta(band_spec, **attrs)
        handle, raster = open_prefilter(filename, band_meta, **reader_kwargs)
        raster = raster_as_2d(raster)
        if getattr(band_spec, 'stored_coords_order', ['y', 'x'])[0] == 'y':
            rows, cols = raster.shape
        else:
            rows, cols = raster.T.shape
        if geo_transform is None:
            band_meta['geo_transform'] = handle.get_transform()
        else:
            band_meta['geo_transform'] = geo_transform
        band_meta['geo_transform'][1]  *= multx
        band_meta['geo_transform'][-1] *= multy

        coords_x, coords_y = geotransform_to_coords(cols,
                                                    rows,
                                                    band_meta['geo_transform'])
        elm_store_dict[band_name] = xr.DataArray(raster,
                                                 coords=[('y', coords_y),
                                                         ('x', coords_x),],
                                                 dims=native_dims,
                                                 attrs=band_meta)

        attrs['band_order'].append(band_name)
    gc.collect()
    return ElmStore(elm_store_dict, attrs=attrs)
