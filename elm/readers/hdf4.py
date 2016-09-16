from collections import OrderedDict
import copy
import gc
import logging

import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import xarray as xr

from elm.readers.util import (geotransform_to_bounds,
                              geotransform_to_coords,
                              row_col_to_xy,
                              raster_as_2d,
                              Canvas,
                              BandSpec,
                              READ_ARRAY_KWARGS)

__all__ = [
    'load_hdf4_meta',
    'load_hdf4_array',
]

logger = logging.getLogger(__name__)

def load_hdf4_meta(datafile):
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = f.GetSubDatasets()

    dat0 = gdal.Open(sds[0][0], GA_ReadOnly)
    band_metas = []
    for s in sds:
        f2 = gdal.Open(s[0], GA_ReadOnly)
        band_metas.append(f2.GetMetadata())
        band_metas[-1]['sub_dataset_name'] = s[0]
    meta = {
             'meta': f.GetMetadata(),
             'band_meta': band_metas,
             'sub_datasets': sds,
             'name': datafile,
            }
    return meta


def load_hdf4_array(datafile, meta, band_specs=None):

    from elm.readers import ElmStore
    from elm.sample_util.band_selection import match_meta
    logger.debug('load_hdf4_array: {}'.format(datafile))
    f = gdal.Open(datafile, GA_ReadOnly)

    sds = meta['sub_datasets']
    band_metas = meta['band_meta']
    band_order_info = []
    if band_specs:
        for band_meta, s in zip(band_metas, sds):
            for idx, band_spec in enumerate(band_specs):
                if match_meta(band_meta, band_spec):
                    band_order_info.append((idx, band_meta, s, band_spec))
                    break

        band_order_info.sort(key=lambda x:x[0])
        if not len(band_order_info):
            raise ValueError('No matching bands with '
                             'band_specs {}'.format(band_specs))
    else:
        band_order_info = [(idx, band_meta, s, 'band_{}'.format(idx))
                           for idx, (band_meta, s) in enumerate(zip(band_metas, sds))]
    native_dims = ('y', 'x')
    elm_store_data = OrderedDict()

    band_order = []
    for _, band_meta, s, band_spec in band_order_info:
        if isinstance(band_spec, BandSpec):
            name = band_spec.name
            reader_kwargs = {k: getattr(band_spec, k)
                             for k in READ_ARRAY_KWARGS
                             if getattr(band_spec, k)}
        else:
            reader_kwargs = {}
            name = band_spec
        attrs = copy.deepcopy(meta)
        attrs.update(copy.deepcopy(band_meta))
        dat0 = gdal.Open(s[0], GA_ReadOnly)
        band_meta.update(reader_kwargs)
        raster = raster_as_2d(dat0.ReadAsArray(**reader_kwargs))
        attrs['geo_transform'] = dat0.GetGeoTransform()

        geo_transform = dat0.GetGeoTransform()
        ysize, xsize = raster.shape
        coord_x, coord_y = geotransform_to_coords(xsize,
                                                  ysize,
                                                  geo_transform)

        canvas = Canvas(geo_transform=geo_transform,
                        xsize=xsize,
                        ysize=ysize,
                        dims=native_dims,
                        ravel_order='C',
                        bounds=geotransform_to_bounds(xsize, ysize, geo_transform))
        attrs['canvas'] = canvas
        elm_store_data[name] = xr.DataArray(raster,
                               coords=[('y', coord_y),
                                       ('x', coord_x)],
                               dims=native_dims,
                               attrs=attrs)

        band_order.append(name)
    del dat0
    attrs = copy.deepcopy(attrs)
    attrs['band_order'] = band_order
    gc.collect()
    return ElmStore(elm_store_data, attrs=attrs)
