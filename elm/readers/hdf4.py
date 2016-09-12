from collections import OrderedDict
import copy
import gc
import logging

import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import xarray as xr

from elm.config import delayed
from elm.readers.util import (geotransform_to_bounds,
                              geotransform_to_coords,
                              row_col_to_xy,
                              raster_as_2d,
                              Canvas,
                              add_es_meta)

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
    meta = {
             'meta': f.GetMetadata(),
             'band_meta': band_metas,
             'sub_datasets': sds,
             'height': dat0.RasterYSize,
             'width':  dat0.RasterXSize,
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
                    band_order_info.append((idx, band_meta, s, band_spec.name))
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
    for _, band_meta, s, name in band_order_info:
        attrs = copy.deepcopy(meta)
        attrs.update(copy.deepcopy(band_meta))
        dat0 = gdal.Open(s[0], GA_ReadOnly)
        raster = raster_as_2d(dat0.ReadAsArray())
        attrs['geo_transform'] = dat0.GetGeoTransform()
        coord_x, coord_y = geotransform_to_coords(dat0.RasterXSize,
                                            dat0.RasterYSize,
                                            attrs['geo_transform'])

        canvas = Canvas(geo_transform=dat0.GetGeoTransform(),
                        xsize=dat0.RasterXSize,
                        ysize=dat0.RasterYSize,
                        dims=native_dims,
                        xbounds=(coord_x[0], coord_x[-1]),
                        ybounds=(coord_y[0], coord_y[-1]),
                        ravel_order='C')
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
