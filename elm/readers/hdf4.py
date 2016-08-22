from collections import OrderedDict
import gc

import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import xarray as xr

from elm.config import delayed
from elm.readers.util import (geotransform_to_bounds,
                              geotransform_to_dims,
                              row_col_to_xy,
                              raster_as_2d)


def load_hdf4_meta(datafile):
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = f.GetSubDatasets()

    dat0 = gdal.Open(sds[0][0], GA_ReadOnly)
    band_metas = []
    for s in sds:
        f2 = gdal.Open(s[0], GA_ReadOnly)
        band_metas.append(f2.GetMetadata())
    meta = {
             'MetaData': f.GetMetadata(),
             'BandMetaData': band_metas,
             'SubDatasets': sds,
             'Height': dat0.RasterYSize,
             'Width':  dat0.RasterXSize,
             'Name': datafile,
            }
    return meta


def load_hdf4_array(datafile, meta, band_specs=None):
    from elm.sample_util.elm_store import ElmStore
    from elm.sample_util.band_selection import match_meta
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = meta['SubDatasets']
    band_metas = meta['BandMetaData']
    band_order_info = []
    if band_specs:
        for band_meta, s in zip(band_metas, sds):
            for idx, band_spec in enumerate(band_specs):
                name = match_meta(band_meta, band_spec)
                if name:
                    band_order_info.append((idx, band_meta, s, name))
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
    meta['BandOrder'] = []
    for _, band_meta, s, name in band_order_info:
        dat0 = gdal.Open(s[0], GA_ReadOnly)
        raster = raster_as_2d(dat0.ReadAsArray())
        band_meta['GeoTransform'] = dat0.GetGeoTransform()
        coord_x, coord_y = geotransform_to_dims(dat0.RasterXSize,
                                            dat0.RasterYSize,
                                            band_meta['GeoTransform'])
        band_meta['Bounds'] = geotransform_to_bounds(dat0.RasterXSize,
                                                     dat0.RasterYSize,
                                                     band_meta['GeoTransform'])

        elm_store_data[name] = xr.DataArray(raster,
                               coords=[('y', coord_y),
                                       ('x', coord_x),
                                       ],
                               dims=native_dims,
                               attrs=band_meta)
        meta['BandOrder'].append(name)
    del dat0
    gc.collect()
    return ElmStore(elm_store_data, attrs=meta)
