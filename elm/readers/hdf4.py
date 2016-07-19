import gc

import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import xarray as xr

from elm.config import delayed
from elm.readers.util import (geotransform_to_bounds,
                              geotransform_to_dims,
                              row_col_to_xy)


def load_hdf4_meta(datafile):
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = f.GetSubDatasets()

    dat0 = gdal.Open(sds[0][0], GA_ReadOnly)
    band_metas = []
    for s in sds:
        f2 = gdal.Open(s[0], GA_ReadOnly)
        band_metas.append(f2.GetMetadata())
    geo_transform = dat0.GetGeoTransform()
    bounds = geotransform_to_bounds(dat0.RasterXSize,
                                    dat0.RasterYSize,
                                    geo_transform)
    meta = {
             'MetaData': f.GetMetadata(),
             'BandMetaData': band_metas,
             'GeoTransform':geo_transform,
             'SubDatasets': sds,
             'Bounds': bounds,
             'Height': dat0.RasterYSize,
             'Width':  dat0.RasterXSize,
             'Name': datafile,
            }
    return meta


def load_hdf4_array(datafile, meta, band_specs):
    from elm.preproc.elm_store import ElmStore
    from elm.sample_util.band_selection import match_meta
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = meta['SubDatasets']
    band_metas = meta['BandMetaData']
    band_order_info = []

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

    dat0 = gdal.Open(s[0], GA_ReadOnly)
    raster = dat0.ReadAsArray()
    shp = (len(band_specs),) + raster.shape
    _, band_meta, s, _ = band_order_info[0]
    store = np.empty(shp, dtype = raster.dtype)
    store[0, :, :] = raster
    del raster
    gc.collect()
    if len(band_order_info) > 1:
        for idx, _, s, _ in band_order_info[1:]:
            dat = gdal.Open(s[0], GA_ReadOnly).ReadAsArray()
            store[idx, :, :] = dat
            del dat
            gc.collect()
    band_labels = [_[-1] for _ in band_specs]
    coord_x, coord_y = geotransform_to_dims(dat0.RasterXSize,
                                            dat0.RasterYSize,
                                            meta['GeoTransform'])
    band_data = xr.DataArray(store,
                           coords=[('band', band_labels),
                                   ('y', coord_y),
                                   ('x', coord_x),
                                   ],
                           dims=['band','y','x',],
                           attrs=meta)

    return ElmStore({'sample': band_data}, attrs=meta)
