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
                              Canvas,
                              BandSpec,
                              row_col_to_xy,
                              raster_as_2d,
                              add_es_meta)

from elm.readers import ElmStore 
from elm.sample_util.band_selection import match_meta

__all__ = [
    'load_hdf5_meta',
    'load_hdf5_array',
]

logger = logging.getLogger(__name__)


def _nc_str_to_dict(nc_str):
    str_list = [g.split('=') for g in nc_str.split(';\n')]
    return dict([g for g in str_list if len(g) == 2])


def load_hdf5_meta(datafile):
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = f.GetSubDatasets()
    band_metas = []
    for s in sds:
        f2 = gdal.Open(s[0], GA_ReadOnly)
        bm = dict()
        for k, v in f2.GetMetadata().items():
            vals = _nc_str_to_dict(v)
            bm.update(vals)
        band_metas.append(bm)

    meta = dict()
    for k, v in f.GetMetadata().items():
        vals = _nc_str_to_dict(v)
        meta.update(vals)

    return dict(meta=meta,
                band_meta=band_metas,
                sub_datasets=sds,
                name=datafile)

def load_array(dataset):
    data_file = gdal.Open(dataset)
    raster = raster_as_2d(data_file.ReadAsArray())
    canvas = Canvas(geo_transform=data_file.GetGeoTransform(),
                    xsize=data_file.RasterXSize,
                    ysize=data_file.RasterYSize,
                    zsize=None,
                    tsize=None,
                    zbounds=None,
                    tbounds=None,
                    bounds=None,
                    dims=None,
                    ravel_order='C')

    coord_x, coord_y = geotransform_to_coords(data_file.RasterXSize,
                                        data_file.RasterYSize,
                                        canvas.geo_transform)
    attrs = dict(canvas=canvas)
    coords = [('y', coord_y),('x', coord_x)] 
    dims = ('y', 'x')
    return xr.DataArray(data=raster,
                        coords=coords,
                        dims=dims,
                        attrs=attrs)

def load_hdf5_array(datafile, meta, band_specs):
    logger.debug('load_hdf5_array: {}'.format(datafile))
    f = gdal.Open(datafile, GA_ReadOnly)
    sds = meta['sub_datasets']
    band_metas = meta['band_meta']
    band_order_info = []
    for band_meta, sd in zip(band_metas, sds):
        for idx, bs in enumerate(band_specs):
            if match_meta(band_meta, bs):
                band_order_info.append((idx, band_meta, sd, bs.name))
                break

    band_order_info.sort(key=lambda x:x[0])
    if not len(band_order_info):
        raise ValueError('No matching bands with '
                         'band_specs {}'.format(band_specs))

    elm_store_data = OrderedDict()

    band_order = []
    for _, band_meta, sd, name in band_order_info:
        attrs = copy.deepcopy(meta)
        attrs.update(copy.deepcopy(band_meta))
        subdata = gdal.Open(sd[0], GA_ReadOnly)
        raster = raster_as_2d(subdata.ReadAsArray())

        canvas = Canvas()
        canvas.geo_transform = subdata.GetGeoTransform()
        canvas.ravel_order = 'C'
        import pdb;pdb.set_trace()
        canvas.ysize = subdata.GetGeoTransform()
        canvas.xsize = subdata.GetGeoTransform()
        canvas.zsize = subdata.GetGeoTransform()
        canvas.tsize = subdata.GetGeoTransform()
        canvas.dims = subdata.GetGeoTransform()
        canvas.zbounds = subdata.GetGeoTransform()
        canvas.tbounds = subdata.GetGeoTransform()
        canvas.bounds = subdata.GetGeoTransform()

        attrs['geo_transform'] = dat0.GetGeoTransform()
        coord_x, coord_y = geotransform_to_coords(dat0.RasterXSize,
                                            dat0.RasterYSize,
                                            attrs['geo_transform'])
        elm_store_data[name] = xr.DataArray(raster,
                               coords=[('y', coord_y),
                                       ('x', coord_x),
                                       ],
                               dims=('y','x'),
                               attrs=attrs)

        band_order.append(name)
    del dat0
    attrs = copy.deepcopy(attrs)
    attrs['band_order'] = band_order
    gc.collect()
    return ElmStore(elm_store_data, attrs=attrs)


