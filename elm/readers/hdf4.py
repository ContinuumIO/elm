import gc

import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import xarray as xr

from elm.config import delayed
from elm.readers.util import geotransform_to_dims, geotransform_to_bounds

directions = ('North', 'South', 'East', 'West')
def get_subdataset_bounds(meta):
    from elm.sample_util.geo_selection import SpatialBounds
    bounds = SpatialBounds(*tuple(float(meta['{}BoundingCoord'.format(word)])
                                  for word in directions))
    time = meta['StartTime']
    return bounds, time

def load_hdf4_meta(datafile):
    print(datafile)
    f = gdal.Open(datafile)
    sds = f.GetSubDatasets()

    dat0 = gdal.Open(sds[0][0])
    band_metas = []
    for s in sds:
        f2 = gdal.Open(s[0])
        band_metas.append(f2.GetMetadata())
    geo_transform = dat0.GetGeoTransform()
    bounds = geotransform_to_bounds(dat0.RasterXSize, dat0.RasterYSize, geo_transform)
    meta = {
             'MetaData': f.GetMetadata(),
             'BandMetaData': band_metas,
             'GeoTransform':geo_transform,
             'SubDatasets': sds,
             'Bounds': bounds,
            }
    return meta


def load_hdf4_array(datafile, meta, band_specs):
    from elm.preproc.elm_store import ElmStore
    from elm.sample_util.band_selection import match_meta
    f = gdal.Open(datafile)
    sds = meta['SubDatasets']
    band_metas = meta['BandMetaData']
    keeping = []

    for band_meta, s in zip(band_metas, sds):
        for idx, band_spec in enumerate(band_specs):
            name = match_meta(band_meta, band_spec)
            if name:
                keeping.append((idx, band_meta, s, name))
                break

    keeping.sort(key=lambda x:x[0])
    if not len(keeping):
        raise ValueError('No matching bands with band_specs {}'.format(band_specs))

    dat0 = gdal.Open(s[0])
    arr = dat0.ReadAsArray()
    shp = (len(keeping),) + arr.shape
    _, band_meta, s, _ = keeping[0]
    store = np.empty(shp, dtype = arr.dtype)
    store[0, :, :] = arr
    if len(keeping) > 1:
        for idx, _, s, _ in keeping[1:]:
            dat = gdal.Open(s[0]).ReadAsArray()
            store[idx, :, :] = dat
            del dat
            gc.collect()
    band_labels = [_[-1] for _ in band_specs]
    latitude, longitude = geotransform_to_dims(dat0.RasterXSize, dat0.RasterYSize, meta['GeoTransform'])
    band_data = xr.DataArray(store,
                           coords=[('band', band_labels),
                                   ('latitude', latitude),
                                   ('longitude', longitude),
                                   ],
                           dims=['band','lat','long',],
                           attrs=meta)

    return ElmStore({'sample': band_data})