from collections import namedtuple
import gdal
import numpy as np
import ogr
from rasterio.coords import BoundingBox
import scipy.interpolate as spi


SPATIAL_KEYS = ('Height', 'Width', 'GeoTransform', 'Bounds')

CANVAS_FIELDS = ('GeoTransform',
                 'ysize',
                 'xsize',
                 'zsize',
                 'tsize',
                 'dims',
                 'ravel_order',
                 'zbounds',
                 'tbounds')
Canvas = namedtuple('Canvas', CANVAS_FIELDS)


def xy_to_row_col(x, y, geo_transform):
    ''' Get row and column idx's from x and y where
    x and y are the coordinates matching the upper left
    corner of cell'''
    col = np.int32((x - geo_transform[0]) / geo_transform[1])
    row = np.int32((y - geo_transform[3]) / geo_transform[5])
    return row, col

def row_col_to_xy(row, col, geo_transform):
    '''Return the x, y coords that correspond to the
    upper left corners of cells at row, col'''
    x = (col * geo_transform[1]) + geo_transform[0]
    y = (row * geo_transform[5]) + geo_transform[3]
    return x, y

def geotransform_to_coords(xsize, ysize, geo_transform):
    return row_col_to_xy(np.arange(ysize), np.arange(xsize), geo_transform)

def geotransform_to_bounds(xsize, ysize, geo_transform):
    left, bottom = row_col_to_xy(0, 0, geo_transform)
    right, top = row_col_to_xy(xsize, ysize, geo_transform)
    return BoundingBox(left, bottom, right, top)

def bands_share_coords(band_metas, raise_error=False):
    geo_trans = set()
    heights = set()
    widths = set()
    for band_meta in band_metas:
        geo_trans.add(tuple(band_meta['GeoTransform']))
        heights.add(int(band_meta['Height']))
        widths.add(int(band_meta['Width']))

    if len(geo_trans) > 1 or len(heights) > 1 or len(widths) > 1:
        if raise_error:
            raise ValueError('Cannot build xarray data structure when '
                             'bands in bands_specs do not all have the same '
                             'Height, Width, and GeoTransform')
        return False
    return True

def raster_as_2d(raster):
    if len(raster.shape) == 3:
        if raster.shape[0] == 1:
            return raster[0, :, :]
        else:
            raise ValueError('Did not expect 3-d TIF unless singleton in 0 or 2 dimension')
    elif len(raster.shape) != 2:
        raise ValueError('Expected a raster with shape (y, x) or (1, y, x)')
    return raster


def canvas_to_coords(canvas):
    x, y = geotransform_to_coords(canvas.xsize, canvas.ysize,
                                          canvas.GeoTransform)
    dims = canvas.dims

    coords = [('y', y), ('x', x)]
    if canvas.zbounds is not None and canvas.zsize is not None:
        z = np.linspace(zbounds[0], zbounds[1], zsize)
    elif canvas.zbounds is None and canvas.zsize is None:
        z = None
    else:
        raise ValueError()
    # TODO Refine later for non-numeric time types
    if canvas.tbounds is not None and canvas.tsize is not None:
        t = np.linspace(tbounds[0], tbounds[1], tsize)
    elif canvas.tbounds is None and canvas.tsize is None:
        t = None
    else:
        raise ValueError()
    coords += [('z', z), ('t', t)]
    coords = dict(coords)
    if not all(d in coords for d in dims):
        raise ValueError()
    coords = [(d, coords[d]) for d in dims]
    if any(coords[d] is None for d in dims):
        raise ValueError()
    return coords


def add_band_order(es):
    band_order = getattr(es, 'BandOrder', sorted(es.data_vars))
    es.attrs['BandOrder'] = band_order
    return es


