from collections import namedtuple, OrderedDict
import logging

import gdal
import numpy as np
import ogr
from rasterio.coords import BoundingBox
import scipy.interpolate as spi

__all__ = ['Canvas', 'CANVAS_FIELDS', 'xy_to_row_col', 'row_col_to_xy',
           'geotransform_to_coords', 'geotransform_to_bounds',
           'canvas_to_coords', 'VALID_X_NAMES', 'VALID_Y_NAMES',
           'xy_canvas']
logger = logging.getLogger(__name__)

SPATIAL_KEYS = ('height', 'width', 'geo_transform', 'bounds')

CANVAS_FIELDS = ('geo_transform',
                 'ysize',
                 'xsize',
                 'zsize',
                 'tsize',
                 'dims',
                 'ravel_order',
                 'zbounds',
                 'tbounds',
                 'bounds')

Canvas = namedtuple('Canvas', CANVAS_FIELDS)

VALID_X_NAMES = ('lon','longitude', 'x') # compare with lower-casing
VALID_Y_NAMES = ('lat','latitude', 'y') # same comment

#def serialize_canvas(canvas):
#    vals = [item if not isinstance(item, Sequence) else list(item)
#            for item in canvas]
#
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
        geo_trans.add(tuple(band_meta['geo_transform']))
        heights.add(int(band_meta['height']))
        widths.add(int(band_meta['width']))

    if len(geo_trans) > 1 or len(heights) > 1 or len(widths) > 1:
        if raise_error:
            raise ValueError('Cannot build xarray data structure when '
                             'bands in bands_specs do not all have the same '
                             'height, width, and geo_transform')
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
    dims = canvas.dims
    x, y = geotransform_to_coords(canvas.xsize, canvas.ysize,
                                  canvas.geo_transform)
    dims2 = []
    label_y, label_x = 'y', 'x'
    for d in dims:
        if d.lower() in VALID_X_NAMES:
            label_x = d
            break
    for d in dims:
        if d.lower() in VALID_Y_NAMES:
            label_y = d
            break
    coords = [(label_y, y), (label_x, x)]
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
    coords = OrderedDict((d, coords[d]) for d in dims)
    if any(coords[d] is None for d in dims):
        raise ValueError('coords.keys(): {} is not '
                         'inclusive of all dims'.format(coords.keys(), dims))
    return coords


def add_es_meta(es):
    band_order = getattr(es, 'band_order', sorted(es.data_vars))
    es.attrs['band_order'] = band_order
    if tuple(es.data_vars.keys()) != ('flat',):
        _add_canvases(es)


def _extract_valid_xy(band_arr):
    x = xname = None
    for name in VALID_X_NAMES:
        x = getattr(band_arr, name, getattr(band_arr, name.lower(), None))
        if x is not None:
            break
    #if x is None:
     #   raise ValueError('Band DataArray does not have an X dimension with a name in {}'.format(VALID_X_NAMES))
    if x is not None:
        xname = name
    y = yname = None
    for name in VALID_Y_NAMES:
        y = getattr(band_arr, name, None)
        if y is not None:
            break
    #if y is None:
     #   raise ValueError('Band DataArray does not have an Y dimension with a name in {}'.format(VALID_Y_NAMES))
    if y is not None:
        yname = name
    return x, name, y, yname


def _add_canvases(es):

    canvas_hashes = set()

    for band in es.data_vars:
        if band == 'flat':
            continue
        band_arr = getattr(es, band)
        x, xname, y, yname = _extract_valid_xy(band_arr)
        z = getattr(band_arr, 'z', None)
        t = getattr(band_arr, 't', None)
        if x is not None:
            xsize = x.size
        else:
            xsize = None
        if y is not None:
            ysize = y.size
        else:
            ysize = None
        if z is not None:
            zsize = z.size
            zbounds = [np.min(z), np.max(z)]
        else:
            zsize = zbounds = None
        if t is not None:
            tsize = t.size
            tbounds = [np.min(t), np.max(t)]
        else:
            tsize = tbounds = None
        canvas = getattr(band_arr, 'canvas', getattr(es, 'canvas', None))
        if canvas is not None:
            geo_transform = canvas.geo_transform
        else:
            geo_transform = band_arr.attrs.get('geo_transform', None)
            if geo_transform is None:
                geo_transform = getattr(band_arr, 'geo_transform', getattr(es, 'geo_transform'))
        band_arr.attrs['canvas'] = Canvas(**OrderedDict((
            ('geo_transform', geo_transform),
            ('ysize', ysize),
            ('xsize', xsize),
            ('zsize', zsize),
            ('tsize', tsize),
            ('dims', band_arr.dims),
            ('ravel_order',getattr(es, 'ravel_order', 'C')),
            ('zbounds', zbounds),
            ('tbounds', tbounds),
            ('bounds', geotransform_to_bounds(xsize, ysize, geo_transform)),
        )))
        canvas_hashes.add(hash((tuple(x) if isinstance(x, Sequence) else x)
                                     for x in band_arr.attrs['canvas']))

    if len(canvas_hashes) == 1:
        es.attrs['canvas'] = band_arr.canvas
        logger.info('Bands share coordinates')
        es.attrs['bounds'] = band_arr.canvas.bounds

def xy_canvas(geo_transform, xsize, ysize, dims, ravel_order='C'):
    return Canvas(**OrderedDict((
        ('geo_transform', geo_transform),
        ('ysize', ysize),
        ('xsize', xsize),
        ('zsize', None),
        ('tsize', None),
        ('dims', dims),
        ('ravel_order', ravel_order),
        ('zbounds', None),
        ('tbounds', None),
        ('bounds', geotransform_to_bounds(xsize, ysize, geo_transform)),
    )))
