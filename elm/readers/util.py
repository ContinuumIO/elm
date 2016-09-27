from collections import namedtuple, OrderedDict, Sequence
from itertools import product
import logging
import numbers
import re

import gdal
import numpy as np
import ogr
from rasterio.coords import BoundingBox
import scipy.interpolate as spi

import attr
from attr.validators import instance_of

from elm.config import import_callable

__all__ = ['Canvas', 'xy_to_row_col', 'row_col_to_xy',
           'geotransform_to_coords', 'geotransform_to_bounds',
           'canvas_to_coords', 'VALID_X_NAMES', 'VALID_Y_NAMES',
           'xy_canvas','dummy_canvas', 'BandSpec',
           'set_na_from_meta']
logger = logging.getLogger(__name__)

SPATIAL_KEYS = ('height', 'width', 'geo_transform', 'bounds')

READ_ARRAY_KWARGS = ('window', 'buf_xsize', 'buf_ysize',)

@attr.s
class Canvas(object):
    geo_transform = attr.ib()
    buf_xsize = attr.ib()
    buf_ysize = attr.ib()
    dims = attr.ib()
    ravel_order = attr.ib(default='C')
    zbounds = attr.ib(default=None)
    tbounds = attr.ib(default=None)
    zsize = attr.ib(default=None)
    tsize = attr.ib(default=None)
    bounds = attr.ib(default=None)


@attr.s
class BandSpec(object):
    search_key = attr.ib()
    search_value = attr.ib()
    name = attr.ib()
    key_re_flags = attr.ib(default=None)
    value_re_flags = attr.ib(default=None)
    buf_xsize = attr.ib(default=None)
    buf_ysize = attr.ib(default=None)
    window = attr.ib(default=None)
    meta_to_geotransform = attr.ib(default=None)
    stored_coords_order = attr.ib(default=('y', 'x'))

VALID_X_NAMES = ('lon','longitude', 'x') # compare with lower-casing
VALID_Y_NAMES = ('lat','latitude', 'y') # same comment

DEFAULT_GEO_TRANSFORM = (-180, .1, 0, 90, 0, -.1)
#def serialize_canvas(canvas):
#    vals = [item if not isinstance(item, Sequence) else list(item)
#            for item in canvas]
#

def dummy_canvas(buf_xsize, buf_ysize, dims, **kwargs):
    dummy = {'geo_transform': DEFAULT_GEO_TRANSFORM,
             'buf_xsize': buf_xsize,
             'buf_ysize': buf_ysize,
             'dims': dims,}
    dummy.update(kwargs)
    dummy['bounds'] = geotransform_to_bounds(dummy['buf_xsize'],
                                             dummy['buf_ysize'],
                                             dummy['geo_transform'])
    return Canvas(**dummy)

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


def geotransform_to_coords(buf_xsize, buf_ysize, geo_transform):
    return row_col_to_xy(np.arange(buf_ysize), np.arange(buf_xsize), geo_transform)


def geotransform_to_bounds(buf_xsize, buf_ysize, geo_transform):
    left, bottom = row_col_to_xy(0, 0, geo_transform)
    right, top = row_col_to_xy(buf_xsize, buf_ysize, geo_transform)
    return BoundingBox(left, bottom, right, top)


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
    x, y = geotransform_to_coords(canvas.buf_xsize, canvas.buf_ysize,
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


def _extract_valid_xy(band_arr):
    x = xname = None
    for name in VALID_X_NAMES:
        x = getattr(band_arr, name, getattr(band_arr, name.lower(), None))
        if x is not None:
            break
    if x is not None:
        xname = name
    y = yname = None
    for name in VALID_Y_NAMES:
        y = getattr(band_arr, name, getattr(band_arr, name.lower(), None))
        if y is not None:
            break
    if y is not None:
        yname = name
    return x, name, y, yname



def xy_canvas(geo_transform, buf_xsize, buf_ysize, dims, ravel_order='C'):
    return Canvas(**OrderedDict((
        ('geo_transform', geo_transform),
        ('buf_ysize', buf_ysize),
        ('buf_xsize', buf_xsize),
        ('dims', dims),
        ('ravel_order', ravel_order),
        ('bounds', geotransform_to_bounds(buf_xsize, buf_ysize, geo_transform)),
    )))


def window_to_gdal_read_kwargs(**reader_kwargs):
    if 'window' in reader_kwargs:
        window = reader_kwargs['window']
        y, x = map(tuple, window)
        xsize = int(np.diff(x)[0])
        ysize = int(np.diff(y)[0])
        xoff = x[0]
        yoff = y[0]
        r = {'xoff': xoff,
             'yoff': yoff,
             'xsize': xsize,
             'ysize': ysize,}
        r.update({k: v for k, v in reader_kwargs.items()
                  if k != 'window'})
        return r
    return reader_kwargs


def take_geo_transform_from_meta(band_spec=None, required=True, **meta):
    if band_spec and getattr(band_spec, 'meta_to_geotransform', False):
        func = import_callable(band_spec.meta_to_geotransform)
        geo_transform = func(**meta)
        if not isinstance(geo_transform, Sequence) or len(geo_transform) != 6:
            raise ValueError('band_spec.meta_to_geotransform {} did not return a sequence of len 6'.format(band_spec.meta_to_geotransform))
        return geo_transform
    elif required:
        geo_transform = grid_header_to_geo_transform(**meta)
        return geo_transform
    return None

GRID_HEADER_WORDS = ('REGISTRATION', 'BINMETHOD',
                     'LATITUDERESOLUTION', 'LONGITUDERESOLUTION',
                     ('NORTHBOUNDINGCOORD', 'NORTHERNMOSTLAT'),
                     ('SOUTHBOUNDINGCOORD', 'SOUTHERNMOSTLAT'),
                     ('EASTBOUNDINGCOORD', 'EASTERNMOSTLON'),
                     ('WESTBOUNDINGCOORD', 'WESTERNMOSTLON'),
                     'ORIGIN',)

def grid_header_to_geo_transform(**meta):
    '''Unwind an attrs dict, trying to find bounding box words
    that can be used to make a geo_transform object.

    Parameters:
        **meta:  some dict
    Returns:
        geo_transform: tuple
    '''
    grid_header = {}
    for word1, v in meta.items():
        if isinstance(v, dict):
            geo_transform1 = grid_header_to_geo_transform(**v)
            if geo_transform1:
                return geo_transform1
            else:
                continue
        word1 = word1.upper()
        word = None
        for g in GRID_HEADER_WORDS:
            if isinstance(g, tuple):
                if any(gi for gi in g if gi in word1):
                    word = g[0]
            else:
                if g in word1:
                    word = g
        if not word:
            continue
        if "RESOLUTION" in word or "COORD" in word or 'MOSTLAT' in word or 'MOSTLON' in word:
            grid_header[word] = float(v)
        else:
            grid_header[word] = v
    if not len(grid_header) >= 6:
        return None
    lat_res, s, n = (grid_header['LATITUDERESOLUTION'],
           grid_header['SOUTHBOUNDINGCOORD'],
           grid_header['NORTHBOUNDINGCOORD'])
    lon_res, e, w = (grid_header['LONGITUDERESOLUTION'],
           grid_header['EASTBOUNDINGCOORD'],
           grid_header['WESTBOUNDINGCOORD'])
    origin = grid_header.get('ORIGIN', 'NORTHWEST')
    if origin == 'SOUTHWEST':
        geo_transform = (w, lon_res, 0, s, 0, lat_res)
    elif origin == 'NORTHWEST':
        geo_transform = (w, lon_res, 0, n, 0, -lat_res)
    else:
        raise ValueError('Did not expect origin: {}'.format(origin))
    return geo_transform



VALID_RANGE_WORDS = ('^valid[\s\-_]*range',)
INVALID_RANGE_WORDS = ('invalid[\s\-_]*range',)
MISSING_VALUE_WORDS = ('missing[\s\-_]*value', 'invalid[\s\-\_]*value',)

def _case_insensitive_lookup(dic, lookup_list, has_seen):
    for k, pattern in product(dic, lookup_list):
        match = re.search(pattern, k, re.IGNORECASE)
        val = dic[k]
        if match:
            if isinstance(val, numbers.Number) or (isinstance(val, Sequence) and all(isinstance(v, numbers.Number) for v in val)):
                return val
        elif isinstance(val, dict):
            if tuple(val) not in has_seen:
                has_seen.add(tuple(val))
                return _case_insensitive_lookup(val, lookup_list, has_seen)

def extract_valid_range(**attrs):
    return _case_insensitive_lookup(attrs, VALID_RANGE_WORDS, set())


def extract_missing_value(**attrs):
    return _case_insensitive_lookup(attrs, MISSING_VALUE_WORDS, set())


def extract_invalid_range(**attrs):
    return _case_insensitive_lookup(attrs, INVALID_RANGE_WORDS, set())


def _set_invalid_na(values, invalid):
    invalid = np.array(invalid, dtype=values.dtype)
    if len(invalid) == 2:
        values[(values > invalid[0])&(values < invalid[1])] = np.NaN
    else:
        values[values == invalid] = np.NaN


def _set_na_from_valid_range(values, valid_range):
    valid_range = np.array(valid_range, dtype=values.dtype)
    if len(valid_range) == 2:
        values[~((values >= valid_range[0])&(values <= valid_range[1]))] = np.NaN
    else:
        logger.info('Ignoring valid range metadata (does not have length of 2)')



def set_na_from_meta(es, **kwargs):
    attrs = es.attrs
    invalid_range_o = extract_invalid_range(**attrs)
    if invalid_range_o is not None:
        logger.debug('Invalid range {}'.format(invalid_range_o))
        _set_invalid_na(val, invalid_range_o)

    valid_range_o   = extract_valid_range(**attrs)
    if valid_range_o is not None:
        logger.debug('Valid range {}'.format(valid_range_o))
        _set_na_from_valid_range(val, valid_range_o)
    missing_value_o = extract_missing_value(**attrs)
    if missing_value_o is not None:
        logger.debug('Missing value {}'.format(missing_value_o))
        val[val == np.array([missing_value_o], dtype=val.dtype)[0]] = np.NaN
    for idx, band in enumerate(es.data_vars):
        band_arr = getattr(es, band)
        val = band_arr.values
        invalid_range_b = extract_invalid_range(**band_arr.attrs)
        if invalid_range_b is not None:
            logger.debug('Invalid range {}'.format(invalid_range_b))
            _set_invalid_na(val, invalid_range_b)
        valid_range_b = extract_valid_range(**band_arr.attrs)
        if valid_range_b is not None:
            logger.debug('Valid range {}'.format(valid_range_b))
            _set_na_from_valid_range(val, valid_range_b)
        missing_value_b = extract_missing_value(**band_arr.attrs)
        if missing_value_b is not None:
            logger.debug('Missing value {}'.format(missing_value_b))
            val[val == np.array([missing_value_b], dtype=val.dtype)[0]] = np.NaN

