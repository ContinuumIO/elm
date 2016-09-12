from itertools import product
from collections import namedtuple
import logging

from numba import njit
import numpy as np

logger = logging.getLogger(__name__)

Spatialbounds = namedtuple('Spatialbounds', 'north south east west')

@njit('int32(float64, float64, float64[:, :])',nogil=True)
def point_in_poly(x, y, poly):
    num = poly.shape[0]
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i, 1] > y) != (poly[j, 1] > y)) and \
            (x < (poly[j, 0] - poly[i, 0]) * (y - poly[i, 1]) / (poly[j, 1] - poly[i, 1]) + poly[i, 0]):
            c = not c
        j = i
    return np.int32(c)

#@njit('''float64[:, :](float64[:], float64[:],float64[:], float64[:],float64[:, :], float64[:, :])''')
# TODO make the njit decorator above work
def _points_in_poly(unique_x, unique_y,
                    all_x_coords, all_y_coords,
                    include_polys_array, exclude_polys_array):
    points = []
    breaks_inc_exc = tuple(map(lambda x: np.where(np.isnan(x[:, 0]))[0],
                 (include_polys_array, exclude_polys_array)))
    keep = (True, False)
    if all_x_coords.size:
        idx_x_y_gen = ((idx, x, y)
                       for (idx, (x, y)) in enumerate(zip(all_x_coords, all_y_coords)))
    else:
        idx_x_y_gen = ((idx, x, y)
                       for (idx, (x, y)) in enumerate(product(unique_x, unique_y)))
    arrs = (include_polys_array, exclude_polys_array)
    for idx, x, y in idx_x_y_gen:
        keep_this_point_1 = True
        keep_this_point_2 = True
        for inc_exc, breaks, polys_array in zip(keep, breaks_inc_exc, arrs):
            keep_this_point_2 = True
            if polys_array.size == 0:
                continue
            for b in range(0, breaks.size + 1):
                if b < breaks.size:
                    b2 = breaks[b]
                else:
                    b2 = polys_array.shape[0]
                if b == 0:
                    b1 = -1
                else:
                    b1 = breaks[b - 1]
                in_poly = point_in_poly(x, y, polys_array[b1 + 1: b2, :])
                if in_poly and not inc_exc:
                    keep_this_point_2 = False
                    break
                elif in_poly and inc_exc:
                    keep_this_point_2 = True
                    break
                elif not in_poly and not inc_exc:
                    keep_this_point_2 = True
                elif not in_poly and inc_exc:
                    keep_this_point_2 = False
                    break
            keep_this_point_1 = keep_this_point_1 and keep_this_point_2
        if keep_this_point_1:
            points.append((idx, x, y))
    return np.array(points, dtype=np.float64)

def points_in_poly(unique_x=None, unique_y=None,
                   all_x_coords=None, all_y_coords=None,
                   include_polys=None, exclude_polys=None):
    '''Filters x,y based on exclude_polygons / include_polygons.

    Params:
        unique_x:    unique array labels of x (lon)
        unique_y:    unique array labels of y (lat)
        all_x_coords:flattened all x points of an array
        all_y_coords:flattened all y points of an array
        include_polys:list of numpy arrays of shape (M, 2)
        exclude_polys:list of numpy arrays of shape (M, 2)
    Returns:
        numpy array of shape (number_of_kept_points, 3)
        where the 3 columns are
            * index into the original array
            * longitude
            * latitude
    '''
    has_err = False
    if unique_x is not None and unique_y is not None:
        match_coords_shape = unique_x.size == unique_y.size
        # make all the right data types
        all_x_coords = unique_x[:0]
        all_y_coords = unique_y[:0]
    elif all_x_coords is not None and all_y_coords is not None:
        match_coords_shape = all_y_coords.size == all_x_coords.size
        # make all the right data types
        unique_x = all_x_coords[:0]
        unique_y = all_y_coords[:0]
    else:
        has_err = True
    for poly_arg in (include_polys, exclude_polys):
        if not (poly_arg is None or isinstance(poly_arg, list)):
            raise ValueError('Expected include_polys / exclude_polys to '
                             'be a list of numpy arrays (or None)')
    if has_err or not match_coords_shape:
        raise ValueError('points_to_poly: give unique_x and unique_y or '
                         'all_x_coords and all_y_coords.\n'
                         'In any case, x.size must == y.size')
    include_polys = include_polys or []
    exclude_polys = exclude_polys or []
    return _points_in_poly(unique_x.astype(np.float64),
                           unique_y.astype(np.float64),
                           all_x_coords.astype(np.float64),
                           all_y_coords.astype(np.float64),
                           _polys_to_nan_poly(include_polys).astype(np.float64),
                           _polys_to_nan_poly(exclude_polys).astype(np.float64))

def _polys_to_nan_poly(polys):
    sizes = [poly.shape[0] for poly in polys]
    polys_array = np.zeros((sum(sizes) + len(sizes), 2),
                           dtype=np.float64) * np.NaN
    start = 0
    for siz, poly in zip(sizes, polys):
        polys_array[start: start + siz, :] = poly
        start += siz + 1
    return polys_array

def _filter_band_data(handle, subhandle, time,
                    include_polys, exclude_polys, data_filter,
                    band_meta, bounds,
                    idxes=None, lons=None, lats=None):
    data = subhandle.ReadAsArray()
    subhandle = None
    if idxes is None:
        lon1, lon2 = sorted((bounds.east, bounds.west))
        lons = np.linspace(lon1, lon2, data.shape[0])
        lat1, lat2 = sorted((bounds.south, bounds.north))
        lats = np.linspace(lat1, lat2, data.shape[1])
        if include_polys or exclude_polys:
            idx_lon_lat = points_in_poly(unique_x=lons,
                                         unique_y=lats,
                                         include_polys=include_polys,
                                         exclude_polys=exclude_polys)
            if idx_lon_lat.size == 0:
                logger.warn('Empty geographic selection in {}'.format(bounds))
                return data[:0], lons, lats
            idxes = idx_lon_lat[:, 0]
            lons = idx_lon_lat[:, 1]
            lats = idx_lon_lat[:, 2]
    if idxes is not None:
        if idxes.shape[0]:
            values = data.ravel()[idxes.astype(np.int32)]
        else:
            logger.warn('Empty geographic selection in {}'.format(bounds))
            return data[:0], lons, lats
    else:
        values = data.ravel()
        lons, lats = np.meshgrid(lons, lats)
        lons = lons.ravel()
        lats = lats.ravel()

    return values, lons, lats, idxes
