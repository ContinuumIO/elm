from collections import namedtuple

from numba import njit
import numpy as np

from iamlp.settings import delayed

SpatialBounds = namedtuple('SpatialBounds', 'north south east west')

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

@njit('float64[:, :](float64[:], float64[:], float64[:,:])')
def points_in_poly(unique_x, unique_y, include_polys_array):
    points = []
    breaks = np.where(np.isnan(include_polys_array[:, 0]))[0]
    for i in range(unique_x.size):
        for j in range(unique_y.size):
            for b in range(0, breaks.size):
                if b == 0:
                    b1 = 0
                else:
                    b1 = breaks[b - 1]
                b2 = breaks[b]
            x = unique_x[i]
            y = unique_y[j]
            idx = i * unique_x.size + j
            for b1, b2 in zip(breaks[:-1], breaks[1:]):
                if point_in_poly(x, y, include_polys_array[b1 + 1: b2, :]):
                    points.append((idx, x, y))
    return np.array(points, dtype=np.float64)

@delayed
def _filter_band_data(handle, subhandle, time,
                    include_polys, data_filter,
                    band_meta, bounds,
                    idxes=None, lons=None, lats=None):
    data = subhandle.ReadAsArray()
    subhandle = None
    if idxes is None:
        lons = np.linspace(bounds.east, bounds.west, data.shape[0])
        lats = np.linspace(bounds.south, bounds.north, data.shape[1])
        if include_polys is not None:
            sizes = [poly.shape[0] for poly in include_polys]

            include_polys_array = np.ones((sum(sizes) + len(sizes), 2),
                                           dtype=np.float64) * np.NaN
            start = 0
            for siz, poly in zip(sizes, include_polys):
                include_polys_array[start: start + siz, :] = poly
                start += siz + 1
            idx_lon_lat = points_in_poly(lons, lats, include_polys_array)
            idxes = idx_lon_lat[:, 0]
            lons = idx_lon_lat[:, 1]
            lats = idx_lon_lat[:, 2]
    if idxes is not None:
        values = data[idxes]
    else:
        values = data.ravel()
        lons, lats = np.meshgrid(lons, lats)
        lons = lons.ravel()
        lats = lats.ravel()
    return values, lons, lats, idxes
