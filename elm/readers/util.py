import gdal
import numpy as np
import ogr
from rasterio.coords import BoundingBox

SPATIAL_KEYS = ('Height', 'Width', 'GeoTransform', 'Bounds')

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

def geotransform_to_dims(xsize, ysize, geo_transform):
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
