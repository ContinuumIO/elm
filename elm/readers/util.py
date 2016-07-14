import gdal
import numpy as np
import ogr

def get_y_right_bottom(xsize, ysize, geo_transform):
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    x_left = geo_transform[0]
    y_top = geo_transform[3]
    x_right = x_left + xsize * pixel_width
    y_bottom = y_top - ysize * pixel_height
    return x_left, x_right, y_top, y_bottom

def geotransform_to_dims(xsize, ysize, geo_transform):
    x_left, x_right, y_top, y_bottom = get_y_right_bottom(xsize, ysize, geo_transform)
    return (np.linspace(x_left, x_right, xsize),
            np.linspace(y_bottom, y_top, ysize))

def geotransform_to_bounds(xsize, ysize, geo_transform):
    x_left, x_right, y_top, y_bottom = get_y_right_bottom(xsize, ysize, geo_transform)
    bbox = ogr.Geometry(ogr.wkbLinearRing)
    bbox.AddPoint(x_left, y_top)
    bbox.AddPoint(x_left, y_bottom)
    bbox.AddPoint(x_right, y_top)
    bbox.AddPoint(x_right, y_bottom)
    bbox.AddPoint(x_left, y_top)
    return bbox