from collections import OrderedDict

import numpy as np
import xarray as xr

from elm.readers import *

def random_elm_store_no_meta(width=100, height=200):
    bands = ['band_1', 'band_2']
    es = OrderedDict()
    for band in bands:
        arr = np.random.uniform(0, 1, width*height).reshape(height, width)
        band_arr = xr.DataArray(arr,
                                 coords=[
                                    ('y', np.arange(height)),
                                    ('x', np.arange(width))
                                ],
                                dims=('y', 'x'),
                                attrs={})
        es[band] = band_arr
    return ElmStore(es, add_canvas=False)


def test_flatten_no_meta():
    es = random_elm_store_no_meta()
    flat = flatten(es)
    inv = inverse_flatten(flat)
    assert np.all(es.band_1.values == inv.band_1.values)
    assert np.all(es.band_2.values == inv.band_2.values)
    assert np.all(flat.flat.values[:, 0] == es.band_1.values.ravel(order='C'))


def test_na_drop_no_meta():
    es = random_elm_store_no_meta()
    flat = flatten(es)
    flat.flat.values[:3, :] = np.NaN
    flat.flat.values[10:12, :] = np.NaN
    na_dropped = drop_na_rows(flat)
    inv = inverse_flatten(na_dropped)
    flat2 = flatten(inv)
    val1 = flat.flat.values
    val2 = flat2.flat.values
    assert np.all(val1[~np.isnan(val1)] == val2[~np.isnan(val2)])
    inv2 = inverse_flatten(flat2)
    val1 = inv.band_1.values
    val2 = inv2.band_1.values
    assert np.all(val1[~np.isnan(val1)] == val2[~np.isnan(val2)])