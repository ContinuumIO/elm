import os

import pytest
import numpy as np

from elm.pipeline.tests.util import random_elm_store
from elm.readers.tests.util import HDF4_FILES, NETCDF_FILES, TIF_FILES
from elm.readers import *
from elm.readers.tests.test_hdf4 import band_specs as hdf4_band_specs
from elm.readers.tests.test_tif import band_specs as tif_band_specs


readers = {'hdf': [load_hdf4_meta, load_hdf4_array],
           'netcdf': [load_netcdf_meta, load_netcdf_array],
           'tif':  [load_dir_of_tifs_meta, load_dir_of_tifs_array]}

band_specs = {'hdf': hdf4_band_specs,
              'tif': tif_band_specs,
              'netcdf': ['HQobservationTime']}

FILES = {'hdf': HDF4_FILES, 'netcdf': NETCDF_FILES, 'tif': TIF_FILES}


def _setup(ftype, fnames_list):
    bs = band_specs[ftype]
    load_meta, load_array = readers[ftype]
    if ftype == 'tif':
        fnames_list = list(set(os.path.dirname(f) for f in fnames_list))
    fname = fnames_list[0]
    if ftype != 'tif':
        meta = load_meta(fname)
    else:
        meta = load_meta(fname, bs)
    es = load_array(fname, meta, bs)
    return es

@pytest.mark.slow
@pytest.mark.parametrize('ftype,fnames_list', sorted(FILES.items()))
def test_reshape(ftype, fnames_list):
    old_shapes = {}
    es = _setup(ftype, fnames_list)
    assert not hasattr(es, 'flat')
    if ftype in ('hdf', 'tif'):
        for band in es.data_vars:
            band_arr = getattr(es, band)
            assert hasattr(band_arr, 'canvas')
            assert hasattr(band_arr, 'geo_transform')
            assert hasattr(band_arr.canvas, 'xsize')
            assert hasattr(band_arr, 'y')
            assert hasattr(band_arr, 'x')
            assert hasattr(band_arr.canvas, 'bounds')
            old_shapes[band] = band_arr.values.shape
    else:
        assert hasattr(es, 'canvas')
        assert hasattr(es, 'geo_transform')
        assert hasattr(es, 'bounds')
        for band in es.data_vars:
            band_arr = getattr(es, band)
            break
    flat = flatten(es)
    def assert_once(flat):
        assert hasattr(flat, 'flat')
        assert hasattr(flat, 'old_canvases')
        assert hasattr(flat.flat, 'old_canvases')
        assert hasattr(flat, 'band_order')
        assert hasattr(flat, 'old_dims')
        assert hasattr(flat.old_canvases[0], 'bounds')
    assert_once(flat)
    flat = flatten(flat)
    assert_once(flat)
    old_canvases, old_dims = flat.old_canvases, flat.old_dims
    flat.flat.values[-2:, :] = np.NaN
    na_dropped = drop_na_rows(flat)
    assert hasattr(na_dropped, 'flat')
    assert na_dropped.flat.values.shape[0] == flat.flat.values.shape[0] - 2
    filled = filled_flattened(na_dropped)
    assert hasattr(filled, 'flat')
    assert filled.flat.values.shape == flat.flat.values.shape
    canvas = es.get_shared_canvas()
    if canvas is None:
        for band in es_new.band_vars:
            canvas = getattr(es_new, band).canvas
            break
    es_new = inverse_flatten(es.select_canvas(canvas).flatten(), band_arr.dims)
    assert hasattr(es_new, 'band_order')
    for cv, dims, band in zip(old_canvases, old_dims, es_new.band_order):
        band_arr = getattr(es_new, band)
        if band in old_shapes:
            assert old_shapes[band] == band_arr.values.shape


@pytest.mark.parametrize('ftype, fnames_list', sorted(x for x in FILES.items() if not 'tif' in x))
def test_elm_store_methods(ftype, fnames_list):
    es = _setup(ftype, fnames_list)
    bands = tuple(es.data_vars)
    assert bands == tuple(es.band_order)
    old_band_arr = getattr(es, bands[0])
    flat = es.flatten()
    assert hasattr(flat, 'flat')
    assert len(flat.flat.values.shape) == 2
    assert flat.flat.values.shape[1] == len(bands)
    assert np.all(flat.flat.band == bands)
    flat.flat.values[:3, :] = np.NaN
    na_dropped = flat.drop_na_rows()
    assert hasattr(na_dropped, 'flat')
    assert na_dropped.flat.values.shape[0] == flat.flat.values.shape[0] - 3

    es_new = na_dropped.inverse_flatten(old_band_arr.dims)
    assert tuple(es_new.band_order) == bands
    assert tuple(es_new.data_vars) == bands
    for band in es_new.data_vars:
        band_arr = getattr(es_new, band)
        assert hasattr(band_arr, 'canvas')
        assert hasattr(band_arr.canvas, 'bounds')
        val = band_arr.values
        assert val[np.isnan(val)].size == 3


@pytest.mark.parametrize('ftype, fnames_list', sorted(FILES.items()))
def test_canvas_select(ftype, fnames_list):
    es = _setup(ftype, fnames_list)
    for band in es.data_vars:
        band_arr = getattr(es, band)
        sel = select_canvas_elm_store(es, band_arr.canvas)
        assert np.all(sel.canvas == band_arr.canvas)
        for band2 in sel.data_vars:
            assert getattr(sel, band2).values.shape == band_arr.values.shape
            break
        break
    canvas_dict_orig = band_arr.canvas._asdict()
    canvas_dict = canvas_dict_orig.copy()
    canvas_dict['xsize'] = canvas_dict['xsize'] // 2
    canvas_dict['ysize'] = canvas_dict['ysize'] // 4
    new_canvas = Canvas(**canvas_dict)
    sel2 = es.select_canvas(new_canvas)
    for band2 in sel2.data_vars:
        band_arr2 = getattr(sel2, band2)
        xidx = [idx for idx, x in enumerate(band_arr2.dims)
                if x.lower() in VALID_X_NAMES][0]
        yidx = [idx for idx, y in enumerate(band_arr2.dims)
                if y.lower() in VALID_Y_NAMES][0]
        assert band_arr2.values.shape[xidx] == band_arr.values.shape[xidx] // 2
        assert band_arr2.values.shape[yidx] == band_arr.values.shape[yidx] // 4
        break

def test_flatten_inverse_flatten():
    ftype, fnames_list = sorted(f for f in FILES.items()
                         if f[0] == 'hdf')[0]
    es = _setup(ftype, fnames_list)
    flat = es.flatten()
    inv = flat.inverse_flatten(('y', 'x'))
    flat2 = inv.flatten()
    flat3 = flat2.drop_na_rows()
    inv2 = flat3.inverse_flatten(('y', 'x'))
    inv3 = inv2.transpose('x', 'y')
    assert np.all(inv3.band_1.values == es.band_1.transpose('x', 'y').values)

