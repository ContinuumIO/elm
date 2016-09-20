from collections import OrderedDict

import attr
import numpy as np
import pytest
import xarray as xr

from elm.readers import *
from elm.readers.tests.test_hdf4 import HDF4_FILES, band_specs as hdf4_band_specs
from elm.readers.tests.test_hdf5 import HDF5_FILES, get_band_specs
from elm.readers.tests.test_tif import TIF_DIR, band_specs as tif_band_specs


def random_elm_store_no_meta(width=100, height=200):
    bands = ['band_1', 'band_2']
    es = OrderedDict()
    for band in bands:
        arr = np.random.uniform(0, 1, width * height).reshape(height, width)
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
    '''Tests ElmStore can be flattened / inverse even with no attrs'''
    es = random_elm_store_no_meta()
    flat = flatten(es)
    inv = inverse_flatten(flat)
    assert np.all(es.band_1.values == inv.band_1.values)
    assert np.all(es.band_2.values == inv.band_2.values)
    assert np.all(flat.flat.values[:, 0] == es.band_1.values.ravel(order='C'))


def test_na_drop_no_meta():
    '''Tests ElmStore can be flattened / inverse even with NaNs
    dropped and no attrs'''
    es = random_elm_store_no_meta()
    flat = flatten(es)
    flat.flat.values[:3, :] = np.NaN
    flat.flat.values[10:12, :] = np.NaN
    na_dropped = drop_na_rows(flat)
    assert na_dropped.flat.values.shape[0] == flat.flat.values.shape[0] - 5
    inv = inverse_flatten(na_dropped)
    flat2 = flatten(inv)
    val1 = flat.flat.values
    val2 = flat2.flat.values
    assert np.all(val1[~np.isnan(val1)] == val2[~np.isnan(val2)])
    inv2 = inverse_flatten(flat2)
    val1 = inv.band_1.values
    val2 = inv2.band_1.values
    assert np.all(val1[~np.isnan(val1)] == val2[~np.isnan(val2)])


@pytest.mark.parametrize('ftype', ('hdf4', 'hdf5', 'tif',))
def test_reader_kwargs_window(ftype):

    '''Assert that "window" can be passed in a BandSpec
    to control the (ymin, ymax), (xmin, xmax) window to read'''
    if ftype == 'hdf5':
        _, band_specs = get_band_specs(HDF5_FILES[0])
        meta = load_hdf5_meta(HDF5_FILES[0])
        full_es = load_hdf5_array(HDF5_FILES[0], meta, band_specs)
    elif ftype == 'hdf4':
        band_specs = hdf4_band_specs
        meta = load_hdf4_meta(HDF4_FILES[0])
        full_es = load_hdf4_array(HDF4_FILES[0], meta, band_specs)
    elif ftype == 'tif':
        band_specs = tif_band_specs[:2]
        meta = load_dir_of_tifs_meta(TIF_DIR, band_specs=band_specs)
        full_es = load_dir_of_tifs_array(TIF_DIR, meta, band_specs)
    band_specs_window = []
    windows = {}
    for b in band_specs:
        name = b.name
        val = getattr(full_es, name).values
        shp = val.shape
        b = attr.asdict(b)
        b['window'] = windows[name] = (((10, 200), (210, 400)))
        band_specs_window.append(BandSpec(**b))

    if ftype == 'hdf4':
        es = load_hdf4_array(HDF4_FILES[0], meta, band_specs_window)
    elif ftype == 'hdf5':
        es = load_hdf5_array(HDF5_FILES[0], meta, band_specs_window)
    elif ftype == 'tif':
        meta_small = load_dir_of_tifs_meta(TIF_DIR, band_specs=band_specs_window)
        es = load_dir_of_tifs_array(TIF_DIR, meta_small, band_specs_window)

    for band in es.data_vars:
        window = windows[band]
        subset = getattr(es, band, None)
        assert subset is not None
        subset = subset.values
        expected_shape = tuple(map(np.diff, window))
        assert subset.shape == expected_shape

