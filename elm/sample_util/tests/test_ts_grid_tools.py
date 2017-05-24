import numpy as np
import pytest
import xarray as xr

from earthio import ElmStore
from elm.config.tests.fixtures import *
from elm.pipeline import steps

def make_3d():
    arr = np.random.uniform(0, 1, 100000).reshape(100, 10, 100)
    return ElmStore({'band_1': xr.DataArray(arr,
                            coords=[('time', np.arange(100)),
                                    ('x', np.arange(10)),
                                    ('y',np.arange(100))],
                            dims=('time', 'x', 'y'),
                            attrs={})}, attrs={}, add_canvas=False)


def test_ts_probs():

    s = steps.TSProbs()
    s.set_params(band='band_1', bin_size=0.5, num_bins=152, log_probs=True)
    orig = make_3d()
    X, _, _ = s.fit_transform(orig)
    assert hasattr(X, 'flat')
    assert X.flat.values.shape[1] == 152
    s.set_params(band='band_1', bin_size=0.5, num_bins=152,
                 log_probs=False)
    X2, _, _ = s.fit_transform(orig)
    assert hasattr(X2, 'flat')
    assert X2.flat.values.shape[1] == 152
    s.set_params(band='band_1', bin_size=0.5, num_bins=152, log_probs=True)
    X3, _, _ = s.fit_transform(orig)
    assert hasattr(X3, 'flat')
    assert X3.flat.values.shape[1] == 152
    with pytest.raises(ValueError):
        s = steps.TSProbs()
        s.fit_transform(orig)
    s.set_params(band='band_1', num_bins=152, log_probs=False)
    X4, _, _ = s.fit_transform(orig)
    assert hasattr(X4, 'flat')
    assert X4.flat.values.shape[1] == 152


def test_ts_describe():

    s = steps.TSDescribe()
    s.set_params(band='band_1', axis=0)
    orig = make_3d()
    X, _, _ = s.fit_transform(orig)
    bands = tuple(X.band)
    assert bands == ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')

