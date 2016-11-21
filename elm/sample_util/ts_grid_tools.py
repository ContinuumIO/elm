'''
---------------------------------

``elm.sample_util.ts_grig_tools``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
import calendar
from collections import OrderedDict
import copy
import gc
from itertools import product, combinations
import logging
import glob
import random

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
from scipy.stats import describe
import xarray as xr

from elm.model_selection.kmeans import kmeans_aic, kmeans_model_averaging
from elm.readers import ElmStore
from elm.sample_util.step_mixin import StepMixin


logger = logging.getLogger(__name__)
slc = slice(None)

def _ij_for_axis(axis, i, j):
    if axis == 0:
        return (slc, i, j)
    elif axis == 1:
        return (i, slc, j)
    elif axis == 2:
        return (i, j, slc)
    else:
        raise ValueError("Expected axis in (0, 1, 2)")


def ts_describe(X, y=None, sample_weight=None, **kwargs):
    '''scipy.describe on the `band` from kwargs
    that is a 3-D DataArray in X

    Parameters:
        X:  ElmStore or xarray.Dataset
        y:  passed through
        sample_weight: passed through
        kwargs: Keywords:
            axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
            band: The name of the DataArray in ElmStore to run scipy.describe on
    Returns:
        X:  ElmStore with DataArray class "flat"
    '''
    band = kwargs['band']
    logger.debug('Start scipy_describe band: {}'.format(band))
    band_arr = getattr(X, band)
    cols = ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')
    num_cols = len(cols)

    inds = _ij_for_axis(kwargs['axis'], 0, 0)
    shp = tuple(s for idx, s in enumerate(band_arr.values.shape)
                if isinstance(inds[idx], int))
    num_rows = np.prod(shp)
    new_arr = np.empty((num_rows, num_cols))
    for row, (i, j) in enumerate(product(*(range(s) for s in shp))):
        ind1, ind2, ind3 = _ij_for_axis(kwargs['axis'], i, j)
        values = band_arr.values[ind1, ind2, ind3]
        d = describe(values)
        t = (d.variance, d.skewness, d.kurtosis, d.minmax[0], d.minmax[1])
        median = np.median(values)
        std = np.std(values)
        non_param_skew = (d.mean - median) / std
        r = t + (median, std, non_param_skew)
        new_arr[row, :] = r
    attrs = copy.deepcopy(X.attrs)
    attrs.update(kwargs)
    da = xr.DataArray(new_arr,
                      coords=[('space', np.arange(num_rows)),
                              ('band', np.array(cols))],
                      dims=('space', 'band'),
                      attrs=attrs)
    X_new = ElmStore({'flat': da}, attrs=attrs, add_canvas=False)
    return (X_new, y, sample_weight)


def ts_probs(X, y=None, sample_weight=None, **kwargs):
    '''Fixed or unevenly spaced histogram binning for
    the time dimension of a 3-D cube DataArray in X

    Parameters:
        X: ElmStore or xarray.Dataset
        y: passed through
        sample_weight: passed through
        kwargs: Keywords:
            axis: Integer like 0, 1, 2 to indicate which is the time axis of cube
            band: The name of DataArray to time series bin (required)
            bin_size: Size of the fixed bin or None to use np.histogram (irregular bins)
            num_bins: How many bins
            log_probs: Return probabilities associated with log counts? True / False
    Returns:
        X: ElmStore with DataArray called flat that has columns composed of:
            * log transformed counts (if kwargs["log_probs"]) or
            * counts (if kwargs["counts"])

        Number of columns will be equal to num_bins
    '''
    band = kwargs['band']
    band_arr = getattr(X, band)
    num_bins = kwargs['num_bins']
    bin_size = kwargs.get('bin_size', None)
    log_probs = kwargs.get('log_probs', None)
    if bin_size is not None:
        bins = np.linspace(-bin_size * num_bins // 2, bin_size * num_bins // 2, num_bins)
    num_rows = np.prod(band_arr.shape[1:])
    col_count =  num_bins
    new_arr = np.empty((num_rows, col_count),dtype=np.float64)
    logger.info("Histogramming...")
    small = 1e-8
    inds = _ij_for_axis(kwargs['axis'], 0, 0)
    shp = tuple(s for idx, s in enumerate(band_arr.values.shape)
                if isinstance(inds[idx], int))
    for row, (i, j) in enumerate(product(*(range(s) for s in shp))):
        ind1, ind2, ind3 = _ij_for_axis(kwargs['axis'], i, j)
        values_slc = band_arr.values[ind1, ind2, ind3]
        if bin_size is not None:
            indices = np.searchsorted(bins, values_slc, side='left')
            binned = np.bincount(indices).astype(np.float64)
            # add small to avoid log zero
            if log_probs:
                was_zero = binned[binned == 0].size
                binned[binned == 0] = small
            else:
                extra = 0.
            binned /= binned.sum()
            if log_probs:
                binned = np.log10(binned)
            new_arr[row, :binned.size] = binned
            if binned.size < new_arr.shape[1]:
                new_arr[row, binned.size:] = 0
        else:
            hist, edges = np.histogram(values_slc, num_bins)
            # add one observation to avoid log zero
            if log_probs:
                was_zero = hist[hist == 0].size
                hist[hist == 0] = small
            else:
                extra = 1.0
            hist = hist.sum()
            if log_probs:
                hist = np.log10(hist)
            new_arr[row, :] = hist

    gc.collect()
    attrs = copy.deepcopy(X.attrs)
    attrs.update(kwargs)
    da = xr.DataArray(new_arr,
                      coords=[('space', np.arange(num_rows)),
                              ('band', np.arange(col_count))],
                      dims=('space', 'band'),
                      attrs=attrs)
    X_new = ElmStore({'flat': da}, attrs=attrs, add_canvas=False)
    return (X_new, y, sample_weight)


class TSProbs(StepMixin):
    def __init__(self, axis=0, band=None, bin_size=None,
                 num_bins=None, log_probs=True):
        __doc__ = ts_probs.__doc__
        self._kwargs = dict(axis=axis, band=band, bin_size=bin_size,
                            num_bins=num_bins, log_probs=log_probs)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        __doc__ = ts_probs.__doc__
        from elm.pipeline.steps import ModifySample
        if not self._kwargs.get('band'):
            raise ValueError("Expected 'band' keyword to TSProbs")
        m = ModifySample(ts_probs, **self._kwargs)
        return (m.fit_transform(X)[0], y, sample_weight)

    def get_params(self):
        return self._kwargs.copy()

    def set_params(self, **params):
        for k, v in params.items():
            self._kwargs[k] = v

class TSDescribe(StepMixin):

    def __init__(self, axis=0, band=None):
        __doc__ = ts_describe.__doc__
        self._kwargs = dict(axis=axis, band=band)

    def fit_transform(self, X, y=None, sample_weight=None, **kwargs):
        __doc__ = ts_describe.__doc__
        from elm.pipeline.steps import ModifySample
        if not self._kwargs.get('band'):
            raise ValueError("Expected 'band' keyword to TSProbs")
        m = ModifySample(ts_describe, **self._kwargs)
        X_new = m.fit_transform(X)[0]
        return (X_new, y, sample_weight)

    def get_params(self):
        return self._kwargs.copy()

    def set_params(self, **params):
        for k, v in params.items():
            self._kwargs[k] = v

__all__ = ['TSDescribe', 'TSProbs']
