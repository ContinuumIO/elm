from __future__ import absolute_import, division, print_function, unicode_literals

import calendar
from collections import OrderedDict
import copy
import gc
from itertools import product, combinations
import logging
import glob
import random

from elm.model_selection.kmeans import kmeans_aic, kmeans_model_averaging
from elm.readers import *
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
from scipy.stats import describe
import xarray as xr



FIRST_YEAR, LAST_YEAR = 1980, 2015

PATTERN = '/mnt/efs/goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4/{:04d}/{:02d}/*.nc4'

MONTHLY_PATTERN = '/mnt/efs/goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2IMNXASM.5.12.4/*/MERRA2_100.instM_2d_asm_Nx.*{:02d}.nc4'

TEMP_BAND = 'T2MMEAN'

YEARS = range(FIRST_YEAR, LAST_YEAR + 1)

CHUNKS = {}#{'lat': 361, 'lon': 576}

DEFAULT_PERCENTILES = (1, 2.5, 5, 25, 50, 75, 95, 97.5)

def split_fname(f):
    parts = f.split('.')
    dt = parts[-2]
    return int(dt[:4]), int(dt[4:6]), int(dt[6:8])


def month_means(month):
    pattern = MONTHLY_PATTERN.format(month)
    files = glob.glob(pattern)
    x = xr.open_mfdataset(files, chunks=CHUNKS or None)
    return x.mean(dim='time')


def sample(month, days, **kwargs):

    print('Sample - Month: {} Days: {}'.format(month, days))
    files = []
    for year in YEARS:
        pattern = PATTERN.format(year, month)
        fs = glob.glob(pattern)
        dates = [split_fname(f) for f in fs]
        keep = [idx for idx, d in enumerate(dates)
                if d[1] == month and d[2] in days]
        files.extend(fs[idx] for idx in keep)
    print('Sample {} files'.format(len(files)))
    x = xr.open_mfdataset(files, chunks=CHUNKS or None)
    x.attrs['sample_kwargs'] = {'month': month, 'days': days}
    x.attrs['band_order'] = [TEMP_BAND]
    x.attrs['old_dims'] = [getattr(x, TEMP_BAND).dims[1:]]
    x.attrs['old_coords'] = {k: v for k, v in x.coords.items()
                             if k in ('lon', 'lat',)}
    return normalize_in_time(x)


def normalize_in_time(x, normalize_by='month', **kwargs):
    month = x.sample_kwargs['month']
    days = x.sample_kwargs['days']
    bin_size = kwargs.get('bin_size', 0.5)
    num_bins = kwargs.get('num_bins', 152)
    normalize_by = kwargs.get('normalize_by', 'month')
    if normalize_by == 'month':
        monthly = month_means(month)
    percentiles = kwargs.get('percentiles', DEFAULT_PERCENTILES)
    bins = np.linspace(-bin_size * (num_bins // 2), bin_size * (num_bins // 2), num_bins + 1)
    band_arr = getattr(x, TEMP_BAND)
    date = pd.DatetimeIndex(tuple(pd.Timestamp(v) for v in band_arr.time.values))

    for year in YEARS:
        for day in days:
            idxes = np.where((date.day == day)&(date.year == year)&(date.month == month))[0]
            slc = (idxes,
                   slice(None),
                   slice(None)
                   )
            one_day = band_arr.values[slc]
            if normalize_by == 'month':
                mean = monthly.T2M.values[slc[1], slc[2]]
            else:
                mean = one_day.mean(axis=0)
            band_arr.values[slc] = (one_day - mean)
    return ElmStore({TEMP_BAND: band_arr}, attrs=x.attrs, add_canvas=False)


def scipy_describe(x, **kwargs):
    print('Start scipy_describe')
    band_arr = getattr(x, TEMP_BAND)
    cols = ('var', 'skew', 'kurt', 'min', 'max', 'median', 'std', 'np_skew')
    inter = tuple(combinations(range(len(cols)), 2))
    cols = cols + tuple((cols[i], cols[j]) for i, j in inter)
    num_cols = len(cols)
    num_rows = np.prod(band_arr.shape[1:])
    new_arr = np.empty((num_rows, num_cols))
    for row, (i, j) in enumerate(product(*(range(s) for s in band_arr.values.shape[1:]))):
        values = band_arr.values[:, i, j]
        d = describe(values)
        t = (d.variance, d.skewness, d.kurtosis, d.minmax[0], d.minmax[1])
        median = np.median(values)
        std = np.std(values)
        non_param_skew = (d.mean - median) / std

        r = t + (median, std, non_param_skew)
        interact = tuple(r[i] * r[j] for i, j in inter)
        new_arr[row, :] = r + interact
    attrs = copy.deepcopy(x.attrs)
    attrs.update(kwargs)
    da = xr.DataArray(new_arr,
                      coords=[('space', np.arange(num_rows)),
                              ('band', np.arange(num_cols))],
                      dims=('space', 'band'),
                      attrs=attrs)
    return ElmStore({'flat': da}, attrs=attrs, add_canvas=False)


def histogram(x, **kwargs):
    band_arr = getattr(x, TEMP_BAND)
    num_bins = kwargs['num_bins']
    bin_size = kwargs.get('bin_size', None)
    log_counts = kwargs.get('log_counts', None)
    edges = kwargs.get('edges', None)
    counts = kwargs.get('counts')
    if counts and log_counts:
        raise ValueError('Choose "counts" or "log_counts"')
    if log_counts:
        columns_from = ['log_counts']
    else:
        columns_from = []
    if edges:
        columns_from.append('edges')
    if counts:
        columns_from.append('counts')
    if bin_size is not None:
        bins = np.linspace(-bin_size * num_bins // 2, bin_size * num_bins // 2, num_bins + 1)
    num_rows = np.prod(band_arr.shape[1:])
    col_count = len(columns_from) * num_bins
    if 'edges' in columns_from:
        col_count += 1
    if bin_size is not None:
        new_arr = np.empty((num_rows, bins.size))
        col_count = bins.size
    else:
        new_arr = np.empty((num_rows, col_count),dtype=np.float64)

    print("Histogramming...")
    values = band_arr.values
    for row, (i, j) in enumerate(product(*(range(s) for s in values.shape[1:]))):
        if bin_size is not None:
            indices = np.searchsorted(bins, values[:, i, j], side='left')
            binned = np.bincount(indices).astype(np.float64)
            binned /= values.shape[0]
            if log_counts:
                binned = np.log10(binned)
            new_arr[row, :binned.size] = binned
            if binned.size < new_arr.shape[1]:
                new_arr[row, binned.size:] = 0
        else:
            hist, edges = np.histogram(values[:, i, j], num_bins)
            hist[hist == 0] = tiny
            hist = hist / values.shape[0]
            if log_counts:
                hist = np.log10(hist)
            if len(columns_from) == 1:
                if log_counts or counts:
                    row_arr = hist
                else:
                    row_arr = edges
            else:
                row_arr = np.concatenate((hist, edges))
            new_arr[row, :] = row_arr

    gc.collect()
    attrs = copy.deepcopy(x.attrs)
    attrs.update(kwargs)
    da = xr.DataArray(new_arr,
                      coords=[('space', np.arange(num_rows)),
                              ('band', np.arange(col_count))],
                      dims=('space', 'band'),
                      attrs=attrs)
    return ElmStore({'flat': da}, attrs=attrs, add_canvas=False)


def sample_args_generator(**kwargs):
    start, end = 1, 1 + kwargs['num_days']
    while end <= calendar.monthrange(1999, kwargs['month'])[1]:
        yield (kwargs['month'], list(range(start, end)))
        start += kwargs['num_days']
        end = start + kwargs['num_days']



