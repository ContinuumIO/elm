from collections import OrderedDict

import numpy as np
import xarray as xr
from xarray_filters import MLDataset

def reduce_series(reducer, weights, arrs):
    arrs = (arr * w for arr, w in zip(arrs, weights))
    arr = xr.concat(arrs)
    arr = getattr(arr, reducer)(axis=0)
    dset = MLDataset(OrderedDict([('features', arr)]))
    return dset


def get_weights_for_bins(end, n_bins, weight_type):
    if weight_type == 'linear':
        weights = np.linspace(end, 0, n_bins + 1)
    elif weight_type == 'uniform':
        weights = np.ones(n_bins + 1)
    elif weight_type == 'log':
        weights = np.logspace(end, 0, n_bins + 1)
    weights = ((weights[:-1] + weights[1:]) / 2.)
    return weights


def differencing_integrating(X,
                             layers=None,
                             first_bin_width=12,
                             last_bin_width=1,
                             hours_back=144,
                             num_bins=12,
                             time_operation=None,
                             weight_type='uniform',
                             reducers=None):

    if not reducers:
        reducers = ('mean',)
    if not isinstance(reducers, (tuple, list)):
        reducers = (reducers,)
    if weight_type == 'linear':
        func = np.linspace
        end = hours_back
        start = last_bin_width
    else:
        func = np.logspace
        end = np.log10(hours_back)
        start = np.log10(last_bin_width)
    bins = func(start, end, num_bins)
    bins = np.unique(np.round(bins).astype(np.int32))
    weights = get_weights_for_bins(end, bins.size, weight_type)
    X = X.copy()
    new_X = OrderedDict(X.data_vars)
    running_fields = []
    running_diffs = []

    for col in layers:
        for first_hr, second_hr in zip(bins[:-1],
                                       bins[1:]):
            for reducer in reducers:
                if isinstance(reducer, (tuple, list)):
                    diff_first = 'diff' == reducer[0]
                    reducer = reducer[1]
                for i in range(first_hr, second_hr):
                    end_period = 'hr_{}_{}'.format(first_hr, col)
                    start_period = 'hr_{}_{}'.format(second_hr, col)
                    end_array = X.data_vars[end_period]
                    start_array = X.data_vars[start_period]
                    running_fields.append(end_array)
                    if 'diff' in reducers:
                        diff = start_array - end_array
                        diff.attrs.update(start_array.attrs.copy())
                        running_diffs.append(diff)
                if 'diff' in reducers:
                    diff_col_name = 'diff_{}_{}_{}'.format(first_hr, second_hr, col)
                    arr = reduce_series(reducer, weights, running_diffs)
                    new_X[diff_col_name] = arr
                    running_diffs = []
                arr = reduce_series(reducer, weights, running_fields)
                new_X[start_period] = arr
                running_fields = []
    X = MLDataset(new_X, attrs=X.attrs)
    return X


