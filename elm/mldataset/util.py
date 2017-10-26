import numpy as np
import dask.array as da


def is_mldataset(arr, raise_err=False):
    try:
        from xarray_filters import MLDataset
        from xarray import Dataset
        return True
    except Exception as e:
        MLDataset = Dataset = None
        if not raise_err:
            return False
        # Much of the ML logic
        # wrapping Xarray would fail
        # if only xarray and not Xarray_filters
        # is installed, but when xarray_filters
        # is installed, xarray.Dataset can be
        # used
        raise ValueError('Cannot use cross validation for xarray Dataset without xarray_filters')
    return MLDataset and isinstance(arr, (MLDataset, Dataset))


def is_arr(arr, raise_err=False):
    is_ml = is_mldataset(arr, raise_err=raise_err)
    return is_ml or isinstance(arr, (np.ndarray, da.Array))