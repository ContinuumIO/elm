import numpy as np
import dask.array as da

from collections import Sequence


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


def _split_transformer_result(Xt, y):
    if isinstance(Xt, Sequence) and len(Xt) == 2 and is_arr(Xt[1]):
        Xt, new_y = Xt
        print('was Sequence', type(Xt), type(new_y), getattr(Xt, 'shape', 'noshape'), getattr(y, 'size', y))
    else:
        new_y = y
    if y is None and new_y is not None:
        y = new_y
    assert not isinstance(y, tuple), repr((Xt, y, new_y))
    return Xt, y
