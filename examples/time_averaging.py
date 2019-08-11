from collections import OrderedDict
import numpy as np
from read_nldas_forcing import SOIL_MOISTURE


def time_averaging(self, arr, weights, y_field=SOIL_MOISTURE):
    if not 'time' in arr.dims:
        return arr
    tidx = arr.dims.index('time')
    siz = [1] * len(arr.dims)
    siz[tidx] = arr.time.values.size
    mx = np.max(siz)
    a, b = weights
    weights = np.linspace(a, b, mx)
    weights /= weights.sum()
    weights.resize(tuple(siz))
    if arr.name != y_field:
        weighted = (arr * weights)
        arr2 = weighted.sum(dim='time')
    else:
        arr2 = arr.isel(time=siz[tidx] - 1)
    arr2.attrs.update(arr.attrs)
    return arr2
