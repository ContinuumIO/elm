from functools import partial

import xarray as xr
try:
    from earthio import ElmStore
except:
    ElmStore = None # TODO handle case where earthio is not installed
from elm.sample_util.change_coords import ModifySample

def two_bands_operation(method, X, y=None, sample_weight=None, spec=None, **kwargs):

    bands = X.band_order.copy()
    es = {}
    if not spec:
        raise ValueError('Expected "spec" in kwargs, e.g. {"ndvi": ["band_4", "band_3]}')
    for idx, (key, (b1, b2)) in enumerate(sorted(spec.items())):
        band1 = getattr(X, b1)
        band2 = getattr(X, b2)
        if method == 'normed_diff':
            new = (band1 - band2) / (band1 + band2)
        elif method == 'diff':
            new = band1 - band2
        elif method == 'sum':
            new = band1 + band2
        elif method == 'ratio':
            new = band1 / band2
        new.attrs.update(band1.attrs)
        es[key] = new
        bands.append(key)
    Xnew = ElmStore(xr.merge([ElmStore(es, add_canvas=False), X]), add_canvas=False)
    Xnew.attrs.update(X.attrs.copy())
    Xnew.attrs['band_order'] = bands
    return (Xnew, y, sample_weight)


bands_normed_diff = partial(two_bands_operation, 'normed_diff')
bands_diff = partial(two_bands_operation, 'diff')
bands_sum = partial(two_bands_operation, 'sum')
bands_ratio = partial(two_bands_operation, 'ratio')

NormedBandsDiff = partial(ModifySample, func=bands_normed_diff)
BandsDiff = partial(ModifySample, func=bands_diff)
BandsSum = partial(ModifySample, func=bands_sum)
BandsRatio = partial(ModifySample, func=bands_ratio)

__all__ = ['NormedBandsDiff',
           'BandsDiff',
           'BandsSum',
           'BandsRatio']
