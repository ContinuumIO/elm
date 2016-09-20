
import numpy as np
import xarray as xr

from elm.readers.elm_store import ElmStore
from elm.readers import check_is_flat

def elm_store_concat(*elms):
    elms2 = []
    sample_ys, sample_weights = [], []
    for es in elms:
        es = es[0]
        if isinstance(es, (list, tuple)):
            assert check_is_flat(es[0])
            elms2.append(es[0])
            if es[1] is not None:
                sample_ys.append(es[1])
            if es[2] is not None:
                sample_weights.append(es[2])
    elms = elms2
    if sample_ys:
        sample_y = np.concatenate(sample_ys)
    else:
        sample_y = None
    if sample_weights:
        sample_weight = np.concatenate(sample_weights)
    else:
        sample_weight = None
    shp = set(es.flat.values.shape for es in elms)
    assert len(shp) == 1
    shp = tuple(shp)[0]
    new_shp = (len(elms) * shp[0], shp[1])
    store = np.empty(new_shp) * np.NaN
    es_indicator = []
    attrs = {}
    for idx, es in enumerate(elms):
        es_indicator.extend([idx] * es.flat.space.size)
        attrs[idx] = es.attrs
    es_indicator = np.array(es_indicator, dtype=np.uint8)
    space_concat = np.concatenate(tuple(es.flat.space for es in elms))
    attrs['indicator'] = es_indicator
    coords = [('space', space_concat),
              ('band', elms[0].band),]
    dims = ('space', 'band')
    row_count = 0
    for idx, es in enumerate(elms):
        store[row_count: row_count + shp[0], :] = es.flat.values
        row_count += shp[0]
    es = ElmStore({'flat': xr.DataArray(store, coords=coords,
                                        dims=dims, attrs=attrs)},
                  attrs=attrs)
    return (es, sample_y, sample_weight)
