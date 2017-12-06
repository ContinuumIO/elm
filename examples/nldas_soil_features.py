from __future__ import print_function, division
import numpy as np

from read_nldas_soils import SOIL_META, read_nldas_soils

_endswith = lambda x, end: x.endswith('_{}'.format(end))

def _avg_cos_hyd_params(soils_dset, attrs=None):
    from ts_raster_steps import reduce_series
    attrs = attrs or soils_dset.attrs.copy(deep=True)
    skip = ('i', 'j', 'x', 'y', 'unknown')
    keep = [x[0] for x in SOIL_META['COS_HYD']
            if x[0] not in skip]
    arrs = {}
    groups = [(keep2, [k for k in soils_dset.data_vars
                       if _endswith(k, keep2)])
              for keep2 in keep]
    for array_label, keys in groups:
        arr = reduce_series('mean', [1] * len(keys),
                            tuple(soils_dset[k] for k in keys))
        arrs[array_label] = arr
    for array_label, arr in soils_dset.data_vars.items():
        if not any(_endswith(array_label, keep2) in x for x in keep):
            arrs[array_label] = arr
    return xr.Dataset(arrs, attrs=attrs)


def flatten_layers(soils_dset, attrs=None):
    arrs = {}
    attrs = attrs or soils_dset.attrs.copy(deep=True)
    for k, v in soils_dset.data_vars.items():
        if 'layer' in v.dims and to_raster:
            which_dim = v.dims.index('layer')
            for idx, layer in enumerate(v.layer):
                slc = [slice(None)] * 3
                array_label = '{}_{}'.format(k, layer)
                arrs[array_label] = v[slc]
                arrs[array_label].attrs['layer'] = layer
        else:
            arrs[k] = v
    return xr.Dataset(arrs, attrs=attrs)


def nldas_soil_features(soils_dset=None,
                        to_raster=True,
                        avg_cos_hyd_params=True,
                        **kw):

    if soils_dset is None:
        soils_dset = read_nldas_soils(**kw)
    if avg_cos_hyd_params:
        soils_dset = _avg_cos_hyd_params(soils_dset)
    if to_raster:
        soils_dset = flatten_layers(soils_dset)
    meta = dict(to_raster=to_raster, avg_cos_hyd_params=avg_cos_hyd_params)
    soils_dset.attrs['soil_features_kw'] = meta
    return soils_dset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read NLDAS inputs-related soil data from https://ldas.gsfc.nasa.gov/nldas/NLDASsoils.php')
    parser.add_argument('--to-raster', action='store_true')
    parser.add_argument('--avg-cos-hyd-params', action='store_true')
    soils_dset = nldas_soil_features(**vars(parser.parse_args()))
