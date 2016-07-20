from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging
import pandas as pd
import re

from elm.config import delayed
from elm.readers.hdf4 import load_hdf4_array, load_hdf4_meta
from elm.model_selection.util import get_args_kwargs_defaults

logger = logging.getLogger(__name__)

def match_meta(meta, band_spec):
    search_key, search_value, name = band_spec
    for mkey in meta:
        if bool(re.search(search_key, mkey, re.IGNORECASE)):
            if bool(re.search(search_value, meta[mkey], re.IGNORECASE)):
                return name
    return False

def get_bands(handle, ds, *band_specs):
    for ds_name, label in ds:
        found_bands = 0
        for band_spec in band_specs:
            subhandle = gdal.Open(ds_name, GA_ReadOnly)
            meta = subhandle.GetMetadata()
            name = match_meta(meta, band_spec)
            if name:
                found_bands += 1
                yield subhandle, meta, name
            else:
                subhandle = None
        if found_bands == len(band_specs):
            break

def _select_from_file_base(filename,
                         band_specs,
                         include_polys=None,
                         metadata_filter=None,
                         filename_filter=None,
                         filename_search=None,
                         dry_run=False,
                         load_meta=None,
                         load_array=None,
                         **kwargs):
    from elm.sample_util.geo_selection import _filter_band_data
    from elm.sample_util.filename_selection import _filename_filter

    keep_file = _filename_filter(filename,
                                 search=filename_search,
                                 func=filename_filter)
    args_required, kwargs_default = get_args_kwargs_defaults(load_meta)
    if len(args_required) == 1:
        meta = load_meta(filename)
    else:
        meta = load_meta(filename, band_specs)
    if metadata_filter is not None:
        keep_file = metadata_filter(filename, meta)
        if not keep_file:
            return False
    # TODO rasterio filter / resample / aggregate
    if dry_run:
        return True
    sample = load_array(filename, meta, band_specs)
    # TODO points in poly
    return sample


def select_from_file(*args, **kwargs):
    return _select_from_file_base(*args, **kwargs)

def include_file(*args, **kwargs):
    kwargs['dry_run'] = True
    return _select_from_file_base(*args, **kwargs)
