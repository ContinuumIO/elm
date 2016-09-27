from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging
import pandas as pd
import re

from elm.readers.util import BandSpec
from elm.model_selection.util import get_args_kwargs_defaults
from elm.sample_util.util import InvalidSample

logger = logging.getLogger(__name__)

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
    logger.debug('filename {} keep_file {}'.format(filename, keep_file))
    args_required, default_kwargs, _ = get_args_kwargs_defaults(load_meta)
    if len(args_required) == 1 and not 'band_specs' in default_kwargs:
        meta = load_meta(filename)
    else:
        meta = load_meta(filename, band_specs=band_specs)
    if metadata_filter is not None:
        keep_file = metadata_filter(filename, meta)
        if dry_run:
           return keep_file

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
