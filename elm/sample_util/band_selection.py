from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults

logger = logging.getLogger(__name__)


def select_from_file(*sampler_args,
                     band_specs=None,
                     metadata_filter=None,
                     filename_filter=None,
                     filename_search=None,
                     dry_run=False,
                     load_meta=None,
                     load_array=None,
                     **kwargs):
    from elm.sample_util.filename_selection import _filename_filter
    filename = sampler_args[0]
    keep_file = _filename_filter(filename,
                                 search=filename_search,
                                 func=filename_filter)
    logger.debug('Filename {} keep_file {}'.format(filename, keep_file))
    args_required, default_kwargs, var_keywords = get_args_kwargs_defaults(load_meta)
    if len(args_required) == 1 and not 'band_specs' in default_kwargs:
        meta = load_meta(filename)
    else:
        meta = load_meta(filename, band_specs=band_specs)
    if metadata_filter is not None:
        keep_file = import_callable(metadata_filter)(filename, meta)
        if dry_run:
           return keep_file

    # TODO rasterio filter / resample / aggregate
    if dry_run:
        return True
    sample = load_array(filename, meta=meta, band_specs=band_specs, reader=kwargs.get('reader', None))
    # TODO points in poly
    return sample
