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

    '''select_from_file is the typical sampler used in the elm config
    file interface system via elm.pipeline.parse_run_config

    It is called twice: once to determine a list of files to get (dry_run
    =True is to check for files to sample), then again to read a given file.

    Parameters:
        sampler_args: tuple of one element - a filename
        band_specs: list of band_specs included in a data_source
        metadata_filter: ignored
        filename_search: a search token for a filenames
        filename_filter: a function that returns True/False to keep filename
        dry_run:  if True, don't actually read file, just return True if accepted
        load_meta: Function, typically from elm.readers, to load metadata
        load_array: Function, typically from elm.readers, to load ElmStore
        kwargs:
            may contain "reader" such as "hdf4", "tif", "hdf5", "netcdf"

    '''
    from elm.sample_util.filename_selection import _filename_filter
    filename = sampler_args[0]
    keep_file = _filename_filter(filename,
                                 search=filename_search,
                                 func=filename_filter)
    logger.debug('Filename {} keep_file {}'.format(filename, keep_file))
    args_required, default_kwargs, var_keywords = get_args_kwargs_defaults(load_meta)
    if dry_run:
        return True
    sample = load_array(filename, meta=meta, band_specs=band_specs, reader=kwargs.get('reader', None))
    return sample
