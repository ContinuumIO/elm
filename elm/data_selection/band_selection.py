from collections import OrderedDict

from gdalconst import GA_ReadOnly
import gdal
import logging
import pandas as pd
import re

from elm.config import delayed
from elm.readers.hdf4_L2_tools import load_hdf4, get_subdataset_bounds

logger = logging.getLogger(__name__)

def match_meta(meta, band_spec):
    search_key, search_value, name = band_spec
    for mkey in meta:
        if bool(re.search(search_key, mkey)):
            if bool(re.search(search_value, meta[mkey])):
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
                         geo_filters=None,
                         metadata_filter=None,
                         filename_filter=None,
                         filename_search=None,
                         data_filter=None,
                         dry_run=False,
                         file_loader=load_hdf4,
                         get_subdataset_bounds=get_subdataset_bounds,
                         **kwargs):
    '''
    Form a selection from a filename based on matching band metadata,
    matching a filename filter or search word, filtering with data_filter
    func.

    Params:
        filename:  string name of file
        band_specs: list of 3-item sublists
        include_polys: list of polygons expressed as 2-column (X, Y)
                       numpy float64 arrays
        metadata_filter: None or a function taking
                         (filename, file metadata, datasets, handle=handle)
                         as a signature.  Can exclude a file by
                         scanning its metadata or datasets.  Should
                         return bool to keep file or not as sample.
        filename_filter: callable returning True or False whether to
                         keep a filename.  Takes filename as single arg
        filename_search: a string compiled as a reg expression or None
                         if not None, filenames must match this search
                         string.
        data_filter:     callable that is called on the assembled sample.
                         data_filter should have the signature:

                         def data_filter(sample_dataframe, **kwargs):
        dry_run:         do as much of the logic as possible but do
                         not actually open file or load arrays
        file_loader:     function taking a filename and returning a tuple
                         of (handle, datasets, filemeta)
                         TODO: note this file_loader return signature
                         is geared toward the HDF4 files. We may want
                         to make it more file-type-independent.
        get_subdataset_bounds: A function with band_meta as single argument,
                         it should return the bounds of the sample
                         as a named tuple with fields north, south
                         east, west.
        kwargs:          not used currently
    '''
    from elm.data_selection.geo_selection import _filter_band_data
    from elm.data_selection.filename_selection import _filename_filter
    if kwargs:
        raise NotImplementedError('Not sure what to do with kwargs '
                                  '{} passed in'.format(kwargs))
    keep_file = _filename_filter(filename,
                                    search=filename_search,
                                    func=filename_filter)
    geo_filters = geo_filters or {}
    include_polys = geo_filters.get('include_polys') or []
    exclude_polys = geo_filters.get('exclude_polys') or []
    if not keep_file:
        return False
    # TODO this section needs to be generalized
    # for non-HDF4 files as well which may
    # not have the same pattern of
    # returning datasets and file metadata
    handle, ds, filemeta = file_loader(filename)
    if metadata_filter is not None:
        keep_file = metadata_filter(filename, filemeta, ds, handle=handle)
        if not keep_file:
            return False
    if dry_run:
        return True
    idxes = None
    last_bounds = None
    last_time = None
    # this call of tuple is essential for some reason
    # without you get system errors related to something in gdal
    keep_bands = tuple(get_bands(handle, ds, *band_specs))
    handle = None
    joined_df = None
    lons, lats = None, None
    band_data = []
    for (band_idx, (subhandle, band_meta, band_name)) in enumerate(keep_bands):
        bounds, time = get_subdataset_bounds(band_meta)
        values, lons, lats, idxes =  _filter_band_data(handle, subhandle, time,
                                              include_polys, exclude_polys, data_filter,
                                              band_meta, bounds,
                                              idxes=idxes,
                                              lons=lons,
                                              lats=lats)
        band_data.append((band_name, values))
        logger.info(repr(bounds))
    if not band_data:
        raise ValueError('Empty sample with filename {} '
                         'band_specs {} (nothing '
                         'matches)'.format(filename, band_specs))
    band_data.extend((('lon', lons), ('lat', lats), ('time', time)))
    joined_df = pd.DataFrame(OrderedDict(band_data))
    joined_df.set_index(['lon', 'lat', 'time'], inplace=True, drop=True)
    if data_filter is not None and not dry_run:
        # E.g. skip clouds by writing your own
        # spectral filter or data_filter
        # is a partial function that can
        # select using DEM or NDVI
        joined_df, lons, lats = data_filter(joined_df,
                                         lons=lons,
                                         lats=lats,
                                         meta=band_meta,
                                         bounds=bounds,
                                         time=time)

    return (joined_df, band_meta, filemeta)



def select_from_file(*args, **kwargs):
    return _select_from_file_base(*args, **kwargs)
