import gdal
import re

from settings import delayed
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

def _select_band_from_file_base(filename,
                         band_specs,
                         include_polys=None,
                         filter_on_metadata=None,
                         filter_on_filename=None,
                         filename_search=None,
                         data_filter=None,
                         dry_run=False):
    keep_file = _filter_on_filename(filename,
                                    search=filename_search,
                                    func=filter_on_filename)
    if not keep_file:
        return
    handle, ds, filemeta = load_hdf4(filename)
    if filter_on_metadata is not None:
        keep_file = filter_on_metadata(filename, filemeta, ds, handle=handle)
        if not keep_file:
            return
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
                                              include_polys, data_filter,
                                              band_meta, bounds,
                                              idxes=idxes,
                                              lons=lons,
                                              lats=lats)
        band_data.append((band_name, values))
    print(bounds)
    band_data.extend((('lon', lons), ('lat', lats), ('time', time)))
    joined_df = pd.DataFrame(OrderedDict(band_data))
    joined_df.set_index(['lon', 'lat', 'time'], inplace=True, drop=True)
    if data_filter is not None and not dry_run:
        # E.g. skip clouds by writing your own
        # spectral filter or data_filter
        # is a partial function that can
        # select using DEM or NDVI
        joined_df, lons, lats = data_filter(joined_df,
                                         lons,
                                         lats,
                                         meta=band_meta,
                                         bounds=bounds,
                                         time=time)

    return (joined_df, band_meta, filemeta)


@delayed
def select_band_from_file(*args, **kwargs):
    return _select_band_from_file_base(*args, **kwargs)
