
import rasterio as rio

from elm.sample_util.band_selection import match_meta

def load_tif_meta(filename):
    r = rio.open(filename)
    meta = {'MetaData': {}}
    meta['GeoTransform'] = r.get_transform()
    meta['Bounds'] = r.bounds()
    return meta, r.meta, r.name

def ls_tif_files(dir_of_tiffs):
    tifs = os.listdir(dir_of_tiffs)
    tifs = [f for f in tifs in f.lower().endswith('.tif') or f.lower.endswith('.tiff')]
    return tifs

def load_dir_of_tifs_meta(dir_of_tiffs):
    tifs = ls_tif_files(dir_of_tiffs)
    meta = {'Metadata': {}}
    band_metas = []
    for tif in tifs:
        m, band_meta = load_tif_meta(tif)
        meta.update(m)
        band_metas.append(band_meta)
        band_metas[-1]['name'] = r.name
    meta['BandMetaData'] = band_metas
    meta['Height'] = r.height
    meta['Width'] = r.width

    return meta

def open_prefilter(filename):
    '''Placeholder for future operations on open file handle
    like resample / aggregate '''
    r = rio.open(filename)
    return r, r.read()

def load_dir_of_tifs_array(dir_of_tiffs, meta, band_specs):
    keeping = []
    tifs = ls_tif_files(dir_of_tiffs)
    for band_meta in meta['BandMetaData']:
        filename = os.path.join(dir_of_tiffs, os.path.dirname(band_meta['name']))
        for idx, band_spec in enumerate(band_specs):
            band_name = match_meta(band_meta, band_spec)
            if band_name:
                keeping.append((idx, band_meta, filename, band_name))
                break
    keeping.sort(key=lambda x:x[0])
    if not len(keeping):
        raise ValueError('No matching bands with band_specs {}'.format(band_specs))

    idx, _, filename, band_name = keeping[0]
    _, arr = open_prefilter(filename)
    shp = (len(keeping),) + arr.shape

    store = np.empty(shp, dtype=arr.dtype)
    store[0, :, :] = arr
    if len(keeping) > 1:
        for idx, _, filename, band_name in keeping[1:]:
            handle, raster = open_prefilter(filename)
            store[idx, :, :] = raster
            del raster
            gc.collect()
    band_labels = [_[-1] for _ in band_specs]
    latitude, longitude = geotransform_to_dims(handle.width, handle.height, meta['GeoTransform'])
    band_data = xr.DataArray(store,
                           coords=[('band', band_labels),
                                   ('latitude', latitude),
                                   ('longitude', longitude),
                                   ],
                           dims=['band','lat','long',],
                           attrs=meta)

    return ElmStore({'sample': band_data})