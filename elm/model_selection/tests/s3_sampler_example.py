import os
from earthio.s3_landsat_util import SceneDownloader
from earthio import load_array, BandSpec
from earthio.landsat_util import landsat_metadata


BAND_SPECS = [BandSpec(search_key='name',
                       search_value='B{}.TIF'.format(band),
                       name='band_{}'.format(band),
                       buf_xsize=800,
                       buf_ysize=800) for band in range(1, 8)]


def get_download_url(row=33, path=15, months=tuple(range(1,13))):
    s3_download = SceneDownloader(s3_tif_dir=os.environ.get('S3_LANDSAT_DIR'))
    clear_image = s3_download.lowest_cloud_cover_image(row=row, path=path,
                                                       months=months)
    download_url = clear_image.download_url.values[0]
    return download_url


def sampler(download_url, **kwargs):
    download_url = get_download_url(**kwargs)
    local_files = s3_download.download_all_bands(download_url)
    this_sample_dir = os.path.dirname(local_files[0])
    X = load_array(this_sample_dir, band_specs=BAND_SPECS)
    landsat_meta = [f for f in local_files if f.endswith('.txt')][0]
    X.attrs.update(vars(landsat_metadata(local_files)))
    return X

