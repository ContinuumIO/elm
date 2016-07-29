import glob
import logging
import os
import pickle
import re

import numpy as np
import rasterio as rio

from sklearn.externals import joblib

logger = logging.getLogger(__name__)

BOUNDS_FORMAT = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'

def split_model_tag(model_tag):
    parts = model_tag.split('.')
    if len(parts) > 2:
        raise ValueError('Expected at most one "." in '
                         'model name {}'.format(model_tag))
    if len(parts) == 1:
        tag = model_tag
        subtag = None
    else:
        tag, subtag = parts
    return tag, subtag

def get_paths_for_tag(elm_train_path, tag, subtags):
    paths = {}
    subtags = subtags or 'all'
    if subtags == 'all':
        ls = glob.glob(os.path.join(elm_train_path, tag, '*.pkl'))
        for f in ls:
            match = re.search('model-([_\w\d]+)-tag-([_\w\d]+).pkl', os.path.basename(f))
            if match:
                t, s = match.groups()
                paths[s] = f
    else:
        for subtag in subtags:
            model_root = os.path.join(elm_train_path, tag, 'model-{}-tag-{}.pkl'.format(tag, subtag))
            paths[subtag] = model_root
    paths['meta'] = os.path.join(elm_train_path, tag, 'model-{}_meta.pkl'.format(tag))
    return paths

def mkdir_p(path):
    '''Ensure the *dirname* of argument path has been created'''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def dump(data, path):
    joblib.dump(data, path)

def save_models_with_meta(models, elm_train_path, tag, meta):
    paths = get_paths_for_tag(elm_train_path, tag,
                              [_[0] for _ in models])
    paths_out = []
    for name, model in models:
        paths_out.append(paths[name])
        mkdir_p(paths[name])
        dump(model, paths[name])
    mkdir_p(paths['meta'])
    dump(meta, paths['meta'])
    return (paths_out, paths['meta'])

def load(path):
    return joblib.load(path)

def load_models_from_tag(elm_train_path, tag):
    tag, subtag = split_model_tag(tag)
    paths = get_paths_for_tag(elm_train_path, tag, subtag)
    logger.info('Pickles: {}'.format(paths))
    models = []
    for k, v in paths.items():
        if k == 'meta':
            continue
        models.append((k, load(v)))
    return (models, load(paths['meta']))

def drop_some_attrs(prediction):
    # I couldn't get it it to dump to netcdf
    # without dropping almost all the metadata
    prediction.attrs = {'GeoTransform': np.array(prediction.attrs['GeoTransform'])}

def predict_to_pickle(prediction, fname_base):
    dump(prediction, fname_base + '.xr')

def predict_to_netcdf(prediction, fname_base):
    mkdir_p(fname_base)
    prediction.sample.values = prediction.sample.values.astype('i4')
    drop_some_attrs(prediction)
    prediction.to_netcdf(fname_base + '.nc')

def band_to_tif(band, filename):
    kwargs = dict(
                driver='GTiff',
                dtype=rio.float32,
                count=1,
                compress='lzw',
                nodata=0,
                bigtiff='YES', # Output will be larger than 4GB
                width=band.shape[0],
                height=band.shape[1],

            )
    raise NotImplementedError('This band_to_tif function is not working - hanging '
                              'indefinitely')
    if 'crs' in band.attrs['MetaData']:
        kwargs['crs'] = band.attrs['MetaData']['crs']
    kwargs['transform'] = band.attrs['GeoTransform']
    mkdir_p(filename)
    print(filename)
    with rio.drivers():
        with rio.open(filename, 'w', **kwargs) as f:
            data = band.astype(rio.float32)
            f.write_band(1, data, window=((0, band.y.size), (0, band.x.size)))


def get_file_name(base, tag, bounds):
    return os.path.join(base,
                        tag,
                        BOUNDS_FORMAT.format(bounds.left,
                                   bounds.bottom,
                                   bounds.right,
                                   bounds.top))


def predict_file_name(elm_predict_path, tag, bounds):
    return get_file_name(elm_predict_path, tag, bounds)

def transform_file_name(elm_transform_path, tag, bounds):
    return get_file_name(elm_transform_path, tag, bounds)


