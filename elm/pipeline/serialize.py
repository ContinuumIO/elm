import glob
import logging
import os
import pickle
import re

import attr
import numpy as np

from sklearn.externals import joblib

__all__ = ['serialize_pipe', 'serialize_prediction']

logger = logging.getLogger(__name__)

BOUNDS_FORMAT = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'


def get_path_for_tag(elm_train_path, tag):
    return os.path.join(elm_train_path, tag + '.pkl')


def mkdir_p(path):
    '''Ensure the *dirname* of argument path has been created'''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def serialize_pipe(pipe, elm_train_path, tag, **meta):
    logger.debug('Save pipe at {} with tag {}'.format(elm_train_path, tag))
    mkdir_p(elm_train_path)
    path = get_path_for_tag(elm_train_path, tag)
    return pipe.save(path)


def load_pipe_from_tag(elm_train_path, tag):
    from elm.pipeline import Pipeline
    logger.debug('Load {} from {}'.format(tag, elm_train_path))
    path = get_path_for_tag(elm_train_path, tag)
    if not os.path.exists(path):
        raise IOError('Cannot load from {} (does not exist)'.format(path))
    return Pipeline.load(path)


def _prepare_serialize_prediction(prediction):
    # I couldn't get it it to dump to netcdf
    # without dropping almost all the metadata
    for band in prediction.data_vars:
        band_arr = getattr(prediction, band)
        c = attr.asdict(band_arr.canvas)
        for k,v in c.items():
            if isinstance(v, (tuple, list)):
                c[k] = np.array(v)
            else:
                if v is None:
                    v = np.NaN
                c[k] = np.array([v])
        band_arr.attrs = c
        prediction.attrs = c


def predict_to_pickle(prediction, fname_base):
    joblib.dump(prediction, fname_base + '.xr')


def predict_to_netcdf(prediction, fname_base):
    mkdir_p(fname_base)
    try:
        _prepare_serialize_prediction(prediction)
        prediction.to_netcdf(fname_base + '.nc')
    except:
        logger.info('Likely failed serializing attrs {}'.format(prediction.attrs))
        raise


def get_file_name(base, tag, bounds):
    return os.path.join(base,
                        tag,
                        BOUNDS_FORMAT.format(bounds.left,
                                   bounds.bottom,
                                   bounds.right,
                                   bounds.top))


def predict_file_name(elm_predict_path, tag, bounds):
    fmt = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'
    return os.path.join(elm_predict_path,
                        tag,
                        fmt.format(bounds.left,
                                   bounds.bottom,
                                   bounds.right,
                                   bounds.top))



def serialize_prediction(config, prediction, sample, tag):
    if not config:
        root = parse_env_vars()['ELM_PREDICT_PATH']
    else:
        root = config.ELM_PREDICT_PATH
    for band in sample.data_vars:
        band_arr = getattr(sample, band)
        fname = predict_file_name(root,
                                  tag,
                                  getattr(band_arr, 'canvas', getattr(sample, 'canvas')).bounds)
        predict_to_netcdf(prediction, fname)
        predict_to_pickle(prediction, fname)
    return True
