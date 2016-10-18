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


def predict_to_pickle(prediction, fname_base):
    joblib.dump(prediction, fname_base + '.xr')


def predict_file_name(elm_predict_path, tag, bounds):
    fmt = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'
    return os.path.join(elm_predict_path,
                        tag,
                        fmt.format(bounds.left,
                                   bounds.bottom,
                                   bounds.right,
                                   bounds.top))


def serialize_prediction(config, y, X, tag, **kwargs):
    '''This function is called by elm.pipeline.parse_run_config
    to serialize the prediction outputs of models run through
    the elm config file interface

    Parameters:
        config:  elm.config.ConfigParser instance
        y:       y prediction ElmStore
        X:       X ElmStore that predicted y
        tag:     unique tag based on sample, estimator, ensemble
        kwargs:  keywords may contain:
                 elm_predict_path: defaulting '''
    if not config:
        root = kwargs.get('elm_predict_path')
        if not root:
            root = parse_env_vars()['ELM_PREDICT_PATH']
    else:
        root = config.ELM_PREDICT_PATH
    for band in X.data_vars:
        band_arr = getattr(X, band)
        fname = predict_file_name(root,
                                  tag,
                                  getattr(band_arr, 'canvas', getattr(X, 'canvas')).bounds)
        predict_to_pickle(y, fname)
    return True
