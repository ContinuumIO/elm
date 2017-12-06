# TODO - how does this Phase I module relate to sklearn.mldataset.serialize_mixin
from __future__ import absolute_import, division, print_function

'''
----------------------

``elm.pipeline.serialize``
~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import glob
import logging
import os
import pickle
import re

import dill
import numpy as np


__all__ = ['serialize_pipe', 'serialize_prediction']

logger = logging.getLogger(__name__)

BOUNDS_FORMAT = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'


def _get_path_for_tag(elm_train_path, tag):
    return os.path.join(elm_train_path, tag + '.pkl')


def mkdir_p(path):
    '''Ensure the *dirname* of argument path has been created'''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def serialize_pipe(pipe, elm_train_path, tag, **meta):
    '''Save a Pipeline to a tag in elm_train_path

    Parameters:
        :pipe: an elm.pipeline.Pipeline instance
        :elm_train_path: root dir for serializing trained ensembles
        :tag: tag for the ensemble
        :\*\*meta: ignored

    Returns: ``None``

    This function is used in the elm config file interface.  See also :func:``elm.pipeline.parse_run_config``
    '''
    logger.debug('Save pipe at {} with tag {}'.format(elm_train_path, tag))
    mkdir_p(elm_train_path)
    path = _get_path_for_tag(elm_train_path, tag)
    return pipe.save(path)


def load_pipe_from_tag(elm_train_path, tag):
    '''Calls Pipeline.load for a tagged saved Pipeline in elm_train_path

    Parameters:
        :elm_train_path:  root dir for serializing training outputs
        :tag:             tag that was given to elm.pipeline.serialize.serialize_pipe

    Returns:
        :elm.pipeline.Pipeline: instance (fitted if it was fitted before saving)
        '''
    from elm.pipeline import Pipeline
    logger.debug('Load {} from {}'.format(tag, elm_train_path))
    path = _get_path_for_tag(elm_train_path, tag)
    if not os.path.exists(path):
        raise IOError('Cannot load from {} (does not exist)'.format(path))
    return Pipeline.load(path)


def predict_to_pickle(prediction, fname_base):
    '''Dump a prediction y data'''
    mkdir_p(fname_base)
    fname = fname_base + '.xr'
    with open(fname, 'wb') as f:
        return dill.dump(prediction, f)


def predict_file_name(elm_predict_path, tag, bounds):
    '''Form a file name from bounds'''
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
        :config:  elm.config.ConfigParser instance or None
                  if elm_predict_path in kwargs or ELM_PREDICT_PATH
                  in environment variables
        :y:       y prediction MLDataset
        :X:       X MLDataset that predicted y
        :tag:     unique tag based on sample, estimator, ensemble
        :kwargs:  keywords may contain:
                  elm_predict_path: defaulting

    Returns: ``True``

    A partial of this function is used by elm.pipeline.parse_run_config

    Assumes:

        X has a earthio.Canvas object with a "bounds" attr
        (bounds are used in the filenaming, assuming that X
        is taken from different images in space)
    '''
    if not config:
        root = kwargs.get('elm_predict_path')
        if not root:
            root = parse_env_vars()['ELM_PREDICT_PATH']
    else:
        root = config.ELM_PREDICT_PATH
    for layer in X.data_vars:
        layer_arr = getattr(X, layer)
        fname = predict_file_name(root,
                                  tag,
                                  getattr(layer_arr, 'canvas', getattr(X, 'canvas')).bounds)
        predict_to_pickle(y, fname)
    return True
