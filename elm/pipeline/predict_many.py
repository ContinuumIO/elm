# TODO update predict_many for Elm PR 192 changes and connect to
# elm.model_selection.* and dask_searchcv.model_selection.*
# estimators like EaSearchCV so they can call
# grid.fit(X, y).predict_many(X) predicting from all
# or a subset of the final generation of estimators
# from EA search process.

from __future__ import absolute_import, division, print_function

from functools import partial
import copy
import datetime
import itertools
import logging
import os

import dask
try:
    from earthio import check_X_data_type
    from earthio.reshape import inverse_flatten
    from xarray_filters import MLDataset
except:
    inverse_flatten = check_X_data_type = MLDataset = None # TODO handle case where earthio not installed
import numpy as np
import xarray as xr


from elm.config import import_callable, parse_env_vars

logger = logging.getLogger(__name__)

__all__ = ['predict_many',]

_next_idx = -1

def _next_name(token):
    '''name in a dask graph'''
    global _next_idx
    n = '{}-{}'.format(token, _next_idx)
    _next_idx += 1
    return n


def _predict_one_sample_one_arg(estimator,
                                serialize,
                                to_raster,
                                predict_tag,
                                elm_predict_path,
                                X_y_sample_weight):
    X, y, sample_weight = X_y_sample_weight
    check_X_data_type(X)
    out = []
    prediction, X_final = estimator.predict(X, return_X=True)
    if prediction.ndim == 1:
        prediction = prediction[:, np.newaxis]
        ndim = 2
    elif prediction.ndim == 2:
        pass
    else:
        raise ValueError('Expected 1- or 2-d output of model.predict but found ndim of prediction: {}'.format(prediction.ndim))

    layers = ['predict']
    attrs = X_final.attrs
    attrs.update(X_final.flat.attrs)
    attrs['elm_predict_date'] = datetime.datetime.utcnow().isoformat()
    attrs['layer_order'] = ['predict',]
    attrs['canvas'] = getattr(X_final.flat, 'canvas', None)
    logger.debug('Predict X shape {} X.flat.dims {} '
                 '- y shape {}'.format(X_final.flat.shape, X_final.flat.dims, prediction.shape))
    prediction = MLDataset({'flat': xr.DataArray(prediction,
                                     coords=[('space', X_final.flat.space),
                                             ('layer', layers)],
                                     dims=('space', 'layer'),
                                     attrs=attrs)},
                             attrs=attrs,
                             add_canvas=False)
    if to_raster:
        new_es = inverse_flatten(prediction, add_canvas=False)
    else:
        new_es = prediction
    if serialize:
        new_es = serialize(y=new_es, X=X_final, tag=predict_tag,
                           elm_predict_path=elm_predict_path)
    out.append(new_es)
    return out


def predict_many(data_source,
                 saved_model_tag=None,
                 ensemble=None,
                 client=None,
                 serialize=None,
                 to_raster=True,
                 elm_predict_path=None):
    '''See elm.pipeline.Pipeline.predict_many method

    '''

    env = parse_env_vars()
    elm_predict_path = elm_predict_path or env.get('ELM_PREDICT_PATH')
    if serialize and elm_predict_path and not os.path.exists(elm_predict_path):
        os.mkdir(elm_predict_path)
    pipe_example = ensemble[0][1]
    ds = data_source.copy()
    X = ds.pop('X', None)
    y = ds.pop('y', None)
    args_list = ds.pop('args_list', None)
    sampler = ds.pop('sampler', None)
    dsk = make_samples_dask(X, y, None, pipe_example, args_list, sampler, ds)
    sample_keys = tuple(dsk)
    args_list = tuple(itertools.product(sample_keys, ensemble))
    keys = []
    last_file_name = None
    for idx, (sample_key, (estimator_tag, estimator)) in enumerate(args_list):
        name = _next_name('predict_many')
        predict_tag = '{}-{}'.format(estimator_tag, sample_key)
        if saved_model_tag:
            predict_tag += '-' + saved_model_tag
        dsk[name] = (_predict_one_sample_one_arg,
                     estimator,
                     serialize,
                     to_raster,
                     predict_tag,
                     elm_predict_path,
                     sample_key,)


        keys.append(name)
    logger.info('Predict {} estimator(s) and {} sample(s) '
                '({} combination[s])'.format(len(ensemble),
                                         len(sample_keys),
                                         len(args_list)))
    preds = []
    if client is None:
        new = dask.get(dsk, keys)
    else:
        new = client.get(dsk, keys)
    return tuple(itertools.chain.from_iterable(new))
