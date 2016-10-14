from functools import partial
import copy
import datetime
import itertools
import logging
import os

import dask
import numpy as np
import xarray as xr


from elm.config import import_callable, parse_env_vars
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config.dask_settings import (wait_for_futures,
                                        no_client_submit)
from elm.sample_util.sample_pipeline import create_sample_from_data_source
from elm.pipeline.serialize import (load_models_from_tag,
                                    serialize_prediction)
from elm.readers import (flatten,
                         inverse_flatten,
                         ElmStore,)
from elm.sample_util.samplers import make_samples_dask

logger = logging.getLogger(__name__)

__all__ = ['predict_many',]

_predict_idx = 0

def _next_name():
    global _predict_idx
    n = 'predict-{}'.format(_predict_idx)
    _predict_idx += 1
    return n


def _predict_one_sample_one_arg(estimator,
                                serialize,
                                to_cube,
                                predict_tag,
                                X_y_sample_weight):
    X, y, sample_weight = X_y_sample_weight
    if not isinstance(X, (ElmStore, xr.Dataset)):
        raise ValueError('Expected an ElmStore or xarray.Dataset')
    out = []
    prediction, X_final = estimator.predict(X, return_X=True)
    if prediction.ndim == 1:
        prediction = prediction[:, np.newaxis]
        ndim = 2
    elif prediction.ndim == 2:
        pass
    else:
        raise ValueError('Expected 1- or 2-d output of model.predict but found ndim of prediction: {}'.format(prediction.ndim))

    bands = ['predict']
    attrs = X_final.attrs
    attrs.update(X_final.flat.attrs)
    attrs['elm_predict_date'] = datetime.datetime.utcnow().isoformat()
    attrs['band_order'] = ['predict',]
    logger.debug('Predict X shape {} X.dims {} '
                 '- y shape {}'.format(X_final.flat.shape, X_final.flat.dims, prediction.shape))
    prediction = ElmStore({'flat': xr.DataArray(prediction,
                                     coords=[('space', X_final.flat.space),
                                             ('band', bands)],
                                     dims=('space', 'band'),
                                     attrs=attrs)},
                             attrs=attrs)
    if to_cube:
        new_es = inverse_flatten(prediction)
    else:
        new_es = prediction
    if serialize:
        new_es = serialize(new_es, sample, predict_tag)
    out.append(new_es)
    return out


def predict_many(data_source,
                 saved_model_tag=None,
                 tagged_models=None,
                 client=None,
                 serialize=None,
                 to_cube=True,
                 elm_train_path=None):

    get_results = partial(wait_for_futures, client=client)

    env = parse_env_vars()
    if serialize == serialize_prediction:
        serialize = partial(serialize, config)

    if tagged_models is None:
        if saved_model_tag is None:
            raise ValueError('Expected saved_model_tag to be not None when tagged_models is None')
        etp = elm_train_path or parse_env_vars()['ELM_TRAIN_PATH']
        logger.info('Load pickled tagged_models from {} {}'.format(etp, saved_model_tag))
        tagged_models, meta = load_models_from_tag(etp, saved_model_tag)

    pipe_example = tagged_models[0][1]
    dsk = make_samples_dask(data_source.get('X'),
                            data_source.get('y'),
                            None,
                            pipe_example,
                            data_source.get('args_list'),
                            data_source.get('sampler'))
    sample_keys = tuple(dsk)
    args_list = tuple(itertools.product(sample_keys, tagged_models))
    keys = []
    last_file_name = None
    for idx, (sample_key, (estimator_tag, estimator)) in enumerate(args_list):
        name = _next_name()
        predict_tag = '{}-{}'.format(estimator_tag, sample_key)
        if saved_model_tag:
            predict_tag += '-' + saved_model_tag
        dsk[name] = (_predict_one_sample_one_arg,
                     estimator,
                     serialize,
                     to_cube,
                     predict_tag,
                     sample_key)


        keys.append(name)
    preds = []
    if client is None:
        new = dask.get(dsk, keys)
    else:
        new = client.get(dsk, keys)
    return tuple(itertools.chain.from_iterable(new))
