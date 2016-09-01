from functools import partial
import copy
import datetime
import itertools
import logging
import os

import numpy as np
import xarray as xr


from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config.dask_settings import (wait_for_futures,
                                        no_executor_submit)
from elm.sample_util.sample_pipeline import get_sample_pipeline_action_data
from elm.pipeline.serialize import (predict_to_netcdf,
                                    predict_to_pickle,
                                    load_models_from_tag,
                                    predict_file_name)
from elm.sample_util.sample_pipeline import run_sample_pipeline
from elm.readers import (flatten,
                         inverse_flatten,
                         ElmStore,)

logger = logging.getLogger(__name__)

def predict_file_name(elm_predict_path, tag, bounds):
    fmt = '{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}'
    return os.path.join(elm_predict_path,
                        tag,
                        fmt.format(bounds.left,
                                   bounds.bottom,
                                   bounds.right,
                                   bounds.top))


def _predict_one_sample(action_data, serialize, model,
                        return_serialized=True, to_cube=True,
                        sample=None, transform_model=None, canvas=None):
    # TODO: control to_cube from config

    name, model = model
    sample, sample_y, sample_weight = run_sample_pipeline(action_data,
                                 sample=sample,
                                 transform_model=transform_model)
    assert hasattr(sample, 'flat')
    canvas = canvas or sample.canvas
    prediction = model.predict(sample.flat.values)
    if prediction.ndim == 1:
        prediction = prediction[:, np.newaxis]
        ndim = 2
    elif prediction.ndim == 2:
        pass
    else:
        raise ValueError('Expected 1- or 2-d output of model.predict but found ndim of prediction: {}'.format(prediction.ndim))

    bands = ['predict']

    attrs = copy.deepcopy(sample.attrs)
    attrs.update(copy.deepcopy(sample.flat.attrs))
    assert 'canvas' in attrs
    attrs['elm_predict_date'] = datetime.datetime.utcnow().isoformat()
    prediction = ElmStore({'flat': xr.DataArray(prediction,
                                     coords=[('space', sample.flat.space),
                                             ('band', bands)],
                                     dims=('space', 'band'),
                                     attrs=attrs)},
                             attrs=attrs)
    if to_cube:
        new_es = inverse_flatten(prediction)
    else:
        new_es = prediction
    if return_serialized:
        return serialize(new_es, sample)
    return prediction

def _predict_one_sample_one_arg(action_data, transform_model, serialize, to_cube, arg):
    filename, model = arg
    logger.info('Predict {}'.format(filename))
    action_data_copy = copy.deepcopy(action_data)
    action_data_copy[0][-1]['filename'] = filename
    return _predict_one_sample(action_data_copy,
                               serialize,
                               model,
                               to_cube=to_cube,
                               transform_model=transform_model)


def _default_serialize(tag, config, prediction, sample):
    for band in sample.data_vars:
        band_arr = getattr(sample, band)
        fname = predict_file_name(config.ELM_PREDICT_PATH,
                                  tag,
                                  getattr(band_arr, 'canvas', getattr(sample, 'canvas')).bounds)
        predict_to_netcdf(prediction, fname)
        predict_to_pickle(prediction, fname)
    return True

def predict_step(config, step, executor,
                 models=None,
                 serialize=None,
                 to_cube=True,
                 transform_model=None):

    if hasattr(executor, 'map'):
        map_function = executor.map
    else:
        map_function = map
    if hasattr(executor, 'submit'):
        submit_func = executor.submit
    else:
        submit_func = no_executor_submit
    get_results = partial(wait_for_futures, executor=executor)
    predict_dict = config.predict[step['predict']]
    action_data = get_sample_pipeline_action_data(predict_dict, config, step)
    sampler_kwargs = action_data[0][-1]
    tag = step['predict']
    if serialize is None:
        serialize = partial(_default_serialize, tag, config)
    if models is None:
        logger.info('Load pickled models from {} {}'.format(config.ELM_TRAIN_PATH, tag))
        models, meta = load_models_from_tag(config.ELM_TRAIN_PATH,
                                            tag)
    args = sampler_kwargs['generated_args']
    arg_gen = tuple(itertools.product(args, models))

    predict = partial(_predict_one_sample_one_arg,
                      action_data,
                      transform_model,
                      serialize,
                      to_cube)
    return get_results(map_function(predict, arg_gen))
