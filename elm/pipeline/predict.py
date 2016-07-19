import copy
from functools import partial
import os

import xarray as xr

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults
from elm.pipeline.executor_util import (wait_for_futures,
                                        no_executor_submit)
from elm.pipeline.sample_pipeline import (all_sample_ops,
                                          flatten_cube)
from elm.pipeline.serialize import (band_to_tif,
                                    load_models_from_tag)
from elm.pipeline.sample_pipeline import run_sample_pipeline

def predict_file_name(elm_predict_path, tag, bounds):
    return os.path.join(elm_predict_path,
                        tag,
                        '{}_{}_{}_{}'.format(bounds.left,
                                             bounds.bottom,
                                             bounds.right,
                                             bounds.top))

def _predict_one_sample(action_data, serialize, model, return_serialized=True):
    sample = run_sample_pipeline(action_data)
    sample_flat = flatten_cube(sample)
    prediction = model.predict(sample_flat)
    prediction.resize(sample.y.size, sample.x.size)
    attrs = {'predict': {'from': dict(sample.attrs)}}
    attrs.update(sample['sample'].attrs)
    print(attrs, sample['sample'])
    prediction = xr.DataArray(prediction,
                              coords=[
                                    ('y', sample.y),
                                    ('x', sample.x),
                                    ],
                              attrs=attrs)
    if return_serialized:
        return serialize(prediction, sample)
    return prediction

def predict_step(config, step, executor,
                 models=None, train_config=None):

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
    if predict_dict.get('sampler'):
        sampler = config.samplers[predict_dict['sampler']]
    else:
        data_source = config.data_sources[predict_dict['data_source']]
    action_data = all_sample_ops(predict_dict, config, step)
    tag = step['predict']
    def serialize(prediction, sample):
        fname = predict_file_name(config.ELM_PREDICT_PATH,
                                  tag,
                                  sample['sample'].Bounds)
        return band_to_tif(prediction, fname)
    if models is None:
        models, meta = load_models_from_tag(config.ELM_PICKLE_PATH,
                                            tag)
    predict = partial(_predict_one_sample, action_data, serialize)
    return get_results(map_function(predict, models))
