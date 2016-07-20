from functools import partial
import copy
import datetime
import os

import numpy as np
import xarray as xr

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults
from elm.pipeline.executor_util import (wait_for_futures,
                                        no_executor_submit)
from elm.pipeline.sample_pipeline import (all_sample_ops,
                                          flatten_cube,
                                          flattened_to_cube)
from elm.pipeline.serialize import (predict_to_netcdf,
                                    predict_to_pickle,
                                    load_models_from_tag)
from elm.pipeline.sample_pipeline import run_sample_pipeline
from elm.preproc.elm_store import ElmStore

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
    prediction1 = model.predict(sample_flat.sample.values).astype('i4')[:, np.newaxis]
    attrs = copy.deepcopy(sample.attrs)
    attrs['elm_predict_date'] = datetime.datetime.utcnow().isoformat()
    prediction = ElmStore({'sample': xr.DataArray(prediction1,
                          coords=[('space',sample_flat.space.values),
                                 ('band', np.array(['class']))],
                          attrs=attrs)},
                        attrs=attrs)
    prediction = flattened_to_cube(prediction, **attrs)
    if return_serialized:
        return serialize(prediction, sample)
    return prediction

def predict_step(config, step, executor,
                 models=None,
                 serialize=None):

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
    if serialize is None:
        def serialize(prediction, sample):
            fname = predict_file_name(config.ELM_PREDICT_PATH,
                                      tag,
                                      sample['sample'].Bounds)
            predict_to_netcdf(prediction, fname)
            predict_to_pickle(prediction, fname)
            return True
    if models is None:
        models, meta = load_models_from_tag(config.ELM_PICKLE_PATH,
                                            tag)
    predict = partial(_predict_one_sample, action_data, serialize)
    return get_results(map_function(predict, models))
