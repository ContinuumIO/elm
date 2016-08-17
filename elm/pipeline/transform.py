import copy
import logging
from functools import partial

import numpy as np
import xarray as xr

from elm.config import import_callable
from elm.pipeline.serialize import load_models_from_tag
from elm.sample_util.util import bands_as_columns

logger = logging.getLogger(__name__)


def transform_pipeline_step(*args, **kwargs):
    from elm.pipeline.train import _train_or_transform_step
    logger.debug('transform_pipeline_step')
    return _train_or_transform_step('transform', *args, **kwargs)


@bands_as_columns
def transform_sample_pipeline_step(sample_x,
                                   action,
                                   config,
                                   transform_models,
                                   **kwargs):

    assert len(transform_models) == 1
    name, transform_model = transform_models[0]
    t = action['transform']
    method = action.get('method', 'fit_transform')
    logger.debug('Transform with method {} and model {}'.format(method, repr(transform_model)))
    transform = config.transform[t]
    logger.debug('transform config {}'.format(transform))
    output =  getattr(transform_model, method)(sample_x.sample.values)
    dims = (sample_x.sample.dims[0], 'components')
    components = np.array(['c_{}'.format(idx) for idx in range(output.shape[1])])
    attrs = copy.deepcopy(sample_x.attrs)
    attrs['transform'] = {'new_shape': list(output.shape)}
    assert not np.any(np.isnan(output))
    assert np.all(np.isfinite(output))
    # TODO if a 'fit' or 'fit_transform' is called in sample_pipeline
    # that transform model needs to be serialized later using the relevant
    # "transform" tag
    sample_x['sample'] = xr.DataArray(output,
                                      coords=[(dims[0],
                                               getattr(sample_x.sample, dims[0]).values),
                                              ('components', components)],
                                      dims=dims,
                                      attrs=attrs)

    return sample_x

def _get_transform_models(action, config, **kwargs):
    method = action.get('method', 'fit_transform')
    tag = action['transform']
    if not 'fit' in method:
        logger.debug('Transform method does not include "fit"')
        logger.info('Load pickled transform_models from {} {}'.format(config.ELM_TRANSFORM_PATH, tag))
        transform_models, meta = load_models_from_tag(config.ELM_TRANSFORM_PATH, tag)
        assert len(transform_models) == 1
        name, transform_model = transform_models[0]
    else:
        model_init_class = import_callable(kwargs['model_init_class'])
        transform_model = model_init_class(**kwargs['model_init_kwargs'])
        name = 'tag_0'
    return [(name, transform_model)]


def init_sample_pipeline_transform_models(config, step):

    sample_pipeline = step.get('sample_pipeline') or []
    transform_dict = {}
    for action in sample_pipeline:
        if 'transform' in action:
            transform = copy.deepcopy(config.transform[action['transform']])
            transform_models = _get_transform_models(action, config, **transform)
            transform_dict[action['transform']] = transform_models
    logger.debug('Initialized {} transform_models'.format(len(transform_dict)))
    return transform_dict

