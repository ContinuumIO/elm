import copy
import logging
from functools import partial

import numpy as np
import xarray as xr

from elm.config import import_callable
from elm.pipeline.serialize import load_models_from_tag
from elm.readers import ElmStore, check_is_flat
from elm.pipeline.util import _make_model_args_from_config

logger = logging.getLogger(__name__)

__all__ = ['transform_sample_pipeline_step',]



def transform_sample_pipeline_step(sample_x,
                                   action,
                                   transform_models,
                                   **kwargs):

    name, transform_model = transform_models[0]
    t = action['transform']
    method = action.get('method', 'fit_transform')
    logger.debug('Transform with method {} and model {}'.format(method, repr(transform_model)))
    output =  getattr(transform_model, method)(sample_x.flat.values)
    dims = ('space', 'band')
    components = np.array(['c_{}'.format(idx) for idx in range(output.shape[1])])
    attrs = copy.deepcopy(sample_x.attrs)
    attrs['transform'] = {'new_shape': list(output.shape)}
    attrs['band_order'] = components
    # TODO if a 'fit' or 'fit_transform' is called in sample_pipeline
    # that transform model needs to be serialized later using the relevant
    # "transform" tag
    return ElmStore({'flat': xr.DataArray(output,
                                    coords=[(dims[0],
                                             getattr(sample_x.flat, dims[0]).values),
                                              ('band', components)],
                                    dims=dims,
                                    attrs=attrs)}, attrs=attrs)



def _get_saved_transform_models(action, config, **kwargs):
    method = action.get('method', 'fit_transform')
    tag = action['transform']
    logger.debug('Transform method does not include "fit"')
    logger.info('Load pickled transform_models from {} {}'.format(config.ELM_TRANSFORM_PATH, tag))
    transform_models, meta = load_models_from_tag(config.ELM_TRANSFORM_PATH, tag)
    return transform_models


def init_saved_transform_models(config, sample_pipeline):

    transform_model = None
    for action in sample_pipeline:
        if 'transform' in action:
            transform = copy.deepcopy(config.transform[action['transform']])
            transform_model = _get_saved_transform_models(action,
                                                          config,
                                                          **transform)
    logger.debug('Initialized transform model {}'.format(transform_model))
    return transform_model


def get_new_or_saved_transform_model(config, sample_pipeline, data_source, step):
    transform_model = None
    train_or_transform = 'train' if 'train' in step else 'transform'
    for item in sample_pipeline:
        if 'transform' in item:
            method = item.get('method', config.transform.get('method', None))
            if method is None:
                raise ValueError('Expected a "method" for transform')
            if 'fit' not in method:
                return init_saved_transform_models(config, sample_pipeline)
            else:
                model_args = _make_model_args_from_config(config,
                                                          config.transform[item['transform']],
                                                          step,
                                                          train_or_transform,
                                                          sample_pipeline,
                                                          data_source)
                model = model_args.model_init_class(**model_args.model_init_kwargs)
                return [('tag_0', model)]
    return None
