# DEPRECATED (temporarily): See also - https://github.com/ContinuumIO/elm/issues/149

from __future__ import absolute_import, division, print_function

'''
----------------------

``elm.pipeline.parse_run_config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
from collections import defaultdict
import copy
from functools import partial
import logging
import os

import dask
try:
    from earthio import load_meta, load_array
except:
    load_array = load_array = None # TODO handle cases where load_* = None

from earthio.config import ConfigParser, import_callable
#from elm.model_selection.evolve import ea_setup
from elm.pipeline.ensemble import ensemble
from elm.pipeline.pipeline import Pipeline
from elm.pipeline.serialize import (serialize_prediction,
                                    serialize_pipe,
                                    load_pipe_from_tag)

from six import string_types

logger = logging.getLogger(__name__)

def _makedirs(config):
    for d in (config.ELM_TRAIN_PATH, config.ELM_PREDICT_PATH):
        if d and not os.path.exists(d):
            os.makedirs(d)


def config_to_pipeline(config, client=None):
    '''
    Run the elm config's train and predict "run"
    actions with dask client and config's updates
    based on args passed to elm-main, such as --train-only
    or --predict-only, or edits to ensemble settings, such as
    --ngen 4

    Parameters:
        :config: elm.config.ConfigParser instance
        :client: dask client or None
    '''

    _makedirs(config)
    idx_to_evo_params = ea_setup(config)
    for idx, step in enumerate(config.run):
        pipeline = step['pipeline']
        if 'train' in step:
            train = config.train[step['train']]
            pipe_steps = make_pipeline_steps(config, pipeline)
            cls = import_callable(train['model_init_class'])
            estimator = cls(**(train.get('model_init_kwargs') or {}))
            pipe_steps.append((step['train'], estimator))
            ensemble_kwargs = train.get('ensemble')
            if isinstance(ensemble_kwargs, string_types):
                ensemble_kwargs = config.ensembles[ensemble_kwargs]
            ensemble_kwargs['client'] = client
        data_source = step['data_source']
        if not isinstance(data_source, dict):
            data_source = config.data_sources[data_source]
        data_source['sampler'] = import_callable(data_source['sampler'])
        data_source['load_meta'] = load_meta
        data_source['load_array'] = load_array
        if callable(data_source.get('args_list')):
            kw = {k: v for k, v in data_source.items() if k != 'args_list'}
            data_source['args_list'] = tuple(data_source['args_list'](**kw))
        if 'train' in step and not getattr(config, 'PREDICT_ONLY', False):
            s = train.get('model_scoring')
            if s:
                scoring = config.model_scoring[s]
                scoring_kwargs = {k: v for k, v in scoring.items()
                                  if k != 'scoring'}
                scoring = import_callable(scoring['scoring'])
            else:
                scoring = None
                scoring_kwargs = {}
            if 'method_kwargs' in train:
                method_kwargs = train['method_kwargs']
            else:
                method_kwargs = {}
            if 'classes' in train:
                method_kwargs['classes'] = train['classes']
            ensemble_kwargs['method_kwargs'] = method_kwargs
            pipe = Pipeline(pipe_steps, scoring=scoring, scoring_kwargs=scoring_kwargs)
            evo_params = idx_to_evo_params.get(idx, None)
            if evo_params:
                kw = dict(evo_params=evo_params)
                kw.update(data_source)
                kw.update(ensemble_kwargs)
                pipe.fit_ea(**kw)
            else:
                kw = {}
                kw.update(data_source)
                kw.update(ensemble_kwargs)
                pipe.fit_ensemble(**kw)

            serialize_pipe(pipe, config.ELM_TRAIN_PATH, step['train'])
        elif 'predict' in step and not getattr(config, 'TRAIN_ONLY', False):
            pipe = load_pipe_from_tag(config.ELM_TRAIN_PATH, step['predict'])

        else:
            logger.info('Do nothing for {} (has no "train" or "predict" key)'.format(step))
        if 'predict' in step:
            # serialize is called with (prediction, sample, tag)
            serialize = partial(serialize_prediction, config)
            pipe.predict_many(serialize=serialize, **data_source)


def parse_run_config(config, client):
    '''Run all steps of a config's "run" section
    Parameters:
        :config: elm.config.ConfigParser instance
        :client: Executor/client from Distributed, thread pool
                or None for serial evaluation
    '''
    config_to_pipeline(config, client)
    return 0
