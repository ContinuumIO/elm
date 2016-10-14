from collections import defaultdict
import copy
import logging

import dask

from elm.config import ConfigParser, import_callable
from elm.pipeline.pipeline import Pipeline
from elm.sample_util.sample_pipeline import make_pipeline_func
from elm.model_selection.evolve import ea_setup
from elm.pipeline.ensemble import ensemble

logger = logging.getLogger(__name__)

def config_to_pipeline(config, client):


    for step in config.run:
        pipeline = step['pipeline']
        train = config.train[step['train']]
        pipe_steps = make_pipeline_func(config, pipeline)
        cls = import_callable(train['model_init_class'])
        estimator = cls(**(train.get('model_init_kwargs') or {}))
        pipe_steps.append(estimator)
        ensemble_kwargs = train.get('ensemble')
        if isinstance(ensemble_kwargs, str):
            ensemble_kwargs = config.ensembles[ensemble_kwargs]
        ensemble_kwargs['client'] = client
        data_source = step['data_source']
        if not isinstance(data_source, dict):
            data_source = config.data_sources[data_source]
        if callable(data_source['args_list']):
            kw = {k: v for k, v in data_source.items() if k != 'args_list'}
            data_source['args_list'] = tuple(data_source['args_list'](**kw))

        s = train.get('model_scoring')
        if s:
            scoring = config.model_scoring[s]
            scoring_kwargs = {k: v for k, v in scoring.items()
                              if k != 'scoring'}
            scoring = import_callable(scoring['scoring'])
        else:
            scoring = None
            scoring_kwargs = {}
        pipe = Pipeline(pipe_steps, scoring=scoring, scoring_kwargs=scoring_kwargs)
        models = pipe.fit_ensemble(**data_source, **ensemble_kwargs)
        pipe.predict_many(**data_source)



def parse_run_config(config, client):
    '''Run all steps of a config's "pipeline"
    Parameters:
        config: elm.config.ConfigParser instance
        client: Executor/client from Distributed, thread pool
                or None for serial evaluation
    '''
    config_to_pipeline(config, client)
    return 0

