import copy
from functools import partial, wraps
import logging
import numbers

from elm.config import import_callable
from elm.model_selection.evolve import ea_setup
from elm.model_selection.util import get_args_kwargs_defaults, ModelArgs
from elm.pipeline.ensemble import ensemble
from elm.pipeline.evolve_train import evolve_train, evolve_transform,_make_config
from elm.pipeline.serialize import load_models_from_tag
from elm.pipeline.util import make_model_args_from_config

logger = logging.getLogger(__name__)


def _train_or_transform_step(train_or_transform,
                             sample_pipeline,
                             data_source,
                             config=None,
                             step=None,
                             train_dict=None,
                             transform_model=None,
                             transform_dict=None,
                             client=None,
                             model_args=None,
                             ensemble_kwargs=None,
                             evo_params=None,
                             evo_dict=None,
                             samples_per_batch=1,
                             sample=None,
                             **sample_pipeline_kwargs):
    '''Evaluate a "train" step in a config's "pipeline"

    Params:
        train_or_transform: string - "train" or "transform"
        config:  config from elm.config.ConfigParser
        step:    current step dictionary in config's pipeline,
        client: None or a threaded/process/distributed Executor
        kwargs:
    Returns:
        models: the fitted models in the ensemble
    '''
    from elm.pipeline.transform import get_new_or_saved_transform_model
    if not train_dict and (not config or not step):
        raise ValueError("Expected 'train_dict' and 'transform_dict' or 'config' and 'step'")
    evo_params = evo_params or None
    if model_args and evo_params:
        raise ValueError('Do not pass "model_args" when using "evo_params"')
    if evo_dict:
        if evo_params:
            raise ValueError('Cannot give evo_params and evo_dict')
        config, step = _make_config( train_or_transform,
                                     sample_pipeline, data_source,
                                     samples_per_batch, train_dict,
                                     transform_dict, ensemble_kwargs,
                                     evo_dict, **sample_pipeline_kwargs)
        evo_params_dict = ea_setup(config)
        evo_params = tuple(evo_params_dict.values())[0]

    transform_model = transform_model or None
    ensemble_kwargs = ensemble_kwargs or {}
    sample_pipeline_kwargs['transform_dict'] = transform_dict
    sample_pipeline_kwargs['transform_model'] = transform_model
    if not ensemble_kwargs and config and step:
        t = getattr(config, train_or_transform)[step[train_or_transform]]
        e = t.get('ensemble')
        if isinstance(e, dict):
            ensemble_kwargs = e
        else:
            ensemble_kwargs = config.ensembles[e]
    model_args2, ensemble_kwargs2 = make_model_args_from_config(train_or_transform,
                                                                sample_pipeline,
                                                                data_source,
                                                                config=config,
                                                                step=step,
                                                                train_dict=train_dict,
                                                                ensemble_kwargs=ensemble_kwargs,
                                                                **sample_pipeline_kwargs)
    if not model_args:
        model_args = model_args2
    ensemble_kwargs.update(ensemble_kwargs2)
    if transform_model is None and config and step:
        transform_model = get_new_or_saved_transform_model(config,
                                                           sample_pipeline,
                                                           data_source,
                                                           step)

    if not ensemble_kwargs.get('tag'):
        ensemble_kwargs['tag'] = model_args.step_name

    if evo_params or evo_dict:
        args = (client,
                step,
                evo_params,
                evo_dict,
                config,
                sample_pipeline,
                data_source,
                transform_model,
                samples_per_batch,
                train_dict,
                transform_dict,
                sample_pipeline_kwargs,
                sample)
        if train_or_transform == 'train':
            return evolve_train(*args, **ensemble_kwargs)
        return evolve_transform(*args, **ensemble_kwargs)

    models = ensemble(client,
                      model_args,
                      transform_model,
                      sample_pipeline,
                      data_source,
                      samples_per_batch=samples_per_batch,
                      config=config,
                      sample_pipeline_kwargs=sample_pipeline_kwargs,
                      sample=sample,
                      **ensemble_kwargs)
    return models

@wraps(_train_or_transform_step)
def train_step(*args, **kwargs):
    logger.debug('args/kwargs:')
    logger.debug(repr((args, kwargs)))

    return _train_or_transform_step('train', *args, **kwargs)

