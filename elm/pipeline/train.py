import copy
from functools import partial
import logging
import numbers

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults, ModelArgs
from elm.pipeline.ensemble import ensemble
from elm.pipeline.evolutionary_algorithm import evolutionary_algorithm
from elm.pipeline.serialize import load_models_from_tag
from elm.pipeline.util import make_model_args_from_config

logger = logging.getLogger(__name__)


def _train_or_transform_step(train_or_transform,
                             config,
                             step,
                             executor,
                             **kwargs):
    '''Evaluate a "train" step in a config's "pipeline"

    Params:
        train_or_transform: string - "train" or "transform"
        config:  config from elm.config.ConfigParser
        step:    current step dictionary in config's pipeline,
        executor: None or a threaded/process/distributed Executor
        kwargs:
    Returns:
        models: the fitted models in the ensemble
    '''
    from elm.pipeline.transform import get_new_or_saved_transform_model
    evo_params = kwargs.get('evo_params') or None
    if evo_params is not None:
        return evolutionary_algorithm(executor,
                                      config,
                                      step,
                                      evo_params,
                                      transform_dict,
                                      **ensemble_kwargs)
    model_args, ensemble_kwargs = make_model_args_from_config(config,
                                                              step,
                                                              train_or_transform)
    transform_model = kwargs.get('transform_model') or None
    if transform_model is None:
        transform_model = get_new_or_saved_transform_model(config, step)
    models = ensemble(executor,
                      model_args,
                      transform_model,
                      **ensemble_kwargs)
    return models


train_step = partial(_train_or_transform_step, 'train')

