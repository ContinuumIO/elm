import copy
from functools import partial
import logging
import numbers

from elm.config import import_callable
from elm.model_selection.util import get_args_kwargs_defaults, ModelArgs
from elm.pipeline.ensemble import ensemble
from elm.pipeline.evolve_train import evolve_train, evolve_transform
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
    assert train_or_transform in ('train', 'transform')
    from elm.pipeline.transform import get_new_or_saved_transform_model
    evo_params = kwargs.get('evo_params') or None
    model_args, ensemble_kwargs = make_model_args_from_config(config,
                                                              step,
                                                              train_or_transform)
    if evo_params is not None:
        args = (executor,
                step,
                evo_params,
                kwargs.get('transform_model') or None,)
        if train_or_transform == 'train':
            return evolve_train(*args, **ensemble_kwargs)
        return evolve_transform(*args, **ensemble_kwargs)
    transform_model = kwargs.get('transform_model') or None
    if transform_model is None:
        transform_model = get_new_or_saved_transform_model(config, step)
    models = ensemble(executor,
                      model_args,
                      transform_model,
                      **ensemble_kwargs)
    return models


train_step = partial(_train_or_transform_step, 'train')

