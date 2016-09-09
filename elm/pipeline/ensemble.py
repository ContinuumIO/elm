import copy
from functools import partial
from itertools import chain
import logging

import dask

from elm.config import import_callable
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.serialize import serialize_models
from elm.pipeline.util import (_fit_list_of_models,
                               _prepare_fit_kwargs,
                               _validate_ensemble_members,
                               _get_model_selection_func,
                               _run_model_selection_func,
                               _fit_one_model)
from elm.sample_util.samplers import make_one_sample

logger = logging.getLogger(__name__)

_next_idx = 0

def _next_name():
    global _next_idx
    n = 'ensemble-{}'.format(_next_idx)
    _next_idx += 1
    return n


def ensemble(client,
             model_args,
             transform_model,
             sample_pipeline_info,
             **ensemble_kwargs):

    '''Train model(s) in ensemble

    Params:
        client: None or a thread/process/distributed Executor
        model_args: ModelArgs namedtuple
        transform_model: dictionary like:
                        {transform_name: [("tag_0", transform_model)]}

                        for transform models like PCA that have already
                        been "fit".  If using "fit" or "fit_transform" in
                        "sample_pipeline" then this transform_model has no
                        effect.
        ensemble_kwargs: kwargs such as "ensemble_size" and "ngen"
                    which control the ensemble size and number of
                    gens in the ensemble (calls to model_selection_func)
        '''
    from elm.config.dask_settings import get_func
    if hasattr(client, 'map'):
        map_function = client.map
    else:
        map_function = map
    (config, sample_pipeline, data_source, transform_model, samples_per_batch) = sample_pipeline_info
    n_batches = data_source.get('n_batches') or 1
    get_results = partial(wait_for_futures, client=client)
    model_selection_kwargs = model_args.model_selection_kwargs or {}
    ensemble_size = ensemble_kwargs['init_ensemble_size']
    ngen = ensemble_kwargs['ngen']
    ensemble_init_func = ensemble_kwargs.get('ensemble_init_func') or None
    model_init_kwargs = model_args.model_init_kwargs
    model_init_class = model_args.model_init_class
    if not ensemble_init_func:
        models = tuple(model_init_class(**model_init_kwargs)
                       for idx in range(ensemble_kwargs['init_ensemble_size']))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(model_init_class,
                                    model_init_kwargs,
                                    **ensemble_kwargs)
    model_selection_func = _get_model_selection_func(model_args)
    fit_kwargs = _prepare_fit_kwargs(model_args, ensemble_kwargs)
    final_names = []
    new_models = ()
    for gen in range(ngen):
        train_dsk = {}
        sample_name = 'sample-{}'.format(gen)
        sample_args = tuple(sample_pipeline_info) + (sample_name,)
        train_dsk.update(make_one_sample(*sample_args))
        keys = tuple('model-{}-gen-{}'.format(idx, gen)
                     for idx in range(len(new_models or models)))
        if gen == 0:
            models = zip(keys, models)
        for name, (_, model) in zip(keys, models):
            fitter = partial(_fit_one_model, model, **fit_kwargs)
            train_dsk[name] = (fitter, sample_name)
        def tkeys(*args):
            return tuple(args)
        new_models_name = _next_name()
        train_dsk[new_models_name] = (tkeys, *keys)
        if get_func is None:
            new_models = tuple(dask.get(train_dsk, new_models_name))
        else:
            new_models = tuple(get_func(train_dsk, new_models_name))
        models = tuple(zip(keys, new_models))
        if model_selection_func:
            models = _run_model_selection_func(model_selection_func,
                                               model_args,
                                               ngen,
                                               gen,
                                               fit_kwargs,
                                               models)
        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
        for m in models:
            assert isinstance(m, tuple) and len(m) == 2, repr(models)
    serialize_models(models, **ensemble_kwargs)
    return models


