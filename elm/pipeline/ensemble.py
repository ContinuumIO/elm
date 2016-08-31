import logging

from functools import partial

from elm.config import import_callable
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.util import (_fit_list_of_models,
                               _prepare_fit_kwargs,
                               serialize_models,
                               _validate_ensemble_members,
                               _get_model_selection_func,
                               _run_model_selection_func)

logger = logging.getLogger(__name__)


def ensemble(executor,
             model_args,
             transform_model,
             **ensemble_kwargs):
    '''Train model(s) in ensemble

    Params:
        executor: None or a thread/process/distributed Executor
        model_args: ModelArgs namedtuple
        transform_model: dictionary like:
                        {transform_name: [("tag_0", transform_model)]}

                        for transform models like PCA that have already
                        been "fit".  If using "fit" or "fit_transform" in
                        "sample_pipeline" then this transform_model has no
                        effect.
        ensemble_kwargs: kwargs such as "ensemble_size" and "ngen"
                    which control the ensemble size and number of
                    generations in the ensemble (calls to model_selection_func)
        '''
    if hasattr(executor, 'map'):
        map_function = executor.map
    else:
        map_function = map
    get_results = partial(wait_for_futures, executor=executor)
    model_selection_kwargs = model_args.model_selection_kwargs or {}
    ensemble_size = ensemble_kwargs['init_ensemble_size']
    ngen = ensemble_kwargs['ngen']
    model_names = ensemble_kwargs.get('model_names', None)
    ensemble_init_func = ensemble_kwargs.get('ensemble_init_func') or None
    model_init_kwargs = model_args.model_init_kwargs
    model_init_class = model_args.model_init_class
    if not ensemble_init_func:
        models = tuple(('tag_{}'.format(idx), model_init_class(**model_init_kwargs))
                        for idx in range(ensemble_kwargs['init_ensemble_size']))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(model_init_class,
                                    model_init_kwargs,
                                    **ensemble_kwargs)
    models = _validate_ensemble_members(models)
    model_selection_func = _get_model_selection_func(model_args)
    fit_kwargs = _prepare_fit_kwargs(model_args, transform_model, ensemble_kwargs)
    model_names = [name for name, model in models]
    for generation in range(ngen):
        logger.info('Ensemble generation {} of {} ({} models)'.format(generation + 1, ngen, len(models)))
        args_kwargs = tuple(((model,) + tuple(model_args.fit_args), fit_kwargs)
                            for name, model in models)
        logger.debug('fit args_kwargs {}'.format(args_kwargs))
        models = _fit_list_of_models(args_kwargs,
                                     map_function,
                                     get_results,
                                     model_names)
        if model_selection_func:
            models = _run_model_selection_func(model_selection_func,
                                               model_args,
                                               ngen,
                                               generation,
                                               fit_kwargs,
                                               models)
        else:
            pass # just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
    serialize_models(models, **ensemble_kwargs)
    return models


