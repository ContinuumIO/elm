import copy
from functools import partial
from itertools import chain
import logging

from elm.config import import_callable
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.serialize import serialize_models
from elm.pipeline.util import (_prepare_fit_kwargs,
                               _validate_ensemble_members,
                               _get_model_selection_func,
                               _run_model_selection_func,
                               run_train_dask)
from elm.sample_util.samplers import make_one_sample

logger = logging.getLogger(__name__)


def ensemble(client,
             model_args,
             transform_model,
             sample_pipeline,
             data_source,
             samples_per_batch=1,
             config=None,
             sample_pipeline_kwargs=None,
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
    models = tuple(zip(('tag_{}'.format(idx) for idx in range(len(models))), models))
    for gen in range(ngen):

        models = run_train_dask(config, sample_pipeline, data_source,
                                transform_model,
                                samples_per_batch, models,
                                gen, fit_kwargs,
                                sample_pipeline_kwargs=sample_pipeline_kwargs,
                                get_func=get_func)
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
    serialize_models(models, **ensemble_kwargs)
    return models


