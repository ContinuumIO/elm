from collections import Sequence
from functools import partial

import numpy as np
import pandas as pd

from elm.config import import_callable
from elm.model_selection.evolve import (ea_general,
                                        evo_init_func,
                                        assign_check_fitness)
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config.dask_settings import wait_for_futures
from elm.pipeline.transform import get_new_or_saved_transform_model
from elm.pipeline.util import (_validate_ensemble_members,
                               _prepare_fit_kwargs,
                               _get_model_selection_func,
                               _run_model_selection_func,
                               make_model_args_from_config,
                               run_train_dask)
from elm.pipeline.serialize import serialize_models


def on_each_generation(individual_to_new_config,
                       config,
                       sample_pipeline,
                       data_source,
                       transform_model,
                       samples_per_batch,
                       step,
                       train_or_transform,
                       ensemble_kwargs,
                       sample_pipeline_kwargs,
                       gen,
                       invalid_ind):
    '''Returns model, fit args, fit kwargs for Individual
    whose fitness must be solved

    Parameters:
        individual_to_new_config: Function taking Individual, returns
                                  elm.config.ConfigParser instance
        step:                     Dictionary step of pipeline
        train_or_transform:       "train" or "transform" (type of step
                                  in pipeline)
        config:                   from elm.config.ConfigParser
        ensemble_kwargs:           Dict with at least
                                   "partial_fit_batches": int key/value
        transform_model:          None or list of 1 model name, model, e.g.:
                                  [('tag_0', IncrementalPCA(....))]
                                  (None if sample_pipeline doesn't use
                                   a transform)
        ind:                      Individual to evaluate
    Returns:
        tuple of (args, kwargs) that go to elm.pipeline.run_model_method:run_model_method
    '''
    from elm.config.dask_settings import get_func
    new_models = []
    fit_kwargs = []
    for idx, ind in enumerate(invalid_ind):
        new_config = individual_to_new_config(ind)
        model_args, _ = make_model_args_from_config(train_or_transform,
                                                    sample_pipeline,
                                                    data_source,
                                                    config=new_config,
                                                    step=step,
                                                    ensemble_kwargs=ensemble_kwargs,
                                                    **sample_pipeline_kwargs)
        fit_kwargs.append(_prepare_fit_kwargs(model_args,
                                         ensemble_kwargs))
        model_init_kwargs = model_args.model_init_kwargs or {}
        model = import_callable(model_args.model_init_class)(**model_init_kwargs)
        new_models.append((ind.name, model))
    models = run_train_dask(config, sample_pipeline, data_source,
                   transform_model, samples_per_batch, new_models,
                   gen, fit_kwargs, sample_pipeline_kwargs=sample_pipeline_kwargs,
                   get_func=get_func)
    fitnesses = [model._score for name, model in models]
    fitnesses = [(item if isinstance(item, Sequence) else [item])
                 for item in fitnesses]
    return models, fitnesses


def _evolve_train_or_transform(train_or_transform,
                               client,
                               step,
                               evo_params,
                               config,
                               sample_pipeline,
                               data_source,
                               transform_model,
                               samples_per_batch,
                               sample_pipeline_kwargs,
                               **ensemble_kwargs):


    get_results = partial(wait_for_futures, client=client)
    control = evo_params.deap_params['control']
    required_args, _, _ = get_args_kwargs_defaults(ea_general)
    evo_args = [evo_params,]
    sample_pipeline_kwargs = sample_pipeline_kwargs or {}
    fit_one_generation = partial(on_each_generation,
                                 evo_params.individual_to_new_config,
                                 config, sample_pipeline, data_source,
                                 transform_model, samples_per_batch,
                                 step,
                                 train_or_transform,
                                 ensemble_kwargs,
                                 sample_pipeline_kwargs)

    try:
        param_history = []
        for a in required_args[1:]:
            if a not in control:
                raise ValueError('Expected {} in {} (control kwargs '
                                 'to evolutionary '
                                 'algorithm)'.format(a, control))
            evo_args.append(control[a])
        ea_gen = ea_general(*evo_args)
        pop, _, _ = next(ea_gen)
        models, fitnesses = fit_one_generation(0, pop)
        assign_check_fitness(pop,
                         fitnesses,
                         param_history,
                         evo_params.deap_params['choices'],
                         evo_params.score_weights)
        invalid_ind = True
        fitted_models = dict(models)
        ngen = evo_params.deap_params['control'].get('ngen') or None
        if not ngen and not evo_params.early_stop:
            raise ValueError('param_grids: pg_name: control: has neither '
                             'ngen or early_stop keys')
        elif not ngen:
            ngen = 1000000
        for gen in range(ngen):
            # on last generation invalid_ind becomes None
            # and breaks this loop
            if gen > 0:
                models, fitnesses = fit_one_generation(gen, invalid_ind)
                fitted_models.update(dict(models))
            (pop, invalid_ind, param_history) = ea_gen.send(fitnesses)

            pop_names = [ind.name for ind in pop]
            fitted_models = {k: v for k, v in fitted_models.items()
                             if k in pop_names}
            if not invalid_ind:
                break # If there are no new solutions to try, break
        pop = evo_params.toolbox.select(pop, ensemble_kwargs['saved_ensemble_size'])
        pop_names = [ind.name for ind in pop]
        models = [(k, v) for k, v in fitted_models.items()
                  if k in pop_names]
        serialize_models(models, **ensemble_kwargs)
    finally:
        columns = ['_'.join(p) for p in evo_params.deap_params['param_order']]
        columns += ['objective_{}_{}'.format(idx, 'min' if sw == -1 else 'max')
                    for idx, sw in enumerate(evo_params.score_weights)]
        if param_history:
            assert len(columns) == len(param_history[0])
            param_history = pd.DataFrame(np.array(param_history),
                                         columns=columns)
            param_history.to_csv(evo_params.history_file,
                                 index_label='parameter_set')
    return models


evolve_train = partial(_evolve_train_or_transform, 'train')
evolve_transform = partial(_evolve_train_or_transform, 'transform')