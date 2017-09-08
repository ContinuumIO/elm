from __future__ import absolute_import, division, print_function, unicode_literals

from collections import Sequence
from functools import partial
from pprint import pformat
import logging

import numpy as np
import pandas as pd

from elm.config import import_callable, ConfigParser
from elm.config.dask_settings import _find_get_func_for_client
from elm.model_selection.evolve import (ea_general,
                                        evo_init_func,
                                        assign_check_fitness,
                                        ind_to_new_pipe)
from elm.config.func_signatures import get_args_kwargs_defaults
from elm.pipeline.util import _validate_ensemble_members
from elm.pipeline.ensemble import _one_generation_dask_graph, ensemble
from elm.pipeline.serialize import serialize_pipe
from earthio.filters.samplers import make_samples_dask

__all__ = ['evolve_train']

logger = logging.getLogger(__name__)

def _on_each_generation(base_model,
                       data_source,
                       deap_params,
                       get_func,
                       partial_fit_batches,
                       method,
                       method_kwargs,
                       dsk,
                       gen,
                       sample_keys,
                       invalid_ind):

    new_models = []
    fit_kwargs = []
    for idx, ind in enumerate(invalid_ind):
        model = ind_to_new_pipe(base_model, deap_params, ind)
        new_models.append((ind.name, model))
    dsk, model_keys, new_models_name = _one_generation_dask_graph(dsk,
                                        new_models,
                                        method_kwargs,
                                        sample_keys,
                                        partial_fit_batches,
                                        gen,
                                        method)
    if get_func is None:
        new_models = tuple(dask.get(dsk, new_models_name))
    else:
        new_models = tuple(get_func(dsk, new_models_name))
    models = tuple(zip(model_keys, new_models))
    logger.info('Trained {} estimators'.format(len(models)))

    fitnesses = [model._score for name, model in models]
    fitnesses = [(item if isinstance(item, Sequence) else [item])
                 for item in fitnesses]
    return models, fitnesses


def evolve_train(pipe,
                 evo_params,
                 X=None,
                 y=None,
                 sample_weight=None,
                 sampler=None,
                 args_list=None,
                 client=None,
                 init_ensemble_size=1,
                 saved_ensemble_size=None,
                 ensemble_init_func=None,
                 scoring_kwargs=None,
                 method='fit',
                 partial_fit_batches=1,
                 classes=None,
                 method_kwargs=None,
                 **data_source):
    '''evolve_train runs an evolutionary algorithm to
    find the most fit elm.pipeline.Pipeline instances

    Parameters:
        pipe: elm.pipeline.Pipeline instance
        evo_params: the EvoParams instance, typically from
            from elm.model_selection import ea_setup
            evo_params = ea_setup(param_grid=param_grid,
                          param_grid_name='param_grid_example',
                          score_weights=[-1]) # minimization

        See also the help from (elm.pipeline.ensemble) where
        most arguments are interpretted similary.

    ''' + ensemble.__doc__
    models_share_sample = True
    method_kwargs = method_kwargs or {}
    scoring_kwargs = scoring_kwargs or {}
    get_func = _find_get_func_for_client(client)
    control = evo_params.deap_params['control']
    required_args, _, _ = get_args_kwargs_defaults(ea_general)
    evo_args = [evo_params,]
    data_source = dict(X=X,y=y, sample_weight=sample_weight, sampler=sampler,
                       args_list=args_list, **data_source)
    fit_one_generation = partial(_on_each_generation,
                                 pipe,
                                 data_source,
                                 evo_params.deap_params,
                                 get_func,
                                 partial_fit_batches,
                                 method,
                                 method_kwargs)


    dsk = make_samples_dask(X, y, sample_weight, pipe, args_list, sampler, data_source)
    sample_keys = list(dsk)
    if models_share_sample:
        np.random.shuffle(sample_keys)
        gen_to_sample_key = lambda gen: [sample_keys[gen]]
    else:
        gen_to_sample_key = lambda gen: sample_keys
    sample_keys = tuple(sample_keys)

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
        sample_keys_passed = gen_to_sample_key(0)
        def log_once(len_models, sample_keys_passed, gen):
            total_calls = len_models * len(sample_keys_passed) * partial_fit_batches
            msg = (len_models, len(sample_keys_passed), partial_fit_batches, method, gen, total_calls)
            fmt = 'Evolve generation {4}: {0} models x {1} samples x {2} {3} calls = {5} calls in total'
            logger.info(fmt.format(*msg))
        log_once(len(pop), sample_keys_passed, 0)
        pop_names = [ind.name for ind in pop]
        models, fitnesses = fit_one_generation(dsk, 0, sample_keys_passed, pop)
        assign_check_fitness(pop,
                         fitnesses,
                         param_history,
                         evo_params.deap_params['choices'],
                         evo_params.score_weights)
        invalid_ind = True
        fitted_models = {n: m for n, (_, m) in zip(pop_names, models)}
        ngen = evo_params.deap_params['control'].get('ngen') or None
        if not ngen and not evo_params.early_stop:
            raise ValueError('param_grids: pg_name: control: has neither '
                             'ngen or early_stop keys')
        elif not ngen:
            ngen = 1000000
        for gen in range(ngen):
            # on last generation invalid_ind becomes None
            # and breaks this loop
            if models_share_sample:
                sample_keys_passed = (gen_to_sample_key(gen % len(sample_keys)),)
            else:
                sample_keys_passed = sample_keys

            if gen > 0:
                log_once(len(invalid_ind), sample_keys_passed, gen)
                names = [ind.name for ind in invalid_ind]
                models, fitnesses = fit_one_generation(dsk, gen, sample_keys_passed, invalid_ind)
                fitted_models.update({n: m for n, (_, m) in zip(names,models)})
            (pop, invalid_ind, param_history) = ea_gen.send(fitnesses)
            pop_names = [ind.name for ind in pop]
            fitted_models = {k: v for k, v in fitted_models.items()
                             if k in pop_names}
            if not invalid_ind:
                break # If there are no new solutions to try, break
        pop = evo_params.toolbox.select(pop, saved_ensemble_size)
        pop_names = [ind.name for ind in pop]
        models = [(k, v) for k, v in fitted_models.items()
                  if k in pop_names]

    finally:
        columns = list(evo_params.deap_params['param_order'])
        columns += ['objective_{}_{}'.format(idx, 'min' if sw == -1 else 'max')
                    for idx, sw in enumerate(evo_params.score_weights)]
        if param_history:
            assert len(columns) == len(param_history[0])
            param_history = pd.DataFrame(np.array(param_history),
                                         columns=columns)
            param_history.to_csv(evo_params.history_file,
                                 index_label='parameter_set')
    return models

