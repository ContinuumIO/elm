from functools import partial

from elm.model_selection.evolve import evo_general
from elm.pipeline.ensemble import (_validate_ensemble_members,
                                   _prepare_fit_kwargs,
                                   _get_model_selection_func,
                                   _run_model_selection_func,
                                   _serialize_models,
                                   _fit_list_of_models)
from elm.pipeline.executor_util import wait_for_futures


def _fit_invalid_ind(individual_to_new_config,
                     invalid_ind,
                     transform_dict,
                     map_function,
                     get_results):
    args_kwargs = tuple(on_each_evo_yield(individual_to_new_config,
                                          step,
                                          train_or_transform,
                                          invalid_ind))
    models = _fit_list_of_models(args_kwargs, map_function, get_results)
    fitnesses = [model._score for model in models]
    return models, fitnesses


def serialize_deap_outputs(final_pop, record, logbook):
    # TODO dump these to pickle
    return True


def on_each_evo_yield(individual_to_new_config,
                      step,
                      train_or_transform,
                      invalid_ind):
    for ind in invalid_ind:
        new_config = individual_to_new_config(ind)
        ma = make_model_args_from_config(new_config,
                                         step,
                                         train_or_transform)
        train_model_args, transform_model_args, _ = ma
        fit_kwargs = _prepare_fit_kwargs(model_args, transform_dict)
        model = import_callable(model_args.model_init_class)(**model_args.model_init_kwargs)
        yield (model,
               train_model_args,
               transform_model_args,
               fit_kwargs)


def evolutionary_algorithm(executor,
                           train_or_transform,
                           config,
                           evo_params,
                           **ensemble_kwargs):
    from elm.pipeline.train import make_model_args_from_config
    if hasattr(executor, 'map'):
        map_function = executor.map
    else:
        map_function = map
    get_results = partial(wait_for_futures, executor=executor)
    control = evo_params.control
    pop = evo_init_func(evo_params)
    required_args, _, _ = get_args_kwargs_defaults(evo_general)
    evo_args = [toolbox, pop]
    for a in required_args[2:]:
        if a not in control:
            raise ValueError('Expected {} in {} (control kwargs '
                             'to evolutionary '
                             'algorithm)'.format(a, control))
        evo_args.append(control[a])
    ea_gen = evo_general(*evo_args)
    next(ea_gen) # dummy
    invalid_ind = pop
    while invalid_ind:
        models, fitnesses = _fit_invalid_ind(evo_params.individual_to_new_config,
                                            step_to_model_args,
                                            pop,
                                            invalid_ind,
                                            transform_dict,
                                            map_function,
                                            get_results)
        (pop, invalid_ind, record, logbook) = ea_gen.send(fitnesses)
    _serialize_models(models, **ensemble_kwargs)
    serialize_deap_outputs(pop, record, logbook)
    return models

