from functools import partial

from elm.config import import_callable
from elm.model_selection.evolve import (evo_general,
                                        evo_init_func,
                                        assign_fitnesses)
from elm.model_selection.util import get_args_kwargs_defaults
from elm.pipeline.executor_util import wait_for_futures
from elm.pipeline.transform import get_new_or_saved_transform_model
from elm.pipeline.util import (_validate_ensemble_members,
                               _prepare_fit_kwargs,
                               _get_model_selection_func,
                               _run_model_selection_func,
                               serialize_models,
                               _fit_list_of_models,
                               make_model_args_from_config)

LAST_TAG_IDX = 0
def next_model_tag():
    global LAST_TAG_IDX
    model_name =  'tag_{}'.format(LAST_TAG_IDX)
    LAST_TAG_IDX += 1
    return model_name


def _on_each_ind(individual_to_new_config, step, train_or_transform,
                 config, transform_model, ind):
    new_config = individual_to_new_config(ind)
    model_args, _ = make_model_args_from_config(new_config,
                                                step,
                                                train_or_transform)
    if transform_model is None:
        transform_model = get_new_or_saved_transform_model(config, step)

    fit_kwargs = _prepare_fit_kwargs(model_args, transform_model)
    model_init_kwargs = model_args.model_init_kwargs or {}
    model = import_callable(model_args.model_init_class)(**model_init_kwargs)
    return ((model,) + tuple(model_args.fit_args), fit_kwargs)


def on_each_evo_yield(individual_to_new_config,
                      step,
                      train_or_transform,
                      invalid_ind,
                      config,
                      transform_model=None):
    args_kwargs = [] # args are (model, other needed args)
                     # kwargs are fit_kwargs
    on_each_ind = partial(_on_each_ind,
                          individual_to_new_config,
                          step, train_or_transform,
                          config, transform_model)

    args_kwargs = map(on_each_ind, invalid_ind)
    return args_kwargs


def _fit_invalid_ind(individual_to_new_config,
                     transform_model,
                     step,
                     train_or_transform,
                     config,
                     map_function,
                     get_results,
                     invalid_ind):
    args_kwargs = tuple(on_each_evo_yield(individual_to_new_config,
                                          step,
                                          train_or_transform,
                                          invalid_ind,
                                          config,
                                          transform_model=transform_model))
    model_names = [next_model_tag() for _ in range(len(invalid_ind))]
    models = _fit_list_of_models(args_kwargs,
                                 map_function,
                                 get_results,
                                 model_names)
    fitnesses = [model._score for name, model in models]
    return models, fitnesses


def evolutionary_algorithm(executor,
                           step,
                           train_or_transform,
                           evo_params,
                           transform_model,
                           **ensemble_kwargs):

    if hasattr(executor, 'map'):
        map_function = executor.map
    else:
        map_function = map
    config = ensemble_kwargs['config']
    get_results = partial(wait_for_futures, executor=executor)
    control = evo_params.deap_params['control']
    pop = evo_init_func(evo_params)
    required_args, _, _ = get_args_kwargs_defaults(evo_general)
    history_file = open(evo_params.history_file, 'w')
    evo_args = [evo_params.toolbox, history_file, pop]
    fit_one_generation = partial(_fit_invalid_ind,
                                 evo_params.individual_to_new_config,
                                 transform_model,
                                 step,
                                 train_or_transform,
                                 config,
                                 map_function,
                                 get_results)
    try:
        for a in required_args[3:]:
            if a not in control:
                raise ValueError('Expected {} in {} (control kwargs '
                                 'to evolutionary '
                                 'algorithm)'.format(a, control))
            evo_args.append(control[a])
        ea_gen = evo_general(*evo_args)
        next(ea_gen) # dummy
        models, fitnesses = fit_one_generation(pop)
        assign_fitnesses(pop, fitnesses, history_file)
        invalid_ind = True
        for idx in range(evo_params.deap_params['control']['ngen']):
            # on last generation invalid_ind becomes None
            # and breaks this loop
            if idx > 0:
                models, fitnesses = fit_one_generation(invalid_ind)
            (pop, invalid_ind,) = ea_gen.send(fitnesses)
            if not invalid_ind:
                break # If there are no new solutions to try, break
        serialize_models(models, **ensemble_kwargs)
    finally:
        history_file.close()
    return models