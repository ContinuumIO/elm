from elm.model_selection.evolve import evo_general
from elm.pipeline.ensemble import (_validate_ensemble_members,
                                   _prepare_fit_kwargs,
                                   _get_model_selection_func,
                                   _run_model_selection_func,
                                   _serialize,
                                   _fit_list_of_models)


def _model_args_to_model(model_args, transform_dict):
    fit_kwargs = _prepare_fit_kwargs(model_args, transform_dict)
    model = import_callable(model_args.model_init_class)(**model_args.model_init_kwargs)
    return ((model,) + tuple(fit_args), fit_kwargs)


def _fit_invalid_ind(individual_to_new_config, step_to_model_args,
                    pop, invalid_ind, transform_dict,
                    map_function, get_results):
    args_kwargs = tuple(on_each_evo_yield(individual_to_new_config,
                                          step_to_model_args,
                                          pop, invalid_ind,
                                          transform_dict))
    models = _fit_list_of_models(args_kwargs, map_function, get_results)
    fitnesses = [model._score for model in models]
    return models, fitnesses


def serialize_deap_outputs(final_pop, record, logbook):
    # TODO dump these to pickle
    return True

def ea_pipeline(map_function,
                get_results,
                model_args,
                transform_model_args,
                transform_dict,
                evo_params,
                **ensemble_kwargs):
    control = evo_params.control
    control.update(copy.deepcopy(ensemble_kwargs))
    if transform_model_args is not None:
        step_to_model_args = {transform_model_args.step_name: transform_model_args}
    else:
        step_to_model_args = {}
    step_to_model_args[model_args.step_name] = model_args
    ensemble_size = ensemble_kwargs['init_ensemble_size']
    n_generations = ensemble_kwargs['n_generations']
    get_results = partial(wait_for_futures, executor=executor)
    model_names = ensemble_kwargs.get('model_names', None)
    pop = evo_init_func(evo_params)

    for fitness, ind in zip(fitnesses, pop):
        ind.fitness.values = fitness
    required_args, _, _ = get_args_kwargs_defaults(evo_general)
    evo_args = [toolbox, pop]
    for a in required_args[2:]:
        if a not in control:
            raise ValueError('Expected {} in {} (control kwargs to evolutionary algorithm)'.format(a, control))
        evo_args.append(control[a])
    ea_gen = evo_general(*evo_args)
    next(ea_gen) # dummy
    invalid_ind = pop
    while True:
        if not invalid_ind:
            break
        models, fitnesses = _fit_invalid_ind(evo_params.individual_to_new_config,
                                            step_to_model_args,
                                            pop,
                                            invalid_ind,
                                            transform_dict,
                                            map_function,
                                            get_results)
        (pop, invalid_ind, record, logbook) = ea_gen.send(fitnesses)
    _serialize(models, **ensemble_kwargs)
    serialize_deap_outputs(pop, record, logbook)
    return models


def on_each_evo_yield(individual_to_new_config,
                      step_to_model_args,
                      pop,
                      invalid_ind,
                      transform_dict):
    for ind in invalid_ind:
        new_step_to_model_args = individual_to_new_config(ind, step_to_model_args)
        for step, model_args in sorted(new_step_to_model_args.items(), key=lambda x:0 if x[1].step_type == 'transform' else 1):
            has_yielded = False
            if model_args.step_type == 'transform':
                transform_dict = copy.deepcopy(transform_dict) if transform_dict else {}
                transform_dict[model_args.step_name] =  _model_args_to_model(model_args, transform_dict)
                model_args_kwargs = transform_dict[model_args.step_name]
            else:
                model_args_kwargs = _model_args_to_model(model_args, transform_dict)
                has_yielded = True
                yield model_args_kwargs
        if not has_yielded:
            yield model_args_kwargs

