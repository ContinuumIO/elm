from collections import defaultdict

from elm.pipeline.train import train_step
from elm.pipeline.predict import predict_step
from elm.pipeline.transform import transform_pipeline_step
from elm.pipeline.download_data_sources import download_data_sources_step

def switch_step(*args, **kwargs):
    '''Evaluate a step in the pipeline'''
    step = args[1]
    if 'transform' in step:
        return ('transform', transform_pipeline_step(*args, **kwargs))
    elif 'train' in step:
        return ('train', train_step(*args, **kwargs))
    elif 'predict' in step:
        return ('predict', predict_step(*args, **kwargs))
    else:
        raise NotImplemented('Put other operations like "change_detection" here')

def on_step(config, step, executor, return_values, transform_dict, **kwargs):
    models = None
    if 'predict' in step and 'train' in return_values:
        if step['predict'] in return_values['train']:
            # instead of loading from disk the
            # models that were just created, use the
            # in-memory models returned by train step
            # already run
            models = return_values['train'][step['predict']]
    kwargs = {'models': models, 'transform_dict': transform_dict}
    step_type, ret_val = switch_step(config, step, executor, **kwargs)
    return_values[step_type][step[step_type]] = ret_val
    if step_type == 'transform':
        transform_dict[step[step_type]] = ret_val
    return (step_type, return_values, transform_dict, ret_val)



def _several_steps_objective(config, steps, executor, return_values, transform_dict, evo_params, **kwargs):
    for step in steps:
        kwargs['evo_params'] = evo_params
        (step_type, return_values, transform_dict, ret_val) = on_step(config, step,
                                                             executor,
                                                             **kwargs)
    return (step_type, return_values, transform_dict, ret_val)


def evolve_pipeline(config,
                    executor,
                    step_name_to_param_grid_name,
                    param_grid_name_to_deap,
                    transform_dict,
                    return_values):
    params_gen = evolve(config, step_name_to_param_grid_name, param_grid_name_to_deap)
    currently_on = 0
    for evo_params in params_gen:
        idxes = ['span_idxes']
        while currently_on < min(idxes):
            # If there are steps in pipeline
            # before first genetic algorithm starts
            out = on_step(step_type, return_values, transform_dict, ret_val)
            (step_type, return_values, transform_dict, ret_val) = out
            currently_on += 1
        steps = [config.pipeline[idx] for idx in idxes]
        out = _several_steps_objective(config, steps, executor, return_values, transform_dict, evo_params, **kwargs)
        (step_type, return_values, transform_dict, ret_val) = out
        currently_on = max(idxes) + 1
    # Finish any non-genetic algorithm steps left over
    while currently_on < len(pipeline):
        out = on_step(step_type, return_values, transform_dict, ret_val)
        (step_type, return_values, transform_dict, ret_val) = out
        currently_on += 1
    return return_values, transform_dict

def pipeline(config, executor):
    '''Run all steps of a config's "pipeline"'''
    return_values = defaultdict(lambda: {})
    transform_dict = {}
    evolve_meta = get_evolve_meta(config)
    if evolve_meta:
        (step_name_to_param_grid_name, param_grid_name_to_deap) = evolve_meta
        return evolve_pipeline(config, executor,
                               step_name_to_param_grid_name,
                               param_grid_name_to_deap,
                               transform_dict,
                               return_values)
    for idx, step in enumerate(config.pipeline):
        (step_type, return_values, transform_dict, ret_val) = on_step(config, step,
                                                             executor,
                                                             **kwargs)
    return return_values, transform_dict