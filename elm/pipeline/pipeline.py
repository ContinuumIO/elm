from collections import defaultdict

from elm.model_selection.evolve import evolve_setup
from elm.pipeline.train import train_step
from elm.pipeline.predict import predict_step
from elm.pipeline.transform import transform_pipeline_step
from elm.pipeline.download_data_sources import download_data_sources_step
from elm.pipeline.util import get_transform_name_for_sample_pipeline

def on_step(*args, **kwargs):
    '''Evaluate a step in the pipeline'''
    step = args[1]
    if 'transform' in step:
        return ('transform', transform_pipeline_step(*args, **kwargs))
    elif 'train' in step:
        return ('train', train_step(*args, **kwargs))
    elif 'predict' in step:
        return ('predict', predict_step(*args, **kwargs))
    else:
        raise NotImplementedError('Put other operations like "change_detection" here')


def pipeline(config, executor):
    '''Run all steps of a config's "pipeline"'''
    return_values = defaultdict(lambda: {})
    transform_dict = {}
    evo_params_dict = evolve_setup(config)
    for idx, step in enumerate(config.pipeline):
        models = None
        if 'predict' in step and 'train' in return_values:
            if step['predict'] in return_values['train']:
                # instead of loading from disk the
                # models that were just created, use the
                # in-memory models returned by train step
                # already run
                models = return_values['train'][step['predict']]
        transform_key = None
        transform_key = get_transform_name_for_sample_pipeline(step)
        if 'transform' in step:
            transform_key = step['transform']
        if transform_key in transform_dict:
            transform_model = transform_dict[transform_key]
        else:
            transform_model = None
        kwargs = {'models': models,
                  'transform_model': transform_model,
                  }
        if idx in evo_params_dict:
            kwargs['evo_params'] = evo_params_dict[idx]
        step_type, ret_val = on_step(config, step, executor, **kwargs)
        return_values[step_type][step[step_type]] = ret_val
        if step_type == 'transform':
            transform_dict[step[step_type]] = ret_val
    return return_values


