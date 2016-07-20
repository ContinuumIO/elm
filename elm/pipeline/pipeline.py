from collections import defaultdict

from elm.pipeline.train import train_step
from elm.pipeline.predict import predict_step
from elm.pipeline.download_data_sources import download_data_sources_step

def on_step(*args, **kwargs):
    '''Evaluate a step in the pipeline'''
    step = args[1]
    if 'train' in step:
        return ('train', train_step(*args, **kwargs))
    elif 'predict' in step:
        return ('predict', predict_step(*args, **kwargs))
    else:
        raise NotImplemented('Put other operations like "change_detection" here')

def pipeline(config, executor):
    '''Run all steps of a config's "pipeline"'''
    return_values = defaultdict(lambda: {})
    for idx, step in enumerate(config.pipeline):
        models = None
        if 'predict' in step and 'train' in return_values:
            if step['predict'] in return_values['train']:
                # instead of loading from disk the
                # models that were just created, use the
                # in-memory models returned by train step
                # already run
                models = return_values['train'][step['predict']]
        kwargs = {'models': models}
        step_type, ret_val = on_step(config, step, executor, **kwargs)
        return_values[step_type][step[step_type]] = ret_val
