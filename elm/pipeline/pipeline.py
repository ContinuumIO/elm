
from elm.pipeline.train import train_step
from elm.pipeline.predict import predict_step
from elm.pipeline.download_data_sources import download_data_sources_step

def on_step(*args):
    '''Evaluate a step in the pipeline'''
    step = args[1]
    if 'train' in step:
        return train_step(*args)
    elif 'predict' in step:
        return predict_step(*args)
    elif 'download_data_sources' in step:
        return download_data_sources_step(*args)
    else:
        raise NotImplemented('Put other operations like "change_detection" here')

def pipeline(config, executor):
    '''Run all steps of a config's "pipeline"'''
    for step in config.pipeline:
        ret_val = on_step(config, step, executor)
