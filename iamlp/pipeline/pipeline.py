
from iamlp.pipeline.train import train_step
from iamlp.pipeline.predict import predict_step
from iamlp.pipeline.required_data_sources import required_data_sources_step

def on_step(*args):
    if 'train' in step:
        return train_step(*args)
    elif 'predict' in step:
        return predict_step(*args)
    elif 'required_data_sources' in step:
        return required_data_sources_step(*args)
    else:
        raise NotImplemented

def pipeline(config, executor):

    for step in config['pipeline']:
        ret_val = on_step(config, step, executor)