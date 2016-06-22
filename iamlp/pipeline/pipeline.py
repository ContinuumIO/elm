
from iamlp.pipeline.train import train_step
from iamlp.pipeline.predict import predict_step
from iamlp.pipeline.required_data_sources import required_data_sources_step

def on_step(config, step):
    if 'train' in step:
        return train_step(config, step)
    elif 'predict' in step:
        return predict_step(config, step)
    elif 'required_data_sources' in step:
        return required_data_sources_step(config, step)

def pipeline(config):
    pipe = config['pipeline']
    for step in pipe:
        ret_val = on_step(config, step)
