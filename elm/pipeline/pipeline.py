from collections import defaultdict
import copy
import logging

import dask

from elm.config import ConfigParser
from elm.model_selection.evolve import ea_setup
from elm.pipeline.train import train_step
from elm.pipeline.predict import predict_step
from elm.pipeline.transform import transform_pipeline_step
from elm.pipeline.util import get_transform_name_for_sample_pipeline
from elm.sample_util.sample_pipeline import get_sample_pipeline_action_data
from elm.pipeline.transform import get_new_or_saved_transform_model

logger = logging.getLogger(__name__)


def _create_args_to_each_step(config, major_step,
                              return_values, transform_dict):
    '''Create the args that go to each step in a pipeline's "steps"
    Params:

        config:  elm.config.ConfigParser instance
        major_step: the step dictionary in the pipeline, e.g config.pipeline[0]
        return_values: dict of return values from previous pipeline steps
        transform_dict: dict of transform models to be
                        used in sample_pipeline if needed
    '''
    samples_per_batch = major_step.get('samples_per_batch') or 1
    random_rows = major_step.get('random_rows')
    if random_rows:
        random_rows = int(random_rows)
        random_rows_per_file = random_rows // samples_per_batch
    else:
        random_rows_per_file = None
    sample_pipeline = ConfigParser._get_sample_pipeline(config, major_step)
    if sample_pipeline and not 'random_rows' in sample_pipeline[-1]:
        if random_rows_per_file:
            sample_pipeline = sample_pipeline + [{'random_rows': random_rows_per_file}]
    data_source = config.data_sources[major_step['data_source']]
    for step in major_step['steps']:
        if step.get('transform') in transform_dict:
            transform_model = transform_dict[step['transform']]
        else:
            transform_model = get_new_or_saved_transform_model(config,
                                                               sample_pipeline,
                                                               data_source,
                                                               step)
        if transform_model:
            break
    sample_pipeline_info = (config, sample_pipeline, data_source,
                           transform_model,
                           samples_per_batch)
    return sample_pipeline_info


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


def _run_steps(return_values, transform_dict, evo_params_dict, client, steps, config, sample_pipeline_info):
    '''Run the "steps" within a sample_pipeline dict's "steps"'''

    for idx, step in enumerate(steps):
        logger.info('Pipeline step: {}'.format(repr(step)))
        logger.info('Run pipeline step {}'.format(step))
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
                  'sample_pipeline_info': sample_pipeline_info}
        if idx in evo_params_dict:
            kwargs['evo_params'] = evo_params_dict[idx]
        step_type, ret_val = on_step(config, step, client, **kwargs)
        return_values[step_type][step[step_type]] = ret_val
        if step_type == 'transform':
            transform_dict[step[step_type]] = ret_val
    return return_values, transform_dict


def pipeline(config, client):
    '''Run all steps of a config's "pipeline"
    Parameters:
        config: elm.config.ConfigParser instance
        client: Executor/client from Distributed, thread pool
                or None for serial evaluation
    '''
    sample_pipeline_info = None
    return_values = defaultdict(lambda: {})
    transform_dict = {}
    evo_params_dict = ea_setup(config)
    for idx, step in enumerate(config.pipeline):
        sample_pipeline_info = _create_args_to_each_step(config, step, return_values, transform_dict)
        rargs = (return_values, transform_dict, evo_params_dict,
                 client, step['steps'], config,
                 sample_pipeline_info)
        return_values, transform_dict = _run_steps(*rargs)
    return return_values


