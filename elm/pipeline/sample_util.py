import copy
import logging

from elm.config import import_callable

logger = logging.getLogger(__name__)

def check_action_data(action_data):
    if not isinstance(action_data, (list, tuple)):
        raise ValueError("Expected action_data to run_sample_pipeline to be a list. "
                        "Got {}".format(type(action_data)))
    for item in action_data:
        if not (isinstance(item, tuple) or len(item) == 3):
            raise ValueError('Expected each item in action_data to be a tuple of 3 items')
        func, args, kwargs = item
        func = import_callable(func)
        if not callable(func):
            raise ValueError('Expected first item in an action_data element '
                             'to be a callable, but got {}'.format(func))
        if not isinstance(args, (tuple, list)):
            raise ValueError('Expected second item in an action_data element '
                             'to be a tuple or list (args to {}). Got {}'.format(func, args))
        if not isinstance(kwargs, dict):
            raise ValueError('Expected third item in an action_data element '
                             'to be a dict (kwargs to {}).  Got {}'.format(func, kwargs))
def run_sample_pipeline(action_data, sample=None):
    check_action_data(action_data)
    if sample is None:
        samp = action_data[0]
        sampler_func_str, sampler_args, sampler_kwargs = samp
        sampler_func = import_callable(sampler_func_str, True, sampler_func_str)
        sample = sampler_func(*sampler_args, **sampler_kwargs)
    start_idx = (1 if sample is not None else 0)
    if len(action_data) > start_idx:
        for action in action_data[start_idx:]:
            func_str, args, kwargs = action
            func = import_callable(func_str, True, func_str)
            logger.debug('func {} args {} kwargs {}'.format(func, args, kwargs))
            sample = func(sample, *args, **kwargs)
    return sample

def all_sample_ops(train_or_predict_dict, config, step):
    '''Given sampling specs in a pipeline train or predict step,
    return a tuple of:
        no_args_sampler function,
        sampler dict,
        args that pwere partialed into no_args_sampler
        kwargs that were partialed into no_args_sampler'''
    d = train_or_predict_dict
    sampler_name = d.get('sampler')
    data_source = d.get('data_source')
    if sampler_name:
        sampler = config.samplers[sampler_name]
        data_generator = sampler['data_generator']
        gen = import_callable(data_generator)(**sampler)
        def sampler_func(*args, **kwargs):
            return next(gen)
        sampler_args = ()
        sampler_kwargs = {}
    else:
        data_source = config.data_sources[d['data_source']]
        sampler = config.samplers[d['data_source']]
        sampler_func = sampler['callable']
        file_key = sampler.get('file_generator', sampler.get('file_list'))
        file_generator = config.file_generators[file_key]
        file_generator = import_callable(file_generator, True, file_generator)
        data_source['LADSWEB_LOCAL_CACHE'] = config.LADSWEB_LOCAL_CACHE
        included_filenames = tuple(file_generator(data_source))
        sampler_args = (sampler_name or d['data_source'], sampler, config.data_sources,)
        sampler_kwargs = {'included_filenames': included_filenames,
                          }
        selection_kwargs = sampler.get('selection_kwargs') or {}
        selection_kwargs.update({
                    'data_filter': selection_kwargs.get('data_filter', {}).get('callable', None),
                    'metadata_filter': selection_kwargs.get('metadata_filter', {}).get('callable', None),
                    'filename_filter': selection_kwargs.get('filename_filter', {}).get('callable', None),
                    'geo_filters': selection_kwargs.get('geo_filters'),
                    'include_polys': [config.polys[k]
                                      for k in selection_kwargs.get('include_polys', [])],
                    'exclude_polys': [config.polys[k]
                                      for k in selection_kwargs.get('exclude_polys', [])],
                })
        sampler_kwargs.update(sampler['selection_kwargs'])

    action_data = [(sampler_func, sampler_args, sampler_kwargs)]
    if 'sample_pipeline' in step:
        actions = make_sample_pipeline_func(config, step)
        action_data.extend(actions)
    return tuple(action_data)

def make_sample_pipeline_func(config, step):
    '''make list of (func, args, kwargs) tuples to run sample_pipeline
    Params:
        config: validated config from elm.config.ConfigParser
        step:   a dictionary that is one step of a "pipeline" list
    '''
    sample_pipeline = step['sample_pipeline']
    action_data = []
    for action in sample_pipeline:
        if 'feature_selection' in action:
            keep_columns = copy.deepcopy(config.train[step['train']].get('keep_columns') or [])
            item = ('elm.data_selection.feature_selection:feature_selection_base',
                    (copy.deepcopy(config.feature_selection[action['feature_selection']]),),
                    {'keep_columns': keep_columns})
        else:
            # add items to action_data of the form:
            # (
            #   module_colon_func_name_as_string,        # string
            #   args_to_func,                            # tuple
            #   kwargs_to_func                           # dict
            # )
            raise NotImplementedError('Put other sample_pipeline logic here, like resampling')
        action_data.append(item)
    check_action_data(action_data)
    return action_data

