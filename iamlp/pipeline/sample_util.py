from iamlp.config import import_callable

def run_sample_pipeline(action_data, **on_each_sample_kwargs):
    samp = action_data[0]
    sampler_func_str, sampler_args, sampler_kwargs = samp[0], samp[1], samp[2]
    sampler_func = import_callable(sampler_func_str, True, sampler_func_str)
    sample = sampler_func(*sampler_args, **sampler_kwargs)
    if len(action_data) > 1:
        for action in action_data[1:]:
            func_str, args, kwargs = action
            func = import_callable(func_str, True, func_str)
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
    sampler_name = d['sampler']
    sampler = config.samplers[sampler_name]
    sampler_func = sampler['callable']
    data_source = config.data_sources[d['data_source']]
    file_key = sampler.get('file_generator', sampler.get('file_list'))
    file_generator = config.file_generators[file_key]
    file_generator = import_callable(file_generator, True, file_generator)
    data_source['LADSWEB_LOCAL_CACHE'] = config.LADSWEB_LOCAL_CACHE
    included_filenames = tuple(file_generator(data_source))
    sampler_args = (sampler_name, sampler, config.data_sources,)
    sampler_kwargs = {'included_filenames': included_filenames,
                      }
    selection_kwargs = sampler.get('selection_kwargs') or {}
    selection_kwargs.update({
                'data_filter': selection_kwargs.get('data_filter', {}).get('callable', None),
                'metadata_filter': selection_kwargs.get('metadata_filter', {}).get('callable', None),
                'filename_filter': selection_kwargs.get('filename_filter', {}).get('callable', None),
                'geo_filters': selection_kwargs.get('geo_filters'),
                'include_polys': [config['polys'][k]
                                  for k in selection_kwargs.get('include_polys', [])],
                'exclude_polys': [config['polys'][k]
                                  for k in selection_kwargs.get('exclude_polys', [])],
            })
    sampler_kwargs.update(sampler['selection_kwargs'])
    action_data = [(sampler_func, sampler_args, sampler_kwargs)]
    if 'on_each_sample' in step:
        actions = make_on_each_sample_func(config, step, **on_each_sample_kwargs)
        action_data.extend(actions)
    return tuple(action_data), sampler, sampler_args, data_source, included_filenames

def make_on_each_sample_func(config, step):
    '''make list of (func, args, kwargs) tuples to run on_each_sample
    Params:
        config: validated config from iamlp.config.ConfigParser
        step:   a dictionary that is one step of a "pipeline" list
    '''
    on_each_sample = step['on_each_sample']
    for action in on_each_sample:
        if 'feature_selector' in action:
            context_columns = config['train'][step['train']].get('context_columns') or []
            item = ('iamlp.data_selectors.feature_selectors:feature_selector_base',
                    config['feature_selectors']['feature_selector'],
                    {'context_columns': context_columns})
        else:
            # add items to actions of the form:
            # (
            #   module_colon_func_name_as_string,
            #   args_to_func,
            #   kwargs_to_func
            # )
            raise NotImplementedError('Put other on_each_sample logic here, like resampling')
        actions.append(item)
    return actions

