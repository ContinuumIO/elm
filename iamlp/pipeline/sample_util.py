from iamlp.config import import_callable_from_string

def run_sample_pipeline(action_data):
    samp = action_data[0]
    sampler_func_str, sampler_args, sampler_kwargs = samp[0], samp[1], samp[2]
    sampler_func = import_callable_from_string(sampler_func_str, True, sampler_func_str)
    sample = sampler_func(*sampler_args, **sampler_kwargs)
    if len(action_data) > 1:
        for action in action_data[1:]:
            func_str, args, kwargs = action
            func = import_callable_from_string(func_str, True, func_str)
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
    sampler = config.samplers[d['sampler']]
    sampler_func = sampler['callable']
    data_source = config.data_sources[d['data_source']]
    file_key = sampler.get('file_generator', sampler.get('file_list'))
    file_generator = config.file_generators[file_key]
    included_filenames = tuple(file_generator(data_source))
    n_per_file = sampler['n_rows_per_sample'] // sampler['files_per_sample']
    sampler_args = (sampler_name, sampler_dict, data_sources,)
    sampler_kwargs = {'included_filenames': included_filenames,
                      'n_per_file': n_per_file,
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
        on_each_sample_kwargs = step.get('on_each_sample_kwargs') or {}
        actions = make_on_each_sample_func(config, step, **on_each_sample_kwargs)
        action_data.extend(actions)
    return tuple(action_data), sampler, data_source, included_filenames

def make_on_each_sample_func(config, step, **on_each_sample_kwargs):
    # TODO: assemble steps such as resampling, aggregation
    # that happen after the image sample is taken
    # This could use partial and import from other
    # subpackages to do each image processing step
    on_each_sample = step['on_each_sample']
    actions = []
    for action in on_each_sample:
        pass # TODO add other actions here in the form:
             # (module_colon_func_name_as_string, args_to_func, kwargs_to_func)
    return actions

