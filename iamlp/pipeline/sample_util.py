from functools import partial

def make_no_args_sampler_func(train_or_predict_dict, config):
    '''Given sampling specs in a pipeline train or predict step,
    return a tuple of:
        no_args_sampler function,
        sampler dict,
        args that were partialed into no_args_sampler
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
    sampler_kwargs = {'included_filenames': included_filenames,}
    no_args_sampler = partial(sampler_func, *sampler_args, **sampler_kwargs)
    return no_args_sampler, sampler, sampler_args, sampler_kwargs

def make_on_each_sample_func(config, step):
    # TODO: assemble steps such as resampling, aggregation
    # that happen after the image sample is taken
    # This could use partial and import from other
    # subpackages to do each image processing step
    return None