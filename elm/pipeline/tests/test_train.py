import contextlib
import inspect

import elm.pipeline.train as elmtrain
from elm.config import DEFAULTS, ConfigParser, import_callable

# TODO: after PR #28 is merged, remove
# this try block around the import
# (PR #28 defines the function check_action_data)
try:
    from elm.pipeline.sample_util import check_action_data
except:
    check_action_data = None
old_ensemble = elmtrain.ensemble

EXPECTED_SELECTION_KEYS = ('exclude_polys',
                           'filename_filter',
                           'include_polys',
                           'metadata_filter',
                           'data_filter',
                           'geo_filters')

def return_all(*args, **kwargs):
    '''An empty function to return what is given to it'''
    return args, kwargs

@contextlib.contextmanager
def patch_ensemble():
    '''This helps test the job of getting arguments to
    ensemble by changing that function to just return its args,kwargs'''
    try:
        elmtrain.ensemble = return_all
        yield
    finally:
        elmtrain.ensemble = old_ensemble

def test_train_makes_args_kwargs_ok():
    with patch_ensemble():
        config = ConfigParser(config=DEFAULTS)
        for step in config.pipeline:
            if 'train' in step:
                break
        train_dict = config.train[step['train']]
        args, kwargs = elmtrain.train_step(config, step, None)
        assert kwargs == train_dict.get('ensemble_kwargs')
        assert args[0] is None # executor
        assert callable(args[1])   # model init func
        assert "KMeans" in repr(args[1])
        assert isinstance(args[2], dict)  # model init kwargs
        model_init_kwargs = args[2]
        for k,v in train_dict['model_init_kwargs'].items():
            assert model_init_kwargs[k] == v
        # check model init kwargs include the defaults for the method
        defaults = {k: v.default for k,v in inspect.signature(args[1]).parameters.items()}
        for k, v in defaults.items():
            if not k in (set(train_dict['model_init_kwargs']) | {'batch_size'}):
                assert model_init_kwargs.get(k) == v
        # assert fit_func, typically "fit" or "partial_fit"
        # is a method of model_init_func
        assert args[3] in dir(args[1])
        if check_action_data: # TODO remove this if statement after PR #28 merge
            check_action_data(args[4])
        # fit_func_kwargs
        assert set(args[5]) == {'post_fit_func', 'selection_kwargs'}

        assert set(args[5]['selection_kwargs']) == set(EXPECTED_SELECTION_KEYS)
        geo_filters = args[5]['selection_kwargs']['geo_filters']
        assert set(geo_filters) == {'include_polys', 'exclude_polys'}
        assert ':' in args[5]['post_fit_func']
        if args[5]['post_fit_func']:
            import_callable(args[5]['post_fit_func'])
        assert ':' in args[6]
        # model_selection_func
        import_callable(args[6])
        # model_selection_kwargs
        model_selection_kwargs = args[7]
        assert model_selection_kwargs.get('model_init_kwargs') == model_init_kwargs
        assert callable(model_selection_kwargs.get('model_init_class'))
        assert isinstance(model_selection_kwargs.get('no_shuffle'), int)
