import contextlib
import inspect

from elm.config import DEFAULTS, ConfigParser, import_callable
from elm.pipeline.tests.util import patch_ensemble_predict
from elm.pipeline.sample_pipeline import check_action_data


EXPECTED_SELECTION_KEYS = ('exclude_polys',
                           'filename_filter',
                           'include_polys',
                           'metadata_filter',
                           'data_filter',
                           'geo_filters')

def expected_fit_kwargs(train_dict):
    fit_kwargs = {
            'fit_func': train_dict.get('fit_func'),
            'get_y_func': train_dict.get('get_y_func'),
            'get_y_kwargs': train_dict.get('get_y_kwargs'),
            'get_weight_func': train_dict.get('get_weight_func'),
            'get_weight_kwargs': train_dict.get('get_weight_kwargs'),
            'batches_per_gen': train_dict['ensemble_kwargs'].get('batches_per_gen'),
            'fit_kwargs': train_dict['fit_kwargs'],
        }
    return fit_kwargs


def test_train_makes_args_kwargs_ok():
    with patch_ensemble_predict() as (elmtrain, elmpredict):
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
        # is a method of model_init_class
        check_action_data(args[3][0])
        # fit_kwargs
        assert args[4] == expected_fit_kwargs(train_dict)
        assert not args[5] or (':' in args[5] and import_callable(args[5]))
        # model_scoring
        import_callable(args[6])
        # model scoring kwargs
        assert isinstance(args[7], dict)
        assert not args[8] or (':' in args[8] and import_callable(args[8]))
        model_selection_kwargs = args[9]
        assert model_selection_kwargs.get('model_init_kwargs') == model_init_kwargs
        assert callable(model_selection_kwargs.get('model_init_class'))


