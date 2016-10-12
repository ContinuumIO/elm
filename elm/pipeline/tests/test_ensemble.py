import contextlib
import inspect
import os

import pytest

from elm.config import DEFAULTS, ConfigParser, import_callable
from elm.pipeline.tests.util import patch_ensemble_predict
from elm.sample_util.sample_pipeline import check_pipe


EXPECTED_SELECTION_KEYS = ('exclude_polys',
                           'filename_filter',
                           'include_polys',
                           'metadata_filter',
                           'data_filter',
                            'geo_filters')

DEFAULTS2 = ConfigParser(config=DEFAULTS)

@pytest.mark.parametrize('ds_name, ds_dict', DEFAULTS2.data_sources.items())
def test_train_makes_args_kwargs_ok(ds_name, ds_dict):
    with patch_ensemble_predict() as (ensemble_module, elmpredict):
        config = ConfigParser(config=DEFAULTS)
        transform_model = None
        for step1 in config.pipeline:
            for step in step1['steps']:
                if 'train' in step:
                    break
        train_dict = config.train[step['train']]
        args, kwargs = ensemble_module.ensemble([{'flatten': 'C'}],
                                             ds_dict,
                                             config=config,
                                             step=step,
                                             client=None,
                                             model_args=None,
                                             ensemble_kwargs=None,
                                             transform_model=transform_model,
                                             evo_params=None,
                                             samples_per_batch=1,
                                             )
        (client,
         model_args,
         transform_model,
         sample_pipeline,
         data_source) = args
        for k, v in config.ensembles[train_dict['ensemble']].items():
            assert kwargs[k] == v
        (model_init_class,
         model_init_kwargs,
         fit_method,
         fit_args,
         fit_kwargs,
         model_scoring,
         model_scoring_kwargs,
         model_selection,
         model_selection_kwargs,
         step_type,
         step_name,
         classes) = model_args
        assert client is None
        assert callable(model_init_class)   # model init func
        assert "KMeans" in repr(model_init_class)
        assert isinstance(model_init_kwargs, dict)  # model init kwargs
        for k,v in train_dict['model_init_kwargs'].items():
            assert model_init_kwargs[k] == v
        # check model init kwargs include the defaults for the method
        sig = inspect.signature(model_init_class)
        defaults = {k: v.default for k,v in sig.parameters.items()}
        for k, v in defaults.items():
            if not k in (set(train_dict['model_init_kwargs']) | {'batch_size'}):
                assert model_init_kwargs.get(k) == v
        # assert fit_func, typically "fit" or "partial_fit"
        # is a method of model_init_class
        check_pipe(fit_args[0])
        # fit_kwargs
        data_source = config.data_sources[config.pipeline[0]['data_source']]
        ensemble = config.ensembles[train_dict['ensemble']]
        assert fit_kwargs == {'fit_kwargs': train_dict.get('fit_kwargs') or {}}
        # model_scoring
        assert not model_scoring or (':' in model_scoring and import_callable(model_scoring))
        # model scoring kwargs

        assert isinstance(model_scoring_kwargs, dict)
        assert not model_selection or (':' in model_selection and import_callable(model_selection))
        assert isinstance(model_selection_kwargs, dict)
        assert model_selection_kwargs.get('model_init_kwargs') == model_init_kwargs
        assert callable(model_selection_kwargs.get('model_init_class'))
        if any(any('transform' in step for step in step1['steps'])
               for step1 in config.pipeline):
            assert transform_model is not None
        else:
            assert transform_model is None

