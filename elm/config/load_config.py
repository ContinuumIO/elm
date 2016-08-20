from collections import Iterable
import copy
import logging
import numbers
import os
import traceback

import numpy as np
import sklearn.feature_selection as skfeat
import sklearn.preprocessing as skpre
import yaml


from elm.config.env import parse_env_vars, ENVIRONMENT_VARS_SPEC
from elm.config.util import (ElmConfigError,
                               import_callable)
from elm.model_selection.util import get_args_kwargs_defaults
from elm.acquire.ladsweb_meta import validate_ladsweb_data_source
from elm.config.defaults import DEFAULTS, CONFIG_KEYS


logger = logging.getLogger(__name__)

def sample_args_generator_from_list(some_list, *args, **kwargs):
    yield from iter(some_list)


PIPELINE_ACTIONS = ('train',
                    'predict',
                    'transform',)
                    # TODO other steps like
                    #'change_detection',
                    #'spatial_summary'

SAMPLE_PIPELINE_ACTIONS = ('transform',
                           'feature_selection',
                           'sklearn_preprocessing',
                           'random_sample',
                           'sample_pipeline',
                           'get_y',
                           'get_weight',)
REQUIRES_METHOD = ('train', 'predict', 'transform')
class ConfigParser(object):
    # The list below reflects the order
    # in which the keys of the config
    # are validated. (the _validate_* private
    # methods are order sensitive.)

    config_keys = CONFIG_KEYS
    defaults = DEFAULTS
    all_words = ('all', ['all'],)
    def __init__(self, config_file_name=None, config=None):
        '''Parses a config structure

        Params:
            config_file_name: file name of a config
            config:           dict loaded from yaml/json already
        '''

        if not config:
            if not config_file_name or not os.path.exists(config_file_name):
                raise ElmConfigError('config_file_name {} does not '
                                       'exist'.format(config_file_name))
            with open(config_file_name) as f:
                self.raw_config = yaml.safe_load(f.read())
        elif config:
            self.raw_config = copy.deepcopy(config)
        else:
            raise ElmConfigError('ConfigParser expects either '
                                   'config_file_name or config '
                                   '(dict) as keyword arguments')
        self.config = copy.deepcopy(DEFAULTS)
        for k, v in copy.deepcopy(self.raw_config).items():
            if isinstance(v, dict):
                if k in self.config:
                    self.config[k].update(v)
                else:
                    self.config[k] = v
            else:
                self.config[k] = v
        self._update_for_env()
        self._interpolate_env_vars()
        self.validate()


    def _interpolate_env_vars(self):
        '''Replace config items like env:ELM_TRANSFORM_PATH with relevant env var'''
        import elm.config.dask_settings as elm_dask_settings
        config_str = yaml.dump(self.config)
        for env_var in (ENVIRONMENT_VARS_SPEC['str_fields_specs'] +
                        ENVIRONMENT_VARS_SPEC['int_fields_specs']):
            env_str = 'env:{}'.format(env_var['name'])
            if env_str in config_str:
                config_str = config_str.replace(env_str,
                                                getattr(elm_dask_settings,
                                                        env_var['name']))
        self.config = yaml.load(config_str)

    def _update_for_env(self):
        '''Update the config based on environment vars'''
        import elm.config.dask_settings as elm_dask_settings
        for k, v in parse_env_vars().items():
            if v:
                setattr(self, k, v)
        for str_var in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
            choices = str_var.get('choices', [])
            val = getattr(self, str_var['name'], None)
            if choices and val not in choices:
                raise ElmConfigError('Expected config key or env '
                                       'var {} to be in '
                                       '{} but got {}'.format(k, choices, val))
            setattr(elm_dask_settings, str_var['name'], val)
        for int_var in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
            val = getattr(self, int_var['name'], None)
            setattr(elm_dask_settings, int_var['name'], val)
        elm_dask_settings.SERIAL_EVAL = self.SERIAL_EVAL = self.config['DASK_EXECUTOR'] == 'SERIAL'
        logger.info('Running with DASK_EXECUTOR={} '
                    'DASK_SCHEDULER={}'.format(elm_dask_settings.DASK_EXECUTOR,
                                               elm_dask_settings.DASK_SCHEDULER))

    def _validate_custom_callable(self, func_or_not, required, context):
        '''Validate a callable given like "numpy:mean" can be imported'''
        if callable(func_or_not):
            return func_or_not
        if func_or_not or (not func_or_not and required):
            if not isinstance(func_or_not, str):
                raise ElmConfigError('In {} expected {} to be a '
                                       'string'.format(func_or_not, context))
        return import_callable(func_or_not, required=required, context=context)

    def _validate_readers(self):
        '''Validate the "readers" section of config'''
        err_msg = "Expected a 'readers' dictionary in config"
        self.readers = readers = self.config.get('readers') or {}
        if not readers or not isinstance(readers, dict):
            raise ElmConfigError(err_msg)
        for k, v in readers.items():
            if not v or not isinstance(v, dict):
                raise ElmConfigError('In readers:{} expected {} to be '
                                       'a dict'.format(k, v))

            load = v.get('load_array')
            bounds = v.get('load_meta')
            self._validate_custom_callable(load, True, 'readers:{} load_array'.format(k))
            self._validate_custom_callable(bounds, True, 'readers:{} load_meta'.format(k))
            self.readers[k] = v


    def _validate_band_specs(self, band_specs, name):
        '''Validate "band_specs"'''

        if not band_specs or not isinstance(band_specs, list):
            raise ElmConfigError('data_sources:{} gave band_specs which are not a '
                                   'list {}'.format(name, band_specs))
        for band_spec in band_specs:
            if not isinstance(band_spec, (tuple, list)) or not len(band_spec) == 3 or not all(isinstance(b, str) for b in band_spec):
                raise ElmConfigError("band_spec {} in data_sources:{} "
                                       "did not have 3 strings "
                                       "(metadata key search phrase,"
                                       "metadata value search phrase, "
                                       "band name)".format(band_spec, name))

    def _validate_one_data_source(self, name, ds):
        '''Validate one data source within "data_sources"
        section of config'''

        if not name or not isinstance(name, str):
            raise ElmConfigError('Expected a "name" key in {}'.format(d))
        reader = ds.get('reader')
        if not reader in self.readers:
            raise ElmConfigError('Data source config dict {} '
                                   'refers to a "reader" {} that is not defined in '
                                   '"readers"'.format(reader, self.readers))
        download = ds.get('download', '') or ''
        if download and not download in self.downloads:
            raise ElmConfigError('data_source {} refers to a '
                                   '"download" {} not defined in "downloads"'
                                   ' section'.format(ds, download))
        self._validate_band_specs(ds.get('band_specs'), name)
        s = ds.get('sample_args_generator')
        if not s in self.sample_args_generators:
            raise ElmConfigError('Expected data_source: '
                                 'sample_args_generator {} to be in '
                                 'sample_args_generators.keys()')
        sample_args_generator = self.sample_args_generators[s]
        self._validate_custom_callable(sample_args_generator,
                                True,
                                'train:{} sample_args_generator'.format(name))
        sample_from_args_func = ds.get('sample_from_args_func')
        self._validate_custom_callable(sample_from_args_func,
                                True,
                                'train:{} sample_from_args_func'.format(name))
        self._validate_selection_kwargs(ds, name)
        keep_columns = ds.get('keep_columns') or []
        self._validate_type(keep_columns, 'keep_columns', (tuple, list))
        ds['keep_columns'] = keep_columns


    def _validate_data_sources(self):
        '''Validate all "data_sources" of config'''
        self.data_sources = self.config.get('data_sources', {}) or {}
        if not self.data_sources or not isinstance(self.data_sources, dict):
            raise ElmConfigError('Expected "data_sources" in config to be a '
                                   'dict. Got: {}'.format(self.data_sources))
        for name, ds in self.data_sources.items():
            self._validate_one_data_source(name, ds)

    def _validate_sample_args_generators(self):
        '''Validate the "sample_args_generators" section of config'''
        self.sample_args_generators = self.config.get('sample_args_generators', {}) or {}
        if not isinstance(self.sample_args_generators, dict):
            raise ElmConfigError('Expected sample_args_generators to be a dict, but '
                                   'got {}'.format(self.sample_args_generators))
        for name, file_gen in self.sample_args_generators.items():
            if not name or not isinstance(name, str):
                raise ElmConfigError('Expected "name" key in sample_args_generators {} ')
            self._validate_custom_callable(file_gen, True,
                                           'sample_args_generators:{}'.format(name))

    def _validate_positive_int(self, val, context):
        '''Validate that a positive int was given'''
        if not isinstance(val, int) and val:
            raise ElmConfigError('In {} expected {} to be an int'.format(context, val))

    def _validate_poly(self, name, poly):
        return True # TODO this should validate a list entry
                    # in polys list of the config.  E.g. how
                    # is the poly loaded from file?

    def _validate_polys(self):
        self.polys = self.config.get('polys', {}) or {}
        for name, poly in self.polys:
            self._validate_poly(name, poly)

    def _validate_selection_kwargs(self, data_source, name):
        '''Validate the "selection_kwargs" related to
        sample pre-processing'''
        selection_kwargs = data_source.get('selection_kwargs')
        if not selection_kwargs:
            return
        if not isinstance(selection_kwargs, dict):
            raise ElmConfigError('In data_source:{} expected '
                                   '"selection_kwargs" to be '
                                   'a dict'.format(selection_kwargs))
        selection_kwargs['geo_filters'] = selection_kwargs.get('geo_filters', {}) or {}
        for poly_field in ('include_polys', 'exclude_polys'):
            pf = selection_kwargs['geo_filters'].get(poly_field, []) or []
            for item in pf:
                if not item in self.polys:
                    raise ElmConfigError('config\'s selection_kwargs dict {} '
                                           '"include_polys" or "exclude_poly" '
                                           'must refer to a list of keys from config\'s '
                                           '"polys"'.format(self.selection_kwargs))
        for filter_name in ('data_filter', 'metadata_filter', 'filename_filter'):
            f = selection_kwargs.get(filter_name, {})
            if f:
                self._validate_custom_callable(f, True,
                                               'selection_kwargs:{} - {}'.format(name, filter_name))
            else:
                selection_kwargs[filter_name] = None
        self.data_sources[name]['selection_kwargs'] = selection_kwargs


    def _validate_resamplers(self):
        self.resamplers = self.config.get('resamplers', {}) or {}
        if self.resamplers:
            raise NotImplementedError('implement resampling logic')

    def _validate_aggregations(self):
        self.aggregations = self.config.get('aggregations', {}) or {}
        if self.aggregations:
            raise NotImplementedError('implement aggregations logic')

    def _validate_masks(self):
        self.masks = self.config.get('masks', {}) or {}
        if self.masks:
            raise NotImplementedError('implement masks logic')

    def _validate_add_features(self):
        self.add_features = self.config.get('add_features', {}) or {}
        if self.add_features:
            raise NotImplementedError('implement add_features logic')

    def _validate_type(self, k, name, typ):
        if k and not isinstance(k, typ):
            raise ElmConfigError('In {} expected a {} but '
                                   'found {}'.format(name, typ, k))

    def _validate_all_or_type(self, k, name, typ):
        if k != 'all':
            self._validate_type(k, name, typ)

    def _validate_feature_selection(self):
        '''Validate the "feature_selection" section of config'''
        feature_selection = self.config.get('feature_selection') or {}
        if not feature_selection:
            return True
        self._validate_type(feature_selection, 'feature_selection', dict)
        for k, s in feature_selection.items():
            self._validate_type(k, 'feature_selection:{}'.format(k), str)
            self._validate_type(s, 'feature_selection:{}'.format(s), dict)
            selection = s.get('selection')
            if selection != 'all':
                self._validate_custom_callable(selection, True,
                                               'feature_selection:{}'.format(k))
                scoring = s.get('scoring')
                no_scoring = ('all', 'sklearn.feature_selection:VarianceThreshold')
                if scoring and scoring not in dir(skfeat) and not selection in no_scoring:
                    self._validate_custom_callable(scoring, True,
                            'feature_selection:{} scoring'.format(k))
                make_scorer_kwargs = s.get('make_scorer_kwargs') or {}
                self._validate_type(make_scorer_kwargs, 'make_scorer_kwargs', dict)

                # TODO is there further validation of make_scorer_kwargs
                # that can be done -
                #    they are passed to sklearn.metrics.make_scorer

                s['make_scorer_kwargs'] = make_scorer_kwargs
            else:
                scoring = None
            s['scoring'] = scoring


            feature_choices = s.get('choices') or 'all'
            self._validate_all_or_type(feature_choices,
                                       'feature_selection:{} choices'.format(k),
                                       (list, tuple))
            s['choices'] = feature_choices
            feature_selection[k] = s
        self.feature_selection = feature_selection

    def _validate_one_model_scoring(self, key, value):
        '''Validate one model_scoring key-value (one scoring configuration)'''
        logger.info(repr((key, value)))
        from elm.model_selection.metrics import METRICS
        scoring = value.get('scoring')
        if scoring in METRICS:
            context = 'model_scoring:{}'.format(key)
            # TODO more validation of custom scorers?
            scoring_agg = value.get('scoring_agg')
            if scoring_agg:
                self._validate_custom_callable(scoring_agg, True, context + '(scoring_agg)')
            greater_is_better = value.get('greater_is_better') or None
            score_weights = value.get('score_weights') or None
            err_msg = 'In {}, expected either one of "greater_is_better", "score_weights"'
            if greater_is_better is not None:
                self._validate_type(greater_is_better, context + '(greater_is_better)', bool)
                self.config['model_scoring'][key]['score_weights'] = [-1 if not greater_is_better else 1]
            if score_weights is not None:
                self._validate_type(score_weights, context + '(score_weights)', Iterable)
            if greater_is_better is None and score_weights is None:
                raise ElmConfigError(err_msg)
        else:
            # I think there is little validation
            # that can be done?
            pass
    def _validate_model_scoring(self):
        '''Validate the "model_scoring" dict of config'''
        ms = self.config.get('model_scoring') or {}
        self._validate_type(ms, 'model_scoring', dict)
        for k, v in ms.items():
            self._validate_type(v, 'model_scoring:{}'.format(k), dict)
            self._validate_one_model_scoring(k, v)
        self.model_scoring = ms

    def _validate_train_or_transform_funcs(self, name, t):
        '''Validate functions given in "train" section of config'''
        if not isinstance(t, dict):
            raise ElmConfigError('In train:{} expected a dict '
                                 'but found {}'.format(name, t))
        training_funcs = (('model_init_class', True),
                           ('get_y_func', False),
                           ('get_weight_func', False),
                           )

        has_fit_func = False
        has_funcs = {}
        sel = t.get('model_selection')
        no_selection = not sel or sel in ('no_selection',)
        for f, required in training_funcs:
            cls_or_func = self._validate_custom_callable(t.get(f), required,
                                           'train:{} - {}'.format(name, f))
            has_funcs[f] = bool(cls_or_func)
            if f == 'model_init_class':
                model_init_class = cls_or_func
                fit_func = getattr(model_init_class, 'partial_fit', getattr(model_init_class, 'fit', None))
                if fit_func is None:
                    raise ElmConfigError('model_init_class {} '
                                         'does not have "fit" or "partial_fit" method'.format(t.get('model_init_class')))
                fargs, fkwargs,var_keyword = get_args_kwargs_defaults(cls_or_func.fit)
        requires_y = any(x.lower() == 'y' for x in fargs)
        if not fkwargs.get('sample_weight') and has_funcs['get_weight_func']:
            raise ElmConfigError('train:{} - {} does not support a '
                                 '"sample_weight" (sample_weights were implied '
                                 'giving "get_sample_weight" '
                                 'function {}'.format(name, model_init_class, t['get_sample_weight']))
        return has_fit_func, requires_y, no_selection

    def _is_transform_major_pipeline_step(self, transform_name):
        pipeline = self.config.get('pipeline') or {}
        self._validate_type(pipeline, 'pipeline', (list, tuple))
        for idx, step in enumerate(pipeline):
            self._validate_type(step, 'pipeline step:{}'.format(idx), dict)
            if step.get('transform') == transform_name:
                return True
        return False

    def _validate_one_train_or_transform_entry(self, train_or_transform, name, t):
        '''Validate one dict within "train" or "transform" section of config'''
        has_fit_func, requires_y, no_selection = self._validate_train_or_transform_funcs(name, t)
        kwargs_fields = tuple(k for k in t if k.endswith('_kwargs'))
        for k in kwargs_fields:
            self._validate_type(t[k], '{}:{}'.format(train_or_transform, k), dict)
        if not no_selection:
            mod = t.get('model_selection')
            if not mod in self.model_selection:
                raise ElmConfigError('{}:model_selection {} is not a '
                                     'key in config\'s model_selection'.format(train_or_transform, mod))
            if t.get('sort_fitness'):
                self._validate_custom_callable(t.get('sort_fitness'),
                                               True,
                                               '{}:{} (sort_fitness)'.format(train_or_transform,repr(t.get('sort_fitness'))))
            ms = t.get('model_scoring')
            if ms:
                self._validate_type(ms, '{}:{} (model_scoring)'.format(train_or_transform,name),
                                    (str, numbers.Number, tuple))
                if not ms in self.model_scoring:
                    raise ElmConfigError('{}:{}\'s model_scoring: {} is not a '
                                         'key in config\'s model_scoring '
                                         'dict'.format(train_or_transform, name, ms))
            else:
                raise ElmConfigError('In {}:{} model_scoring must be defined if model_selection is used'.format(train_or_transform, name))
        data_source = t.get('data_source')
        if train_or_transform == 'transform':
            is_major_step = self._is_transform_major_pipeline_step(name)
        else:
            is_major_step = True
        if not data_source in self.data_sources:
            if train_or_transform == 'train':
                raise ElmConfigError('{} dict at key {} refers '
                                       'to a data_source {} that is '
                                       'not defined in '
                                       '"data_sources"'.format(train_or_transform, name, repr(data_source)))
        if is_major_step:
            data_source = self.data_sources[data_source]
            if requires_y:
                self._validate_custom_callable(data_source.get('get_y_func'),
                                               True,
                                               '{}:get_y_func (required with '
                                               '{})'.format(train_or_transform,
                                                            repr(t.get('model_init_class'))))
            output_tag = t.get('output_tag')
            self._validate_type(output_tag, 'train:output_tag', str)
            band_specs = data_source['band_specs']
            t['band_names'] = [x[-1] for x in band_specs]
            ensemble = t.get('ensemble')
            if not ensemble or not ensemble in self.ensembles:
                raise ElmConfigError('Each train or transform dict must have an '
                                     '"ensemble" key that is also a key in "ensembles"')
        self.config[train_or_transform][name] = getattr(self, train_or_transform)[name] = t

    def _validate_train_or_transform(self, train_or_transform):
        '''Validate the "train" or "transform" section of config'''
        setattr(self, train_or_transform, self.config.get(train_or_transform, {}) or {})
        for name, t in getattr(self, train_or_transform).items():
            self._validate_one_train_or_transform_entry(train_or_transform,name, t)

    def _validate_train(self):
        '''Validate the "train" section of config'''
        return self._validate_train_or_transform('train')

    def _validate_transform(self):
        '''Validate the "transform" section of config'''
        return self._validate_train_or_transform('transform')

    def _validate_sklearn_preprocessing(self):
        '''Validate "sklearn_preprocessing" dict in config'''
        self.sklearn_preprocessing = self.config.get('sklearn_preprocessing') or {}
        self._validate_type(self.sklearn_preprocessing, 'sklearn_preprocessing', dict)
        for k, v in self.sklearn_preprocessing.items():
            self._validate_type(v, 'sklearn_preprocessing:{}'.format(k), dict)
            if v.get('method') in dir(skpre):
                pass
            else:
                self._validate_custom_callable(v.get('method'),
                                               True,
                                               'sklearn_preprocessing:{} - method'.format(k))
            if v['method'].split(':')[-1] == 'FunctionTransformer':
                self._validate_custom_callable(v.get('func'),
                                               True,
                                               'sklearn_preprocessing:{} - func passed to FunctionTransformer'.format(k))


    def _validate_ensembles(self):
        '''Validate the "ensembles" section of config'''
        self.ensembles = self.config.get('ensembles') or {}
        self._validate_type(self.ensembles, 'config - ensembles', dict)
        for f in ('saved_ensemble_size', 'n_generations',
                  'init_ensemble_size', 'batches_per_gen'):
            for k in self.ensembles:
                self._validate_positive_int(self.ensembles[k].get(f), f)

    def _validate_model_selection(self):
        '''Validate "model_selection" section of config'''
        self.model_selection = self.config.get('model_selection') or {}
        self._validate_type(self.model_selection, 'config - model_selection', dict)
        for k,v in self.model_selection.items():
            self._validate_type(v.get('kwargs'),
                                'model_selection:{} - {}'.format(k, 'kwargs'),
                                dict)
            self._validate_custom_callable(v.get('func'),
                                           True,
                                           'model_selection:{} - {}'.format(k, 'func'))

    def _validate_predict(self):
        '''Validate the "predict" section of config'''
        self.predict = self.config.get('predict', {}) or {}
        return True # TODO validate predict config

    @classmethod
    def _get_sample_pipeline(cls, config, step):
        '''Get the "sample_pipeline" which may be composed of
        several named "sample_pipeline" elements, e.g.
        pipeline:
            - method: partial_fit
              sample_pipeline:
              - {get_y: true}
              - {sklearn_preprocessing: require_positive}
              - {sample_pipeline: log10}
              transform: pca
        '''
        has_seen = set()
        sp = step.get('sample_pipeline') or []
        err_msg = 'Invalid reference {} is not a key in "sample_pipelines"'
        if isinstance(sp, str):
            if not sp in config.sample_pipelines:
                raise ElmConfigError('Pipeline step refers to a '
                                     'sample_pipeline ({}) that is not in '
                                     'config\'s sample_pipeline section'.format(sp))
            return config.sample_pipelines[sp]
        def clean(sp, has_seen):
            sample_pipeline = []
            for item in sp:
                if 'sample_pipeline' in item:
                    sp2 = item['sample_pipeline']
                    if sp2 in has_seen:
                        raise ElmConfigError('Recursive "sample_pipeline":{}'.format(sp2))
                    has_seen.add(sp2)
                    if isinstance(sp2, (tuple, list)):
                        sample_pipeline.extend(sp2)
                    elif sp2 and sp2 in config.sample_pipelines:
                        sample_pipeline2, has_seen = clean(config.sample_pipelines[sp2], has_seen)
                        sample_pipeline.extend(sample_pipeline2)
                    else:
                        raise ElmConfigError(err_msg.format(sp2))
                else:
                    sample_pipeline.append(item)
            return sample_pipeline, has_seen
        sample_pipeline, _ = clean(sp, has_seen)
        assert not any('sample_pipeline' in item for item in sample_pipeline),(repr(sample_pipeline))
        return sample_pipeline

    def _validate_sample_pipelines(self):
        '''Validate the "sample_pipelines" section of config'''
        self.sample_pipelines = self.config.get('sample_pipelines') or {}
        self._validate_type(self.sample_pipelines, 'sample_pipelines', dict)
        for k, v in self.sample_pipelines.items():
            msg = 'sample_pipelines:{}'.format(k)
            self._validate_type(v, msg, (list, tuple))
            for item in v:
                self._validate_type(item, msg + ' - {}'.format(item), dict)
                ok = False
                for k in SAMPLE_PIPELINE_ACTIONS:
                    if k in item:
                        ok = True
                        match = SAMPLE_PIPELINE_ACTIONS.index(k)
                        match = SAMPLE_PIPELINE_ACTIONS[match]
                        if not item[k] in (self.config.get(match) or {}) and k != 'sample_pipeline':
                            raise ElmConfigError('sample_pipeline item {0} is of type '
                                                 '{1} and refers to a key ({2}) that is not in '
                                                 '"{1}" dict of config'.format(item, match, item[k]))
                if not ok:
                    raise ElmConfigError('sample_pipeline item {} does not '
                                         'have a key that is in the set {}'
                                         ''.format(item , SAMPLE_PIPELINE_ACTIONS))
        p = (self.config.get('pipeline') or [])
        self._validate_type(p, 'pipeline', (list, tuple))
        for step in p:
            self._validate_type(step, 'pipeline: {}'.format(step), dict)
            if 'sample_pipeline' in step:
                step['sample_pipeline'] = self._get_sample_pipeline(self, step)

    def _validate_param_grids(self):
        from elm.model_selection.evolve import get_param_grid
        self.param_grids = self.config.get('param_grids') or {}
        self._validate_type(self.param_grids, 'param_grids', dict)
        for k, v in self.param_grids.items():
            self._validate_type(v, 'param_grids:{}'.format(k), dict)
            for k2, v2 in v.items():
                self._validate_type(v2, 'param_grids:{} - {}'.format(k, k2),
                                    (str, list, tuple, dict))
            if not 'control' in v:
                raise ElmConfigError('TODO help on the "control" part of a param_grid')
            steps = [step for step in self.pipeline
                     if step.get('param_grid')]
            for step in steps:
                if not step['param_grid'] in self.param_grids:
                    raise ElmConfigError('Pipeline step {} refers to '
                                         'a param_grid which is not defined '
                                         'in param_grids ({})'.format(step, step['param_grid']))
                get_param_grid(self, step)

    def _validate_change_detection(self):
        raise NotImplementedError()
        self.change_detection = self.config.get('change_detection', {}) or {}
        # TODO fill this in
        self._validate_type(self.change_detection, 'change_detection', dict)
        return True

    def _validate_pipeline_train(self, step):
        '''Validate a "train" step within config's "pipeline"'''
        train = step.get('train')
        if not train in self.train:
            raise ElmConfigError('Pipeline refers to an undefined "train"'
                                   ' key: {}'.format(repr(train)))
        sample_pipeline = step.get('sample_pipeline', []) or []
        step['sample_pipeline'] = sample_pipeline
        step['train'] = train
        return step

    def _validate_pipeline_transform(self, step):
        '''Validate a "train" step within config's "pipeline"'''
        transform = step.get('transform')
        if not transform in self.transform:
            raise ElmConfigError('Pipeline refers to an undefined "transform"'
                                   ' key: {}'.format(repr(transform)))
        sample_pipeline = step.get('sample_pipeline', []) or []
        step['sample_pipeline'] = sample_pipeline
        step['transform'] = transform
        return step

    def _validate_pipeline_predict(self, step):
        '''Validate a "predict" step within config's "pipeline"'''
        predict = step.get('predict')
        if not predict in self.predict:
            raise ElmConfigError('Pipeline refers to an undefined "predict"'
                                   ' key: {}'.format(repr(predict)))
        sample_pipeline = step.get('sample_pipeline', []) or []
        step['sample_pipeline'] = sample_pipeline
        step['predict'] = predict
        return step

    def _validate_pipeline(self):
        '''Validate config's "pipeline"'''
        self.pipeline = pipeline = self.config.get('pipeline', []) or []
        if not pipeline or not isinstance(pipeline, (tuple, list)):
            raise ElmConfigError('Expected a "pipeline" list of action '
                                   'dicts in config but found '
                                   '"pipeline": {}'.format(repr(pipeline)))
        for idx, action in enumerate(pipeline):
            if not action or not isinstance(action, dict):
                raise ElmConfigError('Expected each item in "pipeline" to '
                                       'be a dict but found {}'.format(action))
            cnt = 0
            for key in PIPELINE_ACTIONS:
                if key in action:
                    cnt += 1
                    func = getattr(self, '_validate_pipeline_{}'.format(key))
                    self.pipeline[idx] = func(action)
            if cnt != 1:
                raise ElmConfigError('In each action dictionary of the '
                                       '"pipeline" list, expected exactly one '
                                       'of the following keys: {}'.format(PIPELINE_ACTIONS))

    def validate(self):
        '''Validate all sections of config, calling a function
        _validate_{} where {} is replaced by a section name, like
        "train"'''
        for key, typ in self.config_keys:
            validator = getattr(self, '_validate_{}'.format(key))
            validator()
            assert isinstance(getattr(self, key), typ)

    def __str__(self):
        return yaml.dump(self.config)

