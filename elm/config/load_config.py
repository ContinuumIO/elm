'''
This module is used by the command line interface of elm
to parse yaml ensemble or evolutionary algorithm configs.
'''
from collections import Iterable, Sequence
import copy
import logging
import numbers
import os
import traceback

import attr
import numpy as np
import sklearn.feature_selection as skfeat
import sklearn.preprocessing as skpre
import yaml


from elm.config.env import parse_env_vars, ENVIRONMENT_VARS_SPEC
from elm.config.util import (ElmConfigError,
                               import_callable)
from elm.model_selection.util import get_args_kwargs_defaults
from elm.config.config_info import CONFIG_KEYS

logger = logging.getLogger(__name__)


SAMPLE_PIPELINE_ACTIONS = ('transform',
                           'feature_selection',
                           'sklearn_preprocessing',
                           'random_sample',) # others too from change_coords
REQUIRES_METHOD = ('train', 'predict', )
class ConfigParser(object):
    expected_ensemble_kwargs = ('init_ensemble_size',
                                'ngen',
                                'partial_fit_batches',
                                'saved_ensemble_size',)
    # The list below reflects the order
    # in which the keys of the config
    # are validated. (the _validate_* private
    # methods are order sensitive.)
    config_keys = CONFIG_KEYS
    def __init__(self, config_file_name=None, config=None, cmd_args=None):
        '''Parses an elm config dictionary or yaml file

        Params:
            config_file_name: file name of a yaml elm config
            config:           elm config dict

        Returns:
            ConfigParser instance
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
        self.config = copy.deepcopy(self.raw_config)
        self._update_for_env()
        self._update_for_cmd_args(cmd_args=cmd_args)
        self.validate()

    def _update_for_cmd_args(self, cmd_args=None):
        self.cmd_args = {} if not cmd_args else dict(vars(cmd_args))
        for k, v in self.cmd_args.items():
            if k not in ('config',):
                if v:
                    if k in self.expected_ensemble_kwargs:
                        replace = self.config.get('ensembles', {})
                        self._validate_type(replace, 'ensembles', dict)
                        for k2, v2 in replace.items():
                            replace[k2][k] = v
                        self.config['ensembles'] = replace
                    else:
                        setattr(self, k.upper().replace('-', '_'), v)

    def _interpolate_env_vars(self, env):
        '''Replace config items like env:ELM_TRAIN_PATH with relevant env var'''
        updates = {}
        for k, v in self.config.get('data_sources', {}).items():
            if k not in updates:
                updates[k] = {}
            for key in ('X', 'y', 'sample_weight'):
                if key in v:
                    updates[k] = self.config['data_sources'][k].pop(key)
        config_str = yaml.dump(self.config)
        for env_var in (ENVIRONMENT_VARS_SPEC['str_fields_specs'] +
                        ENVIRONMENT_VARS_SPEC['int_fields_specs']):
            env_str = 'env:{}'.format(env_var['name'])
            if env_str in config_str:
                config_str = config_str.replace(env_str,
                                                env[env_var['name']])
        self.config = yaml.load(config_str)
        for key in updates:
            self.config['data_sources'][key].update(updates[key])

    def _update_for_env(self):
        '''Update the config based on environment vars'''
        self._env = parse_env_vars()
        self._interpolate_env_vars(self._env)
        updates = {}
        for str_var in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
            choices = str_var.get('choices', [])
            val = self._env.get(str_var['name'], None)
            if choices and val not in choices and str_var.get('required'):
                raise ElmConfigError('Expected config key or env '
                                       'var {} to be in '
                                       '{} but got {}'.format(str_var['name'], choices, val))

            if val:
                updates[str_var['name']] = val

        for int_var in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
            val = getattr(self, int_var['name'], None)
            if val:
                updates[int_var['name']] =  int(val)

        for k, v in updates.items():
            self.config[k] = v
            setattr(self, k, v)

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
        if not isinstance(readers, dict):
            raise ElmConfigError(err_msg)
        for k, v in readers.items():
            if not v or not isinstance(v, dict):
                raise ElmConfigError('In readers:{} expected {} to be '
                                       'a dict'.format(k, v))

            load = v.get('load_array')
            meta = v.get('load_meta')
            self._validate_custom_callable(load, True, 'readers:{} load_array'.format(k))
            self._validate_custom_callable(meta, True, 'readers:{} load_meta'.format(k))
            self.readers[k] = v


    def _validate_band_specs(self, band_specs, name):
        '''Validate "band_specs"'''
        from elm.readers.util import BandSpec
        if all(isinstance(bs, BandSpec) for bs in band_specs):
            return band_specs
        if not band_specs or not isinstance(band_specs, (tuple, list)):
            raise ElmConfigError('data_sources:{} gave band_specs which are not a '
                                   'list {}'.format(name, band_specs))
        if not all(isinstance(bs, (dict, str)) for bs in band_specs):
            raise ElmConfigError('Expected "band_specs" to be a list of dicts or list of strings')

        new_band_specs = []
        for band_spec in band_specs:
            if isinstance(band_spec, str):
                new_band_specs.append(BandSpec(**{'search_key': 'sub_dataset_name',
                                                'search_value': band_spec,
                                                'name': band_spec}))
            elif not all(k.name in band_spec for k in attr.fields(BandSpec)
                       if not k.default == attr.NOTHING):
                raise ElmConfigError("band_spec {} did not have keys: {}".format(band_spec, attr.fields(BandSpec)))
            else:
                new_band_specs.append(BandSpec(**band_spec))
        return new_band_specs

    def _validate_one_data_source(self, name, ds):
        '''Validate one data source within "data_sources"
        section of config'''

        if not name or not isinstance(name, str):
            raise ElmConfigError('Expected a "name" key in {}'.format(d))
        sampler = ds.get('sampler')
        if sampler:
            self._validate_custom_callable(sampler,
                                    True,
                                    'train:{} sampler'.format(name))
        if 'band_specs' in ds:
            ds['band_specs'] = self._validate_band_specs(ds.get('band_specs'), name)
        reader = ds.get('reader')
        reader_words = ('hdf4', 'hdf5', 'tif', 'netcdf')
        if not reader:
            ds['reader'] = None
        if reader and not reader in self.readers and reader not in reader_words:
            raise ElmConfigError('Data source config dict {} '
                                 'refers to a "reader" {} that is not defined in '
                                 '"readers" or in the tuple {}'.format(reader, self.readers, reader_words))
        s = ds.get('args_list')
        if s and isinstance(s, (list, tuple)):
            def anon(*args, **kwargs):
                return s
            ds['args_list'] = anon
        elif s:
            try:
                ds['args_list'] = import_callable(s)
            except Exception as e:
                raise ElmConfigError('data_source:{} uses a args_list {} that '
                                     'is neither importable nor in '
                                     '"args_list" dict'.format(name, s))
        self._validate_selection_kwargs(ds, name)
        self.data_sources[name] = ds

    def _validate_data_sources(self):
        '''Validate all "data_sources" of config'''
        self.data_sources = self.config.get('data_sources', {}) or {}
        if not self.data_sources or not isinstance(self.data_sources, dict):
            raise ElmConfigError('Expected "data_sources" in config to be a '
                                   'dict. Got: {}'.format(self.data_sources))
        for name, ds in self.data_sources.items():
            self._validate_one_data_source(name, ds)

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
        selection_kwargs = data_source # TODO renaming needed further
        if not selection_kwargs:
            return
        selection_kwargs['geo_filters'] = selection_kwargs.get('geo_filters', {}) or {}
        for poly_field in ('include_polys', 'exclude_polys'):
            pf = selection_kwargs['geo_filters'].get(poly_field, []) or []
            for item in pf:
                if not item in self.polys:
                    raise ElmConfigError('config\'s data_source dict {} '
                                           '"include_polys" or "exclude_poly" '
                                           'must refer to a list of keys from config\'s '
                                           '"polys"'.format(self.selection_kwargs))
        for filter_name in ('data_filter', 'metadata_filter', 'filename_filter'):
            f = selection_kwargs.get(filter_name, {})
            if f:
                self._validate_custom_callable(f, True,
                                               'data_source:{} - {}'.format(name, filter_name))
            else:
                selection_kwargs[filter_name] = None
        self.data_sources[name] = selection_kwargs


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
        feature_selection = self.feature_selection = self.config.get('feature_selection') or {}
        if not feature_selection:
            return True
        self._validate_type(feature_selection, 'feature_selection', dict)
        for k, s in feature_selection.items():
            self._validate_type(k, 'feature_selection:{}'.format(k), str)
            self._validate_type(s, 'feature_selection:{}'.format(s), dict)
            selection = s.get('method')
            if selection and selection not in dir(skfeat):
                raise ValueError('{} is not in dir(sklearn.feature_selection)'.format(selection))
            feature_selection[k] = s
        self.feature_selection = feature_selection

    def _validate_one_model_scoring(self, key, value):
        '''Validate one model_scoring key-value (one scoring configuration)'''
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
        no_selection = not sel
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
        for idx, step1 in enumerate(pipeline):
            for step in step1['steps']:
                self._validate_type(step, 'pipeline step:{}'.format(idx), dict)
                if step.get('transform') == transform_name:
                    return True
        return False

    def _validate_one_train_entry(self, name, t):
        '''Validate one dict within "train" or "transform" section of config'''
        has_fit_func, requires_y, no_selection = self._validate_train_or_transform_funcs(name, t)
        kwargs_fields = tuple(k for k in t if k.endswith('_kwargs'))
        for k in kwargs_fields:
            self._validate_type(t[k], 'train:{}'.format(k), dict)
        mod = t.get('model_selection')

        if mod:
            t['model_selection'] = import_callable(mod)
        if t.get('sort_fitness') is not None:
            self._validate_custom_callable(t.get('sort_fitness'),
                                           True,
                                           'train:{} (sort_fitness)'.format(repr(t.get('sort_fitness'))))
        ms = t.get('model_scoring')
        if ms is not None:
            self._validate_type(ms, 'train:{} (model_scoring)'.format(name),
                                (str, numbers.Number, tuple))
            if not ms in self.model_scoring:
                raise ElmConfigError('train:{}\'s model_scoring: {} is not a '
                                     'key in config\'s model_scoring '
                                     'dict'.format(name, ms))
        self.config['train'][name] = t

    def _validate_train(self):
        '''Validate the "train" or "transform" section of config'''

        self.train = self.config.get('train') or {}
        for name, t in self.train.items():
            self._validate_type(self.config['run'], 'run', (list, tuple))
            self._validate_one_train_entry(name, t)
            for step1 in self.config['run']:
                self._validate_type(step1, 'pipeline: {}'.format(step1), dict)

    def _validate_transform(self):
        '''Validate the "transform" section of config'''
        self.transform = self.config.get('transform') or {}
        for name, t in self.transform.items():
            self._validate_type(t, 'transform:{}'.format(name), dict)
            if not 'model_init_class' in t:
                raise ElmConfigError('Expected {} to have at least "model_init_class" and possibly "model_init_kwargs" in each transform dict')

    def _validate_sklearn_preprocessing(self):
        '''Validate "sklearn_preprocessing" dict in config'''
        self.sklearn_preprocessing = self.config.get('sklearn_preprocessing') or {}
        self._validate_type(self.sklearn_preprocessing, 'sklearn_preprocessing', dict)
        for k, v in self.sklearn_preprocessing.items():
            self._validate_type(v, 'sklearn_preprocessing:{}'.format(k), dict)
            if v.get('method') in dir(skpre) or callable(v.get('method')):
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
        for f in ('saved_ensemble_size', 'ngen',
                  'init_ensemble_size', 'partial_fit_batches'):
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

    def _validate_pipelines(self):
        '''Validate the "pipelines" section of config'''
        from elm.sample_util.change_coords import CHANGE_COORDS_ACTIONS
        self.pipelines = self.config.get('pipelines') or {}
        self._validate_type(self.pipelines, 'pipelines', dict)
        for k, v in self.pipelines.items():
            msg = 'pipelines:{}'.format(k)
            self._validate_type(v, msg, (list, tuple))
            for item in v:
                self._validate_type(item, msg + ' - {}'.format(item), dict)
                ok = False
                ok_words = SAMPLE_PIPELINE_ACTIONS + CHANGE_COORDS_ACTIONS
                for k in ok_words:
                    if k in item:
                        ok = True
                        match = ok_words.index(k)
                        match = ok_words[match]
                        if match in CHANGE_COORDS_ACTIONS:
                            val = item[match]
                            if match == 'change_coords':
                                self._validate_custom_callable(item[match], True,
                                        'pipelines:{} - {} ({})'.format(k, match, item[match]))
                            if match == 'flatten':
                                self._validate_type(val, 'flatten:{}'.format(val), str)
                                if not val == 'C':
                                    raise ElmConfigError('Expected flatten:{} to be "C" for flatten order (more options may be there over time)'.format(val))
                        elif not item[k] in (self.config.get(match) or {}) and k != 'pipeline':
                            raise ElmConfigError('pipeline item {0} is of type '
                                                 '{1} and refers to a key ({2}) that is not in '
                                                 '"{1}" dict of config'.format(item, match, item[k]))
                if not ok:
                    raise ElmConfigError('pipeline item {} does not '
                                         'have a key that is in the set {}'
                                         ''.format(item , SAMPLE_PIPELINE_ACTIONS))
        p = (self.config.get('pipeline') or [])
        self._validate_type(p, 'pipeline', (list, tuple))
        for step in p:
            self._validate_type(step, 'pipeline: {}'.format(step), dict)


    def _validate_param_grids(self):
        from elm.model_selection.evolve import get_param_grid
        self.param_grids = self.config.get('param_grids') or {}
        self._validate_type(self.param_grids, 'param_grids', dict)
        for k, v in self.param_grids.items():
            for step in self.run:
                if not step['param_grid'] in self.param_grids:
                    raise ElmConfigError('Pipeline step {} refers to '
                                         'a param_grid which is not defined '
                                         'in param_grids ({})'.format(step, step['param_grid']))
                get_param_grid(self, step)

    def _validate_pipeline_train(self, step):
        '''Validate a "train" step within config's "pipeline"'''
        train = step.get('train')
        if not train in self.train:
            raise ElmConfigError('Pipeline refers to an undefined "train"'
                                   ' key: {}'.format(repr(train)))
        step['train'] = train
        return step

    def _validate_pipeline_transform(self, step):
        '''Validate a "train" step within config's "pipeline"'''
        transform = step.get('transform')
        if not transform in self.transform:
            raise ElmConfigError('Pipeline refers to an undefined "transform"'
                                   ' key: {}'.format(repr(transform)))
        step['transform'] = transform
        return step

    def _validate_pipeline_predict(self, step):
        '''Validate a "predict" step within config's "pipeline"'''
        return step

    def _validate_run(self):
        '''Validate config's "pipeline"'''

        self.run = run = self.config.get('run', []) or []
        if not run or not isinstance(run, (tuple, list)):
            raise ElmConfigError('Expected a "pipeline" list of action '
                                   'dicts in config but found '
                                   '"pipeline": {}'.format(repr(run)))
        for idx, action in enumerate(run):
            if not action or not isinstance(action, dict):
                raise ElmConfigError('Expected each item in "pipeline" to '
                                       'be a dict but found {}'.format(action))
            if not 'pipeline' in action:
                raise ElmConfigError('Expected a pipeline key in pipeline action {}'.format(action))
            pipeline = action['pipeline']
            if not isinstance(pipeline, (list, tuple)):
                pipeline = self.pipelines[pipeline]
                action['pipeline'] = pipeline
            data_source = action.get('data_source') or ''
            if not data_source in self.data_sources:
                raise ElmConfigError('Expected a data_source key in pipeline action {}'.format(action))

    def validate(self):
        '''Validate all sections of config, calling a function
        _validate_{} where {} is replaced by a section name, like
        "train"'''
        for key, typ in self.config_keys:
            validator = getattr(self, '_validate_{}'.format(key))
            validator()


    def __str__(self):
        return yaml.dump({k: getattr(self, k, self.config.get(k))
                          for k, _ in self.config_keys})

