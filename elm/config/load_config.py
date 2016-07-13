import copy
import os

import sklearn.feature_selection as skfeat
import yaml


from elm.config.env import parse_env_vars, ENVIRONMENT_VARS_SPEC
from elm.config.util import (ElmConfigError,
                               import_callable)
from elm.model_selection.util import get_args_kwargs_defaults
from elm.acquire.ladsweb_meta import validate_ladsweb_data_source
from elm.config.defaults import DEFAULTS, CONFIG_KEYS


def file_generator_from_list(some_list, *args, **kwargs):
    yield from iter(some_list)


PIPELINE_ACTIONS = ('download_data_sources',
                    'train',
                    'predict',
                    'change_detection')

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
            if not os.path.exists(config_file_name):
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
        self.config.update(copy.deepcopy(self.raw_config))
        self._update_for_env()
        self.validate()

    def _update_for_env(self):
        import elm.config.dask_settings as elm_dask_settings
        for k, v in parse_env_vars().items():
            if v:
                vars(self)[k] = v
        for str_var in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
            choices = str_var.get('choices', [])
            val = self.config.get(str_var['name'])
            if choices and val not in choices:
                raise ElmConfigError('Expected config key or env '
                                       'var {} to be in '
                                       '{} but got {}'.format(k, choices, val))
            setattr(elm_dask_settings, k, val)
        for int_var in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
            k = int_var['name']
            val = self.config.get(k)
            setattr(elm_dask_settings, k, val)
        elm_dask_settings.SERIAL_EVAL = self.SERIAL_EVAL = self.config['DASK_EXECUTOR'] == 'SERIAL'

    def _validate_custom_callable(self, func_or_not, required, context):
        if func_or_not or (not func_or_not and required):
            if not isinstance(func_or_not, str):
                raise ElmConfigError('In {} expected {} to be a '
                                       'string'.format(func_or_not, context))
        return import_callable(func_or_not, required=required, context=context)

    def _validate_readers(self):
        err_msg = "Expected a 'readers' dictionary in config"
        self.readers = readers = self.config.get('readers')
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

    def _validate_downloads(self):
        self.downloads = self.config.get('downloads', {}) or {}
        if self.downloads and not isinstance(self.downloads, dict):
            raise ElmConfigError('Expected downloads to be a dict but '
                                   'got {}'.format(self.downloads))
        if not self.downloads:
            # TODO raise error if the files don'talready exist
            return
        for k, v in self.downloads.items():
            self._validate_custom_callable(v, True, 'downloads:{}'.format(k))

    def _validate_band_specs(self, band_specs, name):
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
        if not name or not isinstance(name, str):
            raise ElmConfigError('Expected a "name" key in {}'.format(d))
        validate_ladsweb_data_source(ds, name)
        reader = ds.get('reader')
        if not reader in self.readers:
            raise ElmConfigError('Data source config dict {} '
                                   'refers to a "reader" {} that is not defined in '
                                   '"readers"'.format(reader, self.readers))
        download = ds.get('download', '') or ''
        if not download in self.downloads:
            raise ElmConfigError('data_source {} refers to a '
                                   '"download" {} not defined in "downloads"'
                                   ' section'.format(ds, download))
        self._validate_band_specs(ds.get('band_specs'), name)

    def _validate_data_sources(self):
        self.data_sources = self.config.get('data_sources', {}) or {}
        if not self.data_sources or not isinstance(self.data_sources, dict):
            raise ElmConfigError('Expected "data_sources" in config to be a '
                                   'dict. Got: {}'.format(self.data_sources))
        for name, ds in self.data_sources.items():
            self._validate_one_data_source(name, ds)

    def _validate_file_generators(self):
        self.file_generators = self.config.get('file_generators', {}) or {}
        if not isinstance(self.file_generators, dict):
            raise ElmConfigError('Expected file_generators to be a dict, but '
                                   'got {}'.format(self.file_generators))
        for name, file_gen in self.file_generators.items():
            if not name or not isinstance(name, str):
                raise ElmConfigError('Expected "name" key in file_generators {} ')
            self._validate_custom_callable(file_gen, True,
                                           'file_generators:{}'.format(name))

    def _validate_positive_int(self, val, context):
        if not isinstance(val, int) and val:
            raise ElmConfigError('In {} expected {} to be an int'.format(context, val))

    def _validate_one_sampler(self, sampler, name):
        defaults = tuple(self.defaults['samplers'].values())[0]
        if not sampler or not isinstance(sampler, dict):
            raise ElmConfigError('In samplers:{} dict '
                                   'but found {}'.format(name, sampler))
        sampler['n_rows_per_sample'] = sampler.get('n_rows_per_sample', defaults['n_rows_per_sample'])
        sampler['files_per_sample'] = sampler.get('files_per_sample', defaults['files_per_sample'])
        self._validate_positive_int(sampler['n_rows_per_sample'], name)
        self._validate_positive_int(sampler['files_per_sample'], name)
        file_gen = sampler.get('file_generator')
        data_gen = sampler.get('data_generator')
        if data_gen:
            self._validate_custom_callable(data_gen,
                                True,
                                'train:{} data_generator'.format(name))
        if data_gen and file_gen:
            raise ElmConfigError('in samplers:{} - cannot give '
                                   '"data_generator" and '
                                   '"file_generator"'.format(name))
        self._validate_selection_kwargs(sampler, name)

    def _validate_samplers(self):
        self.samplers = self.config.get('samplers', {}) or {}
        if not self.samplers or not isinstance(self.samplers, dict):
            raise ElmConfigError('Invalid "samplers" config entry {} '
                                   '(expected dict)'.format(self.samplers))
        for name, sampler in self.samplers.items():
            self._validate_one_sampler(sampler, name)


    def _validate_poly(self, name, poly):
        return True # TODO this should validate a list entry
                    # in polys list of the config.  E.g. how
                    # is the poly loaded from file?

    def _validate_polys(self):
        self.polys = self.config.get('polys', {}) or {}
        for name, poly in self.polys:
            self._validate_poly(name, poly)

    def _validate_selection_kwargs(self, sampler, name):
        selection_kwargs = sampler.get('selection_kwargs')
        if not selection_kwargs:
            return
        if not isinstance(selection_kwargs, dict):
            raise ElmConfigError('In sampler:{} expected '
                                   '"selection_kwargs" to be '
                                   'a dict'.format(selection_kwargs))
        selection_kwargs['geo_filter'] = selection_kwargs.get('geo_filter', {}) or {}
        for poly_field in ('include_polys', 'exclude_polys'):
            pf = selection_kwargs['geo_filter'].get(poly_field, []) or []
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
                selection_kwargs.pop(filter_name)
        self.samplers[name]['selection_kwargs'] = selection_kwargs


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
                score_func = s.get('score_func')
                no_score_func = ('all', 'sklearn.feature_selection:VarianceThreshold')
                if score_func and score_func not in dir(skfeat) and not selection in no_score_func:
                    self._validate_custom_callable(score_func, True,
                            'feature_selection:{} score_func'.format(k))
                make_scorer_kwargs = s.get('make_scorer_kwargs') or {}
                self._validate_type(make_scorer_kwargs, 'make_scorer_kwargs', dict)

                # TODO is there further validation of make_scorer_kwargs
                # that can be done -
                #    they are passed to sklearn.metrics.make_scorer

                s['make_scorer_kwargs'] = make_scorer_kwargs
            else:
                score_func = None
            s['score_func'] = score_func


            feature_choices = s.get('choices') or 'all'
            self._validate_all_or_type(feature_choices,
                                       'feature_selection:{} choices'.format(k),
                                       (list, tuple))
            s['choices'] = feature_choices
            feature_selection[k] = s
        self.feature_selection = feature_selection

    def _validate_training_funcs(self, name, t):
        if not isinstance(t, dict):
            raise ElmConfigError('In train:{} expected a dict '
                                   'but found {}'.format(name, t))
        training_funcs = (('model_selection_func', False),
                           ('model_init_class', True),
                           ('post_fit_func', False),
                           ('fit_wrapper_func', True),
                           ('get_y_func', False)
                           )

        fit_func = t.get('fit_func', 'fit')
        has_fit_func = False
        for f, required in training_funcs:
            cls_or_func = self._validate_custom_callable(t[f], required,
                                           'train:{} - {}'.format(name, f))
            if f == 'model_init_class':
                model_init_class = cls_or_func
                has_fit_func = hasattr(model_init_class, fit_func)
                if has_fit_func:
                    fargs, fkwargs = get_args_kwargs_defaults(cls_or_func.partial_fit)
                else:
                    fargs, fkwargs = get_args_kwargs_defaults(cls_or_func.fit)
        if not has_fit_func:
            raise ElmConfigError('{} has no {} method (fit_func)'.format(model_init_class, fit_func))
        requires_y = any(x.lower() == 'y' for x in fargs)
        return has_fit_func, requires_y


    def _validate_one_train_entry(self, name, t):
        has_fit_func, requires_y = self._validate_training_funcs(name, t)
        if requires_y:
            self._validate_custom_callable(t.get('get_y_func'), True, 'train:get_y_func (required with {})'.format(repr(t.get('model_init_class'))))
        kwargs_fields = tuple(k for k in t if k.endswith('_kwargs'))
        for k in kwargs_fields:
            self._validate_type(t[k], 'train:{}'.format(k), dict)

        for f in ('saved_ensemble_size', 'n_generations', 'ensemble_size'):
            self._validate_positive_int(t['ensemble_kwargs'].get(f), f)

        fit_kwargs = t.get('fit_kwargs', {}) or {}
        t['fit_kwargs']['n_batches'] = fit_kwargs['n_batches'] = fit_kwargs.get('n_batches', 1)
        self._validate_type(fit_kwargs['n_batches'], 'fit_kwargs:n_batches', int)
        if fit_kwargs['n_batches'] > 1 and not has_fit_func:
            raise ElmConfigError('With fit_kwargs - n_batches {} '
                                   '(>1) the model must have a '
                                   '"partial_fit" method.'.format(fit_kwargs['n_batches']))
        sampler = t.get('sampler', '')
        if sampler and sampler not in self.samplers:
            raise ElmConfigError('train dict at key {} refers '
                                   'to a sampler {} that is '
                                   'not defined in '
                                   '"samplers"'.format(name, repr(sampler)))
        data_source = t.get('data_source')
        if not data_source in self.data_sources:
            raise ElmConfigError('train dict at key {} refers '
                                   'to a data_source {} that is '
                                   'not defined in '
                                   '"data_sources"'.format(name, repr(data_source)))
        if (data_source and sampler) or (not data_source and not sampler):
            raise ElmConfigError('Conflicting definition of both '
                                   '"data_source" and "sampler" in '
                                   '"train".  Provide one of "data_source": '
                                   '<key from "data_sources", "sampler" '
                                   'key from "samplers"')
        output_tag = t.get('output_tag')
        self._validate_type(output_tag, 'train:output_tag', str)
        band_specs = self.data_sources[data_source]['band_specs']
        t['band_names'] = [x[-1] for x in band_specs]
        keep_columns = t.get('keep_columns') or []
        self._validate_type(keep_columns, 'keep_columns', (tuple, list))
        t['keep_columns'] = keep_columns
        feature_selection = t.get('feature_selection') or 'select_all'
        self._validate_type(feature_selection, 'feature_selection', str)
        if not feature_selection in self.feature_selection:
            raise ElmConfigError('In train:{} expected '
                                   'feature_selection:{} to be a '
                                   'key in feature_selection:'
                                   '{}'.format(k, feature_selection, self.feature_selection))
        self.config['train'][name] = self.train[name] = t

    def _validate_train(self):
        self.train = self.config.get('train', {}) or {}
        for name, t in self.train.items():
            self._validate_one_train_entry(name, t)

    def _validate_predict(self):
        self.predict = self.config.get('predict', {}) or {}
        return True # TODO validate predict config

    def _validate_change_detection(self):
        self.change_detection = self.config.get('change_detection', {}) or {}
        # TODO fill this in
        self._validate_type(self.change_detection, 'change_detection', dict)
        return True

    def _validate_pipeline_download_data_sources(self, step):
        # TODO make sure that the dataset can be downloaded or exists locally
        download_data_sources = step.get('download_data_sources', []) or []

        if isinstance(download_data_sources, str):
            download_data_sources = [download_data_sources]
        self._validate_type(download_data_sources, 'download_data_sources', (list, tuple))
        download_data_sources = download_data_sources
        step['download_data_sources'] = download_data_sources
        return step

    def _validate_pipeline_sample_pipeline(self, sample_pipeline, predict_or_train, options, step):
        # TODO validate operations such as resampling and aggregation
        # after a sample is taken

        return step

    def _validate_pipeline_train(self, step):
        train = step.get('train')
        if not train in self.train:
            raise ElmConfigError('Pipeline refers to an undefined "train"'
                                   ' key: {}'.format(repr(train)))
        sample_pipeline = step.get('sample_pipeline', []) or []
        step = self._validate_pipeline_sample_pipeline(sample_pipeline, 'train',
                                                      self.train[train], step)
        step['sample_pipeline'] = sample_pipeline
        step['train'] = train
        return step

    def _validate_pipeline_predict(self, step):
        predict = step.get('predict')
        if not predict in self.predict:
            raise ElmConfigError('Pipeline refers to an undefined "predict"'
                                   ' key: {}'.format(repr(predict)))
        sample_pipeline = step.get('sample_pipeline', []) or []
        step = self._validate_pipeline_sample_pipeline(sample_pipeline, 'predict', self.predict[predict], step)
        step['sample_pipeline'] = sample_pipeline
        step['predict'] = predict
        return step

    def _validate_pipeline(self):
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
        for key, typ in self.config_keys:
            validator = getattr(self, '_validate_{}'.format(key))
            validator()
            assert isinstance(getattr(self, key), typ)

