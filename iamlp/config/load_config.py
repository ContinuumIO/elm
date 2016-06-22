import copy

import yaml

from iamlp.config.defaults import DEFAULTS
from iamlp.config.env import parse_env_vars, ENVIRONMENT_VARS_SPEC
from iamlp.config.util import IAMLPConfigError
from iamlp.acquire.ladsweb_meta import validate_ladsweb_data_source
import iamlp.config.dask_settings as iamlp_dask_settings

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
    config_keys = [('readers',  dict),
                   ('downloads', dict),
                    ('data_sources', dict),
                    ('file_generators', dict),
                    ('file_lists', dict),
                    ('samplers', dict),
                    ('polys': dict),
                    ('prefilters', dict),
                    ('resamplers', dict),
                    ('aggregations', dict),
                    ('masks', dict),
                    ('add_features', dict),
                    ('train', dict),
                    ('predict', dict),
                    ('pipeline', list),
        ]


    def __init__(self, config_file_name):
        if not os.path.exists(config_file_name):
            raise IAMLPConfigError('config_file_name {} does not '
                                   'exist'.format(config_file_name))
        with open(config_file_name) as f:
            self.raw_config = yaml.safe_load(f.read())
        self.config = copy.deepcopy(DEFAULTS)
        self.config.update(copy.deepcopy(self.raw_config))
        self._update_for_env()
        self.validate()

    def _update_for_env(self):
        for k, v in parse_env_vars().items():
            if v:
                self.config[k] = v
        for str_var in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
            choices = str_var.get('choices', [])
            val = self.config.get(str_var['name'])
            if choices and val not in choices:
                raise IAMLPConfigError('Expected config key or env '
                                       'var {} to be in '
                                       '{} but got {}'.format(k, choices, val))
            setattr(iamlp_dask_settings, k, val)
        for int_var in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
            k = int_var['name']
            val = self.config.get(k)
            setattr(iamlp_dask_settings, k, val)

    def _validate_custom_callable(self, some_dict, label):
        err_msg = 'In {}: {} expected keys of "module" and "callable"'
        if not 'module' in some_dict:
            raise IAMLPConfigError(err_msg)
        if not 'callable' in some_dict:
            raise IAMLPConfigError(err_msg)
        try:
            mod = __import__(some_dict['module'])
            func = getattr(mod, some_dict['callable'], None)
            if not callable(func):
                raise IAMLPConfigError('In {}, callable given '
                                       '{} could be imported but '
                                       'was not actually a '
                                       'Python callable'.format(some_dict, some_dict['callable']))
            return mod, func
        except Exception as e:
            raise IAMLPConfigError('There was an error importing from {}'.format(some_dict))

    def _validate_readers(self):
        err_msg = "Expected a readers dictionary in config"
        self.readers = readers = self.config.get('readers')

        if not readers or not isinstance(readers, dict):
            raise IAMLPConfigError(err_msg)
        for k,v in readers.items():
            load = v.get('load')
            bounds = v.get('bounds')
            v['load']['module'], v['load']['callable'] = self._validate_custom_callable(load, 'readers:{} load'.format(k))
            v['bounds']['module'], v['bounds']['callable'] = self._validate_custom_callable(bounds, 'readers:{} bounds'.format(k))
            self.readers[k] = v

    def _validate_downloads(self):
        self.downloads = self.config.get('downloads', {}) or {}
        if not self.downloads:
            # TODO raise error if the files don'talready exist
            return
        self._validate_custom_callable(self.downloads)
        name = self.downloads.get('name', '')
        if not name or not isinstance(name, str):
            raise IAMLPConfigError('Error in downloads '
                                   'section: Expected a "name" '
                                   'key in {}'.format(self.downloads))

    def _validate_band_specs(self, band_specs, name):
        if not band_specs or isinstance(band_specs, list):
            raise IAMLPConfigError('data_sources:{} gave band_specs which are not a '
                                   'list {}'.format(name, band_specs))
        for band_spec in band_specs:
            if not isinstance(band_spec, (tuple, list)) or not len(band_spec) == 3 and or not all(isinstance(b, str) for b in band_spec):
                raise IAMLPConfigError("band_spec {} in data_sources:{} "
                                       "did not have 3 strings "
                                       "(metadata key search phrase,"
                                       "metadata value search phrase, "
                                       "band name)".format(band_spec, name))

    def _validate_one_data_source(self, name, ds):
        if not name or not isinstance(name, str):
            raise IAMLPConfigError('Expected a "name" key in {}'.format(d))
        validate_ladsweb_data_source(ds, name)
        reader = ds.get('reader')
        if not reader in self.readers:
            raise IAMLPConfigError('Data source config dict {} '
                                   'refers to a "reader" {} that is not defined in '
                                   '"readers"'.format(reader, self.readers))
        download = ds.get('download', '') or ''
        if not download in self.downloads:
            raise IAMLPConfigError('data_source {} refers to a '
                                   '"download" {} not defined in "downloads"'
                                   ' section'.format(data_source, download))
        self._validate_band_specs(ds.get('band_specs'), name)

    def _validate_data_sources(self):
        self.data_sources = self.config.get('data_sources', {}) or {}
        if not self.data_sources or not isinstance(self.data_sources, dict):
            raise IAMLPConfigError('Expected "data_sources" in config to be a '
                                   'dict. Got: {}'.format(self.data_sources))
        for name, ds in self.data_sources.items():
            self._validate_one_data_source(name, ds)

    def _validate_file_generators(self):
        self.file_generators = self.config.get('file_generators', {}) or {}
        if not isinstance(self.file_generators, dict):
            raise IAMLPConfigError('Expected file_generators to be a dict, but '
                                   'got {}'.format(self.file_generators))
        for name, file_gen in self.file_generators.items():
            if not name or not isinstance(name, str):
                raise IAMLPConfigError('Expected "name" key in file_generators {} ')
            self._validate_custom_callable(file_gen, name)

    def _validate_file_lists(self):
        self.file_lists = self.config.get('file_lists', {}) or {}
        if not isinstance(self.file_lists, dict):
            raise IAMLPConfigError('Expected file_lists {} to be a dict'.format(self.file_lists))
        for name, file_list in self.file_lists.items():
            self.file_generators[name] = partial(file_generator_from_list, file_list)

    def _validate_positive_int(self, val, context):
        if not isinstance(val, int) and val:
            raise IAMLPConfigError('In {} expected {} to be an int'.format(context, val))

    def _validate_one_sampler(self, sampler, name):
        self._validate_custom_callable(sampler, name)
        sampler['n_rows_per_sample'] = sampler.get('n_rows_per_sample', DEFAULTS['n_rows_per_sample'])
        sampler['files_per_sample'] = sampler.get('files_per_sample', DEFAULTS['files_per_sample'])
        self._validate_positive_int(sampler['n_rows_per_sample'], name)
        self._validate_positive_int(sampler['files_per_sample'], name)
        file_gen = sampler.get('file_generator')
        file_list = sampler.get('file_list')
        if (not file_gen and not file_list) or (file_gen and file_list):
            raise IAMLPConfigError('In sampler {} expected either (and not both of) '
                                   '"file_generator": "some_name" or '
                                   '"file_list": "some_name"'.format(sampler))
        file_arg = file_gen or file_list
        if not file_arg in self.file_generators:
            raise IAMLPConfigError('In sampler {} expected a "file_list" '
                                   'or "file_gen" name that is a key in '
                                   '"file_lists" or "file_generators", '
                                   'respectively.  Got {}'.format(sampler, file_arg))
    def _validate_samplers(self):
        self.samplers = self.config.get('samplers', {}) or {}
        if not self.samplers or isinstance(self.samplers, dict):
            raise IAMLPConfigError('Invalid "samplers" config entry {} '
                                   '(expected dict)'.format(self.samplers))
        for name, sampler in self.samplers.items():
            self._validate_one_sampler(sampler, name)

    def _validate_poly(self, name, poly):
        if not isinstance(name, str) and name:
            raise IAMLPConfigError('In poly {} expected a "name" to be used in '
                                   '"prefilters"'.format(poly))
        return True # TODO this should validate a list entry
                    # in polys list of the config.  E.g. how
                    # is the poly loaded from file?

    def _validate_polys(self):
        self.polys = self.config.get('polys', {}) or {}
        for name, poly in self.polys:
            self._validate_poly(name, poly)

    def _validate_prefilter(self, name, prefilter):
        prefilter['geo_filter'] = prefilter.get('geo_filter', {}) or {}
        for poly_field in ('include_polys', 'exclude_polys'):
            pf = prefilter['geo_filter'].get(poly_field, []) or []
            for item in pf:
                if not item in self.polys:
                    raise IAMLPConfigError('config\'s prefilters dict {} '
                                           '"include_polys" or "exclude_poly" '
                                           'must refer to a list of keys from config\'s '
                                           '"polys"'.format(self.prefilters))
        for filter_name in ('data_filter', 'metadata_filter', 'filename_filter'):
            f = prefilter.get(filter_name, {})
            if f:
                prefilter[f]['module'], prefilter[f]['callable'] = self._validate_custom_callable(f, filter_name)
        self.prefilters[name] = prefilter

    def _validate_prefilters(self):
        self.prefilters = self.config.get('prefilters', {}) or {}
        for name, prefilter in self.prefilters.items():
            self._validate_prefilter(name, prefilter)

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

    def _validate_one_train_entry(self, name, t):
        training_mod_fields = ('model_selector', 'model_init', 'post_fit_func')
        for f in training_mod_fields:
            t[f]['module'], t[f]['callable'] = self._validate_custom_callable(t[f], f)
        t['model_init_kwargs'] = t.get('model_init_kwargs', {}) or {}
        ensemble_kwargs = t.get('ensemble_kwargs', {}) or {}
        if not isinstance(ensemble_kwargs, dict):
            raise IAMLPConfigError('Expected train:{} to have '
                                   '"ensemble_kwargs" as a '
                                   'dict. Got {}'.format(name, ensemble_kwargs))
        for f in ('no_shuffle', 'n_generations'):
            self._validate_positive_int(ensemble_kwargs.get(f), f)

        partial_fit_kwargs = t.get('partial_fit_kwargs', {}) or {}
        if not isinstance(partial_fit_kwargs, dict):
            raise IAMLPConfigError('Expected "partial_fit_kwargs" in train:{} '
                                   'to be a dict but got {}'.format(name, partial_fit_kwargs))
        if not isinstance(t['model_init_kwargs'], dict):
            raise IAMLPConfigError('Expected train:{}\'s '
                                   'model_init_kwargs to be a '
                                   'dict but got {}'.format(name, t['model_init_kwargs']))
        sampler = t.get('sampler', '')
        if not sampler in self.samplers:
            raise IAMLPConfigError('train dict at key {} refers '
                                   'to a sampler {} that is '
                                   'not defined in '
                                   '"samplers"'.format(name, repr(sampler)))
        data_source = t.get('data_source')
        if not data_source in self.data_sources:
            raise IAMLPConfigError('train dict at key {} refers '
                                   'to a data_source {} that is '
                                   'not defined in '
                                   '"data_sources"'.format(name, repr(data_source)))
        output_tag = t.get('output_tag')
        if not output_tag or not isinstance(output_tag, str):
            raise IAMLPConfigError('Expected an "output_tag" key with string '
                                   'value in "train": {}'.format(name))
        band_specs = self.data_sources[data_source]['band_specs']
        t['band_names'] = [x[-1] for x in band_specs]
        # validating ml_features may be tough, TODO?
        t['ml_features'] = t.get('ml_features', []) or []
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
        return True

    def _validate_pipeline_require(self, step):
        # TODO make sure that the dataset can be downloaded or exists locally
        download_data_sources = step.get('download_data_sources', [])
        return True

    def _validate_pipeline_train(self, step):
        train = step.get('train')
        if not train in self.train:
            raise IAMLPConfigError('Pipeline refers to an undefined "train"'
                                   ' key: {}'.format(repr(train)))
        on_each_sample = step.get('on_each_sample', []) or []
        self._validate_on_each_sample(on_each_sample, 'train', self.train[train])

    def _validate_pipeline_predict(self, step):
        predict = step.get('predict')
        if not predict in self.predict:
            raise IAMLPConfigError('Pipeline refers to an undefined "predict"'
                                   ' key: {}'.format(repr(predict)))
        on_each_sample = step.get('on_each_sample', []) or []
        self._validate_on_each_sample(on_each_sample, 'predict', self.predict[predict])

    def _validate_pipeline(self):
        pipeline = self.config.get('pipeline', []) or []
        if not pipeline or not isinstance(pipeline, (tuple, list)):
            raise IAMLPConfigError('Expected a "pipeline" list of action '
                                   'dicts in config but found '
                                   '"pipeline": {}'.format(repr(pipeline)))
        for action in pipeline:
            cnt = 0
            for key in PIPELINE_ACTIONS:
                if key in action:
                    cnt += 1
                    func = getattr(self, '_validate_pipeline_{}'.format(key))
                    func(action)
            if cnt != 1:
                raise IAMLPConfigError('In each action dictionary of the '
                                       '"pipeline" list, expected exactly one '
                                       'of the following keys: {}'.format(PIPELINE_ACTIONS))

    def validate(self):
        for key, typ in self.config_keys:
            validator = getattr(self, '_validate_{}'.format(key))
            validator()
            assert isinstance(getattr(self, key), typ)

