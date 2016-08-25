import copy
import logging
from pprint import pformat

import numpy as np
import xarray as xr
from sklearn.utils import check_array as _check_array

from elm.config import import_callable, ConfigParser
from elm.model_selection.util import get_args_kwargs_defaults
from elm.model_selection.util import filter_kwargs_to_func
from elm.readers import (ElmStore, flatten as _flatten)
from elm.sample_util.change_coords import (change_coords_action,
                                           CHANGE_COORDS_ACTIONS)
from elm.sample_util.filename_selection import get_generated_args
from elm.readers.util import row_col_to_xy

logger = logging.getLogger(__name__)

_SAMPLE_PIPELINE_SPECS = {}


def check_action_data(action_data):
    '''Check that each action in action_data from get_sample_pipeline_action_data
    is a tuple of (func, args, kwargs)

    Params:
        action_data: list of (func, args, kwargs) tuples
    Returns True or raises ValueError

    '''
    if not isinstance(action_data, (list, tuple)):
        raise ValueError("Expected action_data to run_sample_pipeline to be a list. "
                        "Got {}".format(type(action_data)))
    for item in action_data:
        if (not isinstance(item, tuple) or len(item) != 4):
            raise ValueError('Expected each item in action_data to '
                             'be a tuple of 4 items. Got {}'.format(pformat(item)))
        sample_pipeline_step, func, args, kwargs = item
        func = import_callable(func, True, func)
        if not callable(func):
            raise ValueError('Expected first item in an action_data element '
                             'to be a callable, but got {}'.format(pformat(func)))
        if not isinstance(args, (tuple, list)):
            raise ValueError('Expected second item in an action_data element '
                             'to be a tuple or list (args to {}). Got {}'.format(pformat(func), pformat(args)))
        if not isinstance(kwargs, dict):
            raise ValueError('Expected third item in an action_data element '
                             'to be a dict (kwargs to {}).  Got {}'.format(pformat(func), pformat(kwargs)))
    return True


def run_sample_pipeline(action_data, sample=None, transform_model=None):
    '''Given action_data as a list of (func, args, kwargs) tuples,
    run each function passing args and kwargs to it
    Params:
        action_data:     list from get_sample_pipeline_action_data typically
        sample:          None if the sample is not already taken
        transform_model: An example:
                             [('tag_0', PCA(.....))]
    '''
    sample_y, sample_weight = None, None
    check_action_data(action_data)
    if sample is not None:
        if len(action_data) == 1:
            return sample
        start_idx = 1
    else:
        start_idx = 0
    for action in action_data[start_idx:]:
        sample_pipeline_step, func_str, args, kwargs = action
        logger.debug('On sample_pipeline step: {}'.format(sample_pipeline_step))
        if func_str.endswith('transform_sample_pipeline_step'):
            logger.debug('transform sample_pipeline step')
            samp_pipeline_step = args[0]
            args = tuple(args) + (transform_model,)
        func = import_callable(func_str, True, func_str)
        if sample is None:
            logger.debug('sample create sample_pipeline step')
            required_args, default_kwargs, var_keyword = get_args_kwargs_defaults(func)
            if not var_keyword:
                kwargs = filter_kwargs_to_func(func, kwargs)
            sample = func(*args, **kwargs)
        elif 'get_y' in args:
            logger.debug('get_y sample_pipeline step')
            sample_y = func(sample, **kwargs)
            check_array(sample_y,
                        'get_y_func called on "sample", **{}'.format(kwargs),
                        ensure_2d=False)
            logger.debug('Defined sample_y with shape {}'.format(sample_y.shape))
        elif 'get_weight' in args:
            logger.debug('get_weight sample_pipeline step')
            sample_weight = func(sample, sample_y, **kwargs)
            check_array(sample_weight,
                        'get_weight_func called on (sample, sample_y, **{}'.format(kwargs),
                        ensure_2d=False)
            logger.debug('Defined sample_weight with shape {}'.format(sample_weight.shape))
        elif func_str.endswith('feature_selection_base'):
            logger.debug('feature_selection sample_pipeline step')
            kw = copy.deepcopy(kwargs)
            kw['sample_y'] = sample_y
            sample = func(sample, *args, **kw)
        else:
            sample = func(sample, *args, **kwargs)

        if not isinstance(sample, ElmStore):
            raise ValueError('Expected the return value of {} to be an '
                             'elm.readers:ElmStore'.format(func))

        logger.debug('Shapes {}'.format(tuple(getattr(sample, b).values.shape for b in sample.data_vars)))
    return (sample, sample_y, sample_weight)


def get_sample_pipeline_action_data(train_or_predict_dict, config, step):
    '''Given sampling specs in a pipeline train or predict step,
    return action_data, a list of (func, args, kwargs) actions

    Params:
        train_or_predict_dict: a "train" or "predict" dict from config
        config:                full config
        step:                  a dictionary that is the current step
                               in the pipeline, like a "train" or "predict"
                               step
    '''
    d = train_or_predict_dict

    data_source = d['data_source']

    data_source = config.data_sources[d['data_source']]
    s = d.get('sample_args_generator',
                                  data_source.get('sample_args_generator'))
    if s:
        sample_args_generator = config.sample_args_generators[s]
        sample_args_generator = import_callable(sample_args_generator, True, sample_args_generator)
        sample_args_generator_kwargs = d.get('sample_args_generator_kwargs',
                                         data_source.get('sample_args_generator_kwargs')) or {}
    else:
        sample_args_generator = None
        sample_args_generator_kwargs = {}
    sample_args_generator_kwargs['data_source'] = data_source
    sampler_func = data_source['sample_from_args_func'] # TODO: this needs to be
                                                        # added to ConfigParser
                                                        # validation (sample_from_args_func requirement)
    band_specs = data_source.get('band_specs') or None
    sampler_args = data_source.get('sampler_args') or ()
    sampler_kwargs = data_source.get('sampler_kwargs') or {}
    # TODO the usage of sampler_args in config needs
    # to be validated
    if band_specs:
        sampler_args = (band_specs,)
    if sample_args_generator:
        sampler_kwargs.update(sample_args_generator_kwargs)
    reader_name = data_source.get('reader') or None
    if reader_name:
        reader = config.readers[reader_name]
        load_meta = import_callable(reader['load_meta'], True, reader['load_meta'])
        load_array = import_callable(reader['load_array'], True, reader['load_array'])
    else:
        reader = load_array = load_meta = None
    get_k = lambda k, v: data_source.get('selection_kwargs',{}).get(k, sampler_kwargs.get(k, d.get(k, data_source.get(k, v)) ))

    selection_kwargs = {
        'load_meta':       load_meta,
        'load_array':      load_array,
    }
    selection_kwargs.update(data_source.get('selection_kwargs') or {})

    for k in selection_kwargs:
        if '_filter' in k and selection_kwargs[k] and k != 'geo_filters':
            selection_kwargs[k] = import_callable(selection_kwargs[k])
    if sample_args_generator:
        kw = copy.deepcopy(selection_kwargs)
        kw.update(data_source)
        kw = {k: v for k, v in kw.items() if not k in ('band_specs',)}
        generated_args = get_generated_args(sample_args_generator,
                                            band_specs,
                                            no_file_open=False,
                                            **kw)
        sampler_kwargs['generated_args'] = generated_args
    sampler_kwargs.update(selection_kwargs)
    action_data = [('create_sample', sampler_func, sampler_args, sampler_kwargs)]
    sample_pipeline = step.get('sample_pipeline')
    if 'sample_pipeline' in step:
        actions = make_sample_pipeline_func(config, step)
        action_data.extend(actions)
    return tuple(action_data)

def make_sample_pipeline_func(config, step):
    '''Make list of (func, args, kwargs) tuples to run sample_pipeline
    Params:
        config: validated config from elm.config.ConfigParser
        step:   a dictionary that is one step of a "pipeline" list
    '''

    sample_pipeline = ConfigParser._get_sample_pipeline(config, step)
    actions = []
    if 'train' in step:
        key1 = 'train'
    elif 'predict' in step:
        key1 = 'predict'
    elif 'transform' in step:
        key1 = 'transform'
    else:
        raise ValueError('Expected "feature_selection" as a '
                         'key within a "train" or "predict" pipeline '
                         'action ({})'.format(action))
    d = getattr(config, key1)[step[key1]]
    data_source = config.data_sources[d['data_source']]
    for action in sample_pipeline:
        if 'feature_selection' in action:

            keep_columns = copy.deepcopy(data_source.get('keep_columns') or [])
            item = ('elm.sample_util.feature_selection:feature_selection_base',
                    (copy.deepcopy(config.feature_selection[action['feature_selection']]),),
                    {'keep_columns': keep_columns})
        elif 'random_sample' in action:
            item = ('elm.sample_util.random_rows:random_rows',
                    (action['random_sample'],),
                    {})
        elif 'transform' in action:
            item = ('elm.pipeline.transform:transform_sample_pipeline_step',
                    (action, config),
                    config.transform[action['transform']])

        elif 'sklearn_preprocessing' in action:
            scaler_kwargs = config.sklearn_preprocessing[action['sklearn_preprocessing']]
            scaler = scaler_kwargs['method']
            item = ('elm.sample_util.encoding_scaling:sklearn_preprocessing',
                    (scaler,),
                    scaler_kwargs)
        elif 'get_y' in action:
            func = data_source['get_y_func']
            args = ('get_y',)
            kwargs = data_source.get('get_y_kwargs') or {}
            item = (func, args, kwargs)
        elif 'get_weight' in action:
            func = data_source['get_weight_func']
            args = ('get_weight',)
            kwargs = data_source.get('get_weight_kwargs') or {}
            item = (func, args, kwargs)
        elif any(k in CHANGE_COORDS_ACTIONS for k in action):
            func, args, kwargs = change_coords_action(config, step, action)
            item = (func, args, kwargs)
        else:
            # add items to actions of the form:
            # (
            #   module_colon_func_name_as_string,        # string
            #   args_to_func,                            # tuple
            #   kwargs_to_func                           # dict
            # )
            # NOTE also add the key name, like 'transform' to the top of
            # elm.config.load_config global variable:
            # "SAMPLE_PIPELINE_ACTIONS"
            raise NotImplementedError('sample_pipeline action {} not recognized.'.format(action))
        actions.append((action,) + item)
    return actions


def check_array(arr, msg, **kwargs):
    '''Util func for checking sample remains finite and not-NaN'''
    if arr is None:
        raise ValueError('Array cannot be None ({}): '.format(msg))
    try:
        _check_array(arr, **kwargs)
    except Exception as e:
        shp = getattr(arr, 'shape', '(has no shape attribute)')
        logger.info('Failed on check_array on array with shape '
                    '{}'.format(shp))

        raise ValueError('check_array ({}) failed with {}'.format(msg, repr(e)))

def final_on_sample_step(fitter,
                         model, X,
                         iter_offset,
                         fit_kwargs,
                         sample_y=None,
                         sample_weight=None,
                         classes=None
                      ):
    '''This is the final function called on a sample_pipeline
    or a simple sample that is input to training.  It ensures
    that:
       * Corresponding Y data are looked up for the X sample
       * The correct fit kwargs are passed to fit or partial_fit,
         depending on the method
    Params:
       fitter:  a model attribute like "fit" or "partial_fit"
       model:   a sklearn model like MiniBatchKmeans()
       X:       an ElmStore or xarray.Dataset with 'flat' DataArray
       fit_kwargs: kwargs to fit_func from config
       Y: numpy array (same row count as s) or None
       sample_weight: numpy array (same row count as s) or None
       classes:  if using classification, all possible classes as iterable
                 or array of integers
       '''
    args, kwargs, var_keyword = get_args_kwargs_defaults(fitter)
    fit_kwargs = fit_kwargs or {}
    fit_kwargs = copy.deepcopy(fit_kwargs)
    Y = sample_y

    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    if 'iter_offset' in kwargs:
        fit_kwargs['iter_offset'] = iter_offset
    if 'check_input' in kwargs:
        fit_kwargs['check_input'] = True
    if any(a.lower() == 'y' for a in args) and Y is None:
        raise ValueError('Fit function {} requires a Y positional '
                         'argument but config\'s train section '
                         'get_y_func is not a callable'.format(fitter))
    if Y is not None:
        fit_args = (X.flat.values, Y)
        logger.debug('fit to X (shape {}) and Y (shape {})'.format(fit_args[0].shape, fit_args[1].shape))
    else:
        fit_args = (X.flat.values,)
        logger.debug('fit to X (shape {})'.format(fit_args[0].shape))
    check_array(X.flat.values, "final_on_sample_step - X.flat.values")
    if Y is not None:
        check_array(Y, "final_on_sample_step - Y")
    if sample_weight is not None:
        check_array(sample_weight, 'final_on_sample_step - sample_weight')
    if 'classes' in kwargs:
        if classes is None:
            raise ValueError('With model {} expected "classes" (unique classes int\'s) to be passed in config\'s "train" or "transform" dictionary')
        fit_kwargs['classes'] = classes
    if 'batch_size' in model.get_params():
        logger.debug('set batch_size {}'.format(X.flat.values.shape[0]))
        model.set_params(batch_size=X.flat.values.shape[0])
    return fit_args, fit_kwargs

