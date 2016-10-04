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
from elm.readers import load_meta, load_array


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


def _split_pipeline_output(output, sample, sample_y,
                           sample_weight, context):
    if not isinstance(output, (tuple, list)):
        return output, sample_y, sample_weight
    if not output:
        raise ValueError('{} sample_pipeline func returned falsy {}'.format(context, repr(output)))
    if len(output) == 1:
        return output, sample_y, sample_weight
    elif len(output) == 2:
        return tuple(output) + (sample_weight,)
    elif len(output) == 3:
        return tuple(output)
    else:
        raise ValueError('{} sample_pipeline func returned '
                         'more than 3 outputs in a '
                         'tuple/list'.format(context))


def run_sample_pipeline(action_data, sample=None, sample_y=None, sample_weight=None, transform_model=None):
    '''Given action_data as a list of (func, args, kwargs) tuples,
    run each function passing args and kwargs to it
    Params:
        action_data:     list from get_sample_pipeline_action_data typically
        sample:          None if the sample is not already taken
        transform_model: An example:
                             [('tag_0', PCA(.....))]
    '''
    check_action_data(action_data)
    for action in action_data:
        sample_pipeline_step, func_str, args, kwargs = action
        kwargs = kwargs.copy()
        kwargs['sample_y'] = sample_y
        kwargs['sample_weight'] = sample_weight
        logger.debug('On sample_pipeline step: {}'.format(sample_pipeline_step))
        func = import_callable(func_str, True, func_str)
        func_out = None

        if 'create_sample' in sample_pipeline_step and sample is None:
            logger.debug('sample create sample_pipeline step')
            required_args, default_kwargs, var_keyword = get_args_kwargs_defaults(func)
            kwargs = {k: v for k,v in kwargs.items() if k not in required_args}

            output = func(*args, **kwargs)
            sample, sample_y, sample_weight = _split_pipeline_output(output,
                                               sample, sample_y,
                                               sample_weight, repr(func))
        elif 'create_sample' in sample_pipeline_step:
            continue
        elif 'transform' in sample_pipeline_step:
            logger.debug('transform sample_pipeline step')
            args = tuple(args) + (transform_model,)
        elif 'get_y' in sample_pipeline_step:
            logger.debug('get_y sample_pipeline step')
            sample_y = func(sample, **kwargs)
            check_array(sample_y,
                        'get_y_func called on "sample", **{}'.format(kwargs),
                        ensure_2d=False)
            logger.debug('Defined sample_y with shape {}'.format(sample_y.shape))
        elif 'get_weight' in sample_pipeline_step:
            logger.debug('get_weight sample_pipeline step')
            sample_weight = func(sample, sample_y, **kwargs)
            check_array(sample_weight,
                        'get_weight_func called on (sample, sample_y, **{}'.format(kwargs),
                        ensure_2d=False)
            logger.debug('Defined sample_weight with shape {}'.format(sample_weight.shape))
        elif 'feature_selection' in sample_pipeline_step:
            logger.debug('feature_selection sample_pipeline step')
            kw = copy.deepcopy(kwargs)
            kw['sample_y'] = sample_y
            kw.pop('sample_weight')
            func_out = func(sample, *args, **kw)
        else:
            func_out = func(sample, *args, **kwargs)
        if func_out is not None:
            sample, sample_y, sample_weight = _split_pipeline_output(func_out, sample, sample_y,
                                                   sample_weight, repr(func))
        if not isinstance(sample, (ElmStore, xr.Dataset)):
            raise ValueError('Expected the return value of {} to be an '
                             'elm.readers:ElmStore'.format(func))

        logger.debug('Shapes {}'.format(tuple(getattr(sample, b).values.shape for b in sample.data_vars)))
    return (sample, sample_y, sample_weight)


def get_sample_pipeline_action_data(sample_pipeline, config=None, step=None,
                                    data_source=None, **sample_pipeline_kwargs):
    '''Given sampling specs in a pipeline train or predict step,
    return action_data, a list of (func, args, kwargs) actions

    Params:
        train_or_predict_dict: a "train" or "predict" dict from config
        config:                full config
        step:                  a dictionary that is the current step
                               in the pipeline, like a "train" or "predict"
                               step
    '''
    sampler_func = data_source['sample_from_args_func'] # TODO: this needs to be
                                                        # added to ConfigParser
                                                        # validation (sample_from_args_func requirement)
    band_specs = data_source.get('band_specs') or None
    sampler_args = data_source.get('sampler_args') or ()
    # TODO the usage of sampler_args in config needs
    # to be validated
    if band_specs:
        sampler_args = (band_specs,) + tuple(sampler_args)
    reader_name = data_source.get('reader') or None
    if reader_name:
        if isinstance(reader_name, dict):
            reader = reader_name
        elif config and reader_name in config.readers:
            reader = config.readers[reader_name]
        _load_meta = import_callable(reader['load_meta'], True, reader['load_meta'])
        _load_array = import_callable(reader['load_array'], True, reader['load_array'])
    else:
        _load_array = load_array
        _load_meta = load_meta
    data_source['load_meta'] = _load_meta
    data_source['load_array'] = _load_array
    for k in data_source:
        if '_filter' in k and data_source[k] and k != 'geo_filters':
            data_source[k] = import_callable(data_source[k])
    kw = {k: v for k, v in data_source.items() if not k in ('band_specs',)}
    sample_args_generator = data_source.get('sample_args_generator') or None
    if sample_args_generator:
        if isinstance(sample_args_generator, (tuple, str)) and config and sample_args_generator in config.sample_args_generators:
            sample_args_generator = import_callable(config.sample_args_generators[sample_args_generator])
        else:
            sample_args_generator = import_callable(sample_args_generator)
        logger.debug('Calling sample_args_generator')
        generated_args = get_generated_args(sample_args_generator,
                                            band_specs,
                                            sampler_func,
                                            **kw)
        data_source['generated_args'] = generated_args
    else:
        data_source['generated_args'] = [(sampler_args, data_source)
                                          for _ in range(data_source.get('n_batches') or 1)]

    action_data = [('create_sample', sampler_func, sampler_args, data_source)]
    actions = make_sample_pipeline_func(config=config, step=step,
                                        sample_pipeline=sample_pipeline,
                                        data_source=data_source,
                                        **sample_pipeline_kwargs)
    action_data.extend(actions)
    return tuple(action_data)

def make_sample_pipeline_func(config=None,
                              step=None,
                              sample_pipeline=None,
                              data_source=None,
                              feature_selection=None,
                              transform_dict=None,
                              sklearn_preprocessing=None,
                              **kwargs):
    '''Make list of (func, args, kwargs) tuples to run sample_pipeline
    Params:
        config: validated config from elm.config.ConfigParser
        step:   a dictionary that is one step of a "pipeline" list
    '''

    actions = []
    for action_idx, action in enumerate(sample_pipeline):
        if 'feature_selection' in action:
            if feature_selection is None and config:
                _feature_selection = copy.deepcopy(config.feature_selection[action['feature_selection']])
            elif feature_selection is None:
                raise ValueError('Expected "feature_selection" or "config"')
            elif action['feature_selection'] in feature_selection:
                _feature_selection = feature_selection[action['feature_selection']]
            else:
                _feature_selection = feature_selection
            keep_columns = _feature_selection.get('keep_columns')
            if not keep_columns and data_source:
                keep_columns = data_source.get('keep_columns', [])
            item = ('elm.sample_util.feature_selection:feature_selection_base',
                    (_feature_selection,),
                    {'keep_columns': keep_columns})
        elif 'random_sample' in action:
            item = ('elm.sample_util.random_rows:random_rows',
                    (action['random_sample'],),
                    {})
        elif 'transform' in action:
            if transform_dict:
                if action['transform'] in transform_dict:
                    trans = transform_dict[action['transform']]
                else:
                    trans = transform_dict
            elif config:
                trans = config.transform[action['transform']]
            else:
                trans = {}
            item = ('elm.pipeline.transform:transform_sample_pipeline_step',
                    (action, config),
                     trans)

        elif 'sklearn_preprocessing' in action:
            if sklearn_preprocessing is None and config:
                _sklearn_preprocessing = config.sklearn_preprocessing[action['sklearn_preprocessing']]
            elif sklearn_preprocessing is None:
                raise ValueError('Expected "config" if not giving "sklearn_preprocessing"')
            elif action['sklearn_preprocessing'] in sklearn_preprocessing:
                _sklearn_preprocessing = sklearn_preprocessing[action['sklearn_preprocessing']]
            else:
                _sklearn_preprocessing = sklearn_preprocessing
            scaler = _sklearn_preprocessing['method']
            item = ('elm.sample_util.encoding_scaling:sklearn_preprocessing',
                    (scaler,),
                    _sklearn_preprocessing)
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
            func, args, kwargs = change_coords_action(action)
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


def _has_arg(a):
    return not (a is None or a == [] or (hasattr(a, 'size') and a.size == 0))


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
    has_y = _has_arg(Y)
    has_sw = _has_arg(sample_weight)
    if has_sw:
        fit_kwargs['sample_weight'] = sample_weight
    if 'iter_offset' in kwargs:
        fit_kwargs['iter_offset'] = iter_offset
    if 'check_input' in kwargs:
        fit_kwargs['check_input'] = True
    #if any(a.lower() == 'y' for a in args) and not has_y:
     #   raise ValueError('Fit function {} requires a Y positional '
      #                   'argument but config\'s train section '
       #                  'get_y_func is not a callable'.format(fitter))
    if has_y:
        fit_args = (X.flat.values, Y)
        logger.debug('fit to X (shape {}) and Y (shape {})'.format(fit_args[0].shape, fit_args[1].shape))
    else:
        fit_args = (X.flat.values,)
        logger.debug('fit to X (shape {})'.format(fit_args[0].shape))
    check_array(X.flat.values, "final_on_sample_step - X.flat.values")
    if has_y:
        check_array(Y, "final_on_sample_step - Y")
    if has_sw:
        check_array(sample_weight, 'final_on_sample_step - sample_weight')
    if 'classes' in kwargs:
        if classes is None:
            raise ValueError('With model {} expected "classes" (unique classes int\'s) to be passed in config\'s "train" or "transform" dictionary')
        fit_kwargs['classes'] = classes
    if 'batch_size' in model.get_params():
        logger.debug('set batch_size {}'.format(X.flat.values.shape[0]))
        model.set_params(batch_size=X.flat.values.shape[0])
    return fit_args, fit_kwargs

