import copy
from functools import partial
import logging
from pprint import pformat

import numpy as np
import xarray as xr
from sklearn.utils import check_array as _check_array
import sklearn.preprocessing as skpre
import sklearn.feature_selection as skfeat

from elm.config import import_callable, ConfigParser
from elm.model_selection.util import get_args_kwargs_defaults
from elm.model_selection.util import filter_kwargs_to_func
from elm.readers import (ElmStore, flatten as _flatten)
from elm.sample_util.change_coords import (change_coords_dict_action,
                                           CHANGE_COORDS_ACTIONS)
from elm.sample_util.filename_selection import get_args_list
from elm.readers.util import row_col_to_xy
from elm.readers import load_meta, load_array, ElmStore


logger = logging.getLogger(__name__)

_SAMPLE_PIPELINE_SPECS = {}


def _split_pipeline_output(output, X, y,
                           sample_weight, context):

    if not isinstance(output, (tuple, list)):
        return output, y, sample_weight
    if output is None:
        return X, y, sample_weight
    if len(output) == 1:
        return output[0], y, sample_weight
    elif len(output) == 2:
        xx = output[0] if output[0] is not None else X
        yy = output[1] if output[1] is not None else y
        return (xx, yy, sample_weight,)
    elif len(output) == 3:
        xx = output[0] if output[0] is not None else X
        yy = output[1] if output[1] is not None else y
        sw = output[2] if output[2] is not None else sample_weight
        return (xx, yy, sw)
    else:
        raise ValueError('{} pipeline func returned '
                         'more than 3 outputs in a '
                         'tuple/list'.format(context))


def create_sample_from_data_source(config=None, **data_source):
    '''Given sampling specs in a pipeline train or predict step,
    return pipe, a list of (func, args, kwargs) actions

    Params:
        train_or_predict_dict: a "train" or "predict" dict from config
        config:                full config
        step:                  a dictionary that is the current step
                               in the pipeline, like a "train" or "predict"
                               step
    '''
    sampler_func = data_source['sampler'] # TODO: this needs to be
                                                        # added to ConfigParser
                                                        # validation (sampler requirement)
    sampler_func = import_callable(sampler_func)
    band_specs = data_source.get('band_specs') or None
    sampler_args = data_source.get('sampler_args') or ()

    reader_name = data_source.get('reader') or None
    if isinstance(reader_name, str) and reader_name:
        if config and reader_name in config.readers:
            reader = config.readers[reader_name]
        _load_meta = partial(load_meta, reader=reader_name)
        _load_array = partial(load_array, reader=reader_name)
    elif isinstance(reader_name, dict):
        reader = reader_name
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
    return sampler_func(*sampler_args, **data_source)

# MOve this logic somwehre
# pipe = [('create_sample', sampler_func, sampler_args, data_source)]
# actions = make_pipeline_func(config=config, step=step,
#                                    pipeline=pipeline,
#                                    data_source=data_source,
#                                    **pipeline_kwargs)
# pipe.extend(actions)
# return tuple(pipe)


def make_pipeline_func(config, pipeline):
    '''Make list of (func, args, kwargs) tuples to run pipeline
    Params:
        config: validated config from elm.config.ConfigParser
        step:   a dictionary that is one step of a "pipeline" list
    '''
    from elm.pipeline import steps
    actions = []
    for action_idx, action in enumerate(pipeline):
        is_dic = isinstance(action, dict)
        if not is_dic:
            step_cls = action
        elif 'feature_selection' in action:
            _feature_selection = copy.deepcopy(config.feature_selection[action['feature_selection']])
            kw = _feature_selection.copy()
            kw.update(action)
            scaler = feature_selection['method']
            scaler = import_callable(getattr(skfeat, scaler, scaler))
            if 'func_kwargs' in _feature_selection:
                func = import_callable(_feature_selection['func'])
                scaler = partial(func, feature_selection['func_kwargs'])
                _feature_selection['func'] = func
            kw = {k: v for k, v in _feature_selection.items()
                  if k not in ('func_kwargs', 'method')}
            step_cls = steps.SklearnBase(_method=scaler, _mod=skpre, **kw)
        elif 'random_sample' in action:
            step_cls = RandomSample(action['random_sample'], **action)
        elif 'transform' in action:
            trans = config.transform[action['transform']]
            cls = import_callable(trans['model_init_class'])
            t = cls(**(trans.get('model_init_kwargs') or {}))
            step_cls = steps.Transform(t)
        elif 'sklearn_preprocessing' in action:
            _sklearn_preprocessing = config.sklearn_preprocessing[action['sklearn_preprocessing']]
            scaler = _sklearn_preprocessing['method']
            scaler = getattr(skpre, scaler, scaler)
            kw = {k: v for k, v in _sklearn_preprocessing.items()
                  if not k in ('method','func_kwargs')}
            step_cls = steps.SklearnBase(_method=scaler, _mod=skpre, **kw)
        elif any(k in CHANGE_COORDS_ACTIONS for k in action):
            _sp_step = [k for k in action if k in CHANGE_COORDS_ACTIONS][0]
            _context = _sp_step
            step_cls = steps.ChangeCoordsMixin(arg=_sp_step, _sp_step=_sp_step, _context=_context, **action)
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
            raise NotImplementedError('pipeline action {} not recognized.'.format(action))
        actions.append((action, step_cls) )
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
                         fit_kwargs,
                         y=None,
                         sample_weight=None,
                         classes=None,
                         require_flat=True,
                         prepare_for='train'):
    if isinstance(X, np.ndarray):
        X_values = X             # numpy array 2-d
    elif isinstance(X, (ElmStore, xr.Dataset)):
        if hasattr(X, 'flat'):
            X_values = X.flat.values
        else:
            logger.info("After running Pipeline, X is not an ElmStore with a DataArray called 'flat' and X is not a numpy array.  Found {}".format(type(X)))
            logger.info("Trying elm.readers.reshape:flatten on X. If this fails, try a elm.pipeline.steps:ModifySample step to create ElmStore with 'flat' DataArray")
            X = _flatten(X)
            X_values = X.flat.values
    else:
        X_values = X # may not be okay for sklearn models,e.g KMEans but can be passed thru Pipeline
    args, kwargs, var_keyword = get_args_kwargs_defaults(fitter)
    fit_kwargs = fit_kwargs or {}
    fit_kwargs = copy.deepcopy(fit_kwargs)
    has_y = _has_arg(y)
    has_sw = _has_arg(sample_weight)
    if has_sw:
        fit_kwargs['sample_weight'] = sample_weight
    if 'check_input' in kwargs:
        fit_kwargs['check_input'] = True
    if has_y:
        if prepare_for == 'train':
            fit_args = (X_values, y)
        else:
            fit_args = (X,)
        logger.debug('X (shape {}) and y (shape {})'.format(X_values.shape, y.shape))
    else:
        if prepare_for == 'train':
            fit_args = (X_values,)
        else:
            fit_args = (X,)
        logger.debug('X (shape {})'.format(X_values.shape))
    check_array(X_values, "final_on_sample_step - X")
    if has_y:
        check_array(y, "final_on_sample_step - y")
    if has_sw:
        check_array(sample_weight, 'final_on_sample_step - sample_weight')
    if 'classes' in kwargs:
        if classes is None:
            raise ValueError('With model {} expected "classes" (unique classes int\'s) to be passed in config\'s "train" or "transform" dictionary')
        fit_kwargs['classes'] = classes
    params = model.get_params()
    if 'batch_size' in params:
        logger.debug('set batch_size {}'.format(X_values.shape[0]))
        params['batch_size']  = X_values.shape[0]
        model.set_params(**params)
    return fit_args, fit_kwargs


