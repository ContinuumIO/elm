from functools import partial, wraps
from itertools import chain, product
import copy
import logging
import numbers
import random

import dask

from elm.config import import_callable
from elm.config.dask_settings import _find_get_func_for_client
from elm.pipeline.util import (_run_model_selection,
                               _next_name)
from elm.sample_util.samplers import make_samples_dask

logger = logging.getLogger(__name__)

__all__ = ['ensemble']


def _fit_once(method, model, fit_score_kwargs, args):
    '''Dask helper internal func to call a model's method

    Parameters:
        method: str such as "fit", "fit_transform", "transform",
                "partial_fit"
        model: elm.pipeline.Pipeline instance
        fit_score_kwargs: kwargs to method
        args:   Tuple of 1, 2 or 3 elements:

                (X,)
                (X, y,)
                (X, y, sample_weight)

                where X is an ElmStore with Dataset "flat"
                y and sample_weight if given are numpy arrays
    Returns:
        the output of Pipeline instance's "method" call, see also
        elm.pipeline.Pipeline
    '''
    X = args[0]
    if len(args) > 1:
        y = args[1]
    else:
        y = None
    if len(args) > 2:
        sample_weight = args[2]
    else:
        sample_weight = None
    fitting_func = getattr(model, method, None)
    if fitting_func is None:
        raise ValueError("Estimator {} has no method {}".format(model, fitting_func))
    kw = dict(**fit_score_kwargs)
    if y is not None:
        kw['y'] = y
    if sample_weight is not None:
        kw['sample_weight'] = sample_weight
    return fitting_func(X, method_kwargs=kw)


def _one_generation_dask_graph(dsk,
                               models,
                               fit_score_kwargs,
                               sample_keys,
                               partial_fit_batches,
                               gen,
                               method):

    '''Run a group of models' fit method on a group of samples
    Parameters:
        dsk: dask spec dict with keys / values for samples
        models: list of (tag, elm.pipeline.Pipeline instances)
        fit_score_kwargs: kwargs to "fit" or other method, such as
                          "classes" for a supervised classifier
        sample_keys: keys in dsk which form the "arg" argument
                     to _fit_once helper above
        partial_fit_batches: how many partial_fit's
        gen: which generation is it - passed to model_selection func
        method: One of: "fit", "fit_transform", "transform", "partial_fit"

    Returns:
        tuple of (dsk, model_keys, new_models_name)
        where:
            dsk: dsk dask spec dict modified in place with new model keys
            model_keys: the tag of each model
            new_models_name: a collector key of all fitted models
    '''
    model_keys = [_[0] for _ in models]
    collect_keys = []
    token = '{}-gen-{}'.format(method, gen)
    for (key, model), arg in product(models, sample_keys):
        name = _next_name(token)
        dsk[name] = (partial(_fit_once, method, model, fit_score_kwargs), arg)
        collect_keys.append(name)
    if partial_fit_batches > 1:
        for idx in range(1, partial_fit_batches):
            token_pf = token + '_batch_{}'.format(idx)
            collect_keys2 = []
            for key in collect_keys:
                for sample_key in sample_keys:
                    name = _next_name(token_pf)
                    dsk[name] = ((lambda model, arg: _fit_once(method, model, fit_score_kwargs, arg)), key, sample_key)
                    collect_keys2.append(name)
            collect_keys = collect_keys2
    def tuple_of_args(*args):
        return tuple(args)
    new_models_name = _next_name('ensemble_generation_{}'.format(gen))
    dsk[new_models_name] = (tuple_of_args, *collect_keys)
    return dsk, model_keys, new_models_name

def ensemble(pipe,
             ngen,
             X=None,
             y=None,
             sample_weight=None,
             sampler=None,
             args_list=None,
             client=None,
             init_ensemble_size=1,
             saved_ensemble_size=None,
             ensemble_init_func=None,
             models_share_sample=True,
             model_selection=None,
             model_selection_kwargs=None,
             scoring_kwargs=None,
             method='fit',
             partial_fit_batches=1,
             classes=None,
             method_kwargs=None,
             **data_source):

    '''Fit or partial_fit an ensemble of models to a series of samples

    Call this function from an elm.pipeline.Pipeline instance's methods:
        "fit_ensemble"
        "fit_transform_ensemble"
        "transform_ensemble"

    Parameters:
        pipe: instance of elm.pipeline.Pipeline
        ngen: number of ensemble generations
        X:    elm.readers.ElmStore, if not using "sampler" and "args_list"
        y:    numpy array if not using "sampler" and "args_list", or None
              if not needed by Pipeline
        sample_weight: numpy array if not using "sampler" and "args_list", or None
              if not needed by Pipeline
        sampler: Callable - required if not giving X.  Called at least once
                on each element of args_list where each element is unpacked
                with *one_element_of_args_list
        args_list: List of args - required if not giving X.  See sampler above
        client: dask-distributed or ThreadPool client
        init_ensemble_size: number of ensemble members, ignored if giving
              ensemble_init_func
        saved_ensemble_size: how many members to keep at final generation
        ensemble_init_func: Callable to return list of elm.pipeline.Pipeline
             instances that initialize ensemble
        models_share_sample: If True, ensure that in each generation, every
             member is fit to the same sample.  If False, fit every model
             to every sample
        model_selection: Callable after each generation to take a list of
            (tag, Pipeline) tuples and return a list of new such tuples, or
            None or repeatedly train each model on each generation without
            replacement of model parameters
        model_selection_kwargs: kwargs passed to model_selection
        scoring_kwargs: kwargs that are passed to score_one_model.
                See also elm.model_selection.scoring
        method: This is the method of Pipeline that called this ensemble
            function, typically "fit"
        classes: Unique sequence of class integers passed to supervised
            classifiers that need the known y classes.
        method_kwargs: any other arguments to pass to method
        **data_source: keywords passed to "sampler" if given
    Returns:

        new_models: list of (tag, Pipeline instance) tuples on which
            "predict_many" can be called
    '''
    get_func = _find_get_func_for_client(client)
    fit_score_kwargs = method_kwargs or {}
    if not 'classes' in fit_score_kwargs and classes is not None:
        fit_score_kwargs['classes'] = classes
    model_selection_kwargs = model_selection_kwargs or {}
    ensemble_size = init_ensemble_size or 1
    partial_fit_batches = partial_fit_batches or 1
    if partial_fit_batches > 1:
        method = 'partial_fit'
    if not ensemble_init_func:
        models = tuple(copy.deepcopy(pipe) for _ in range(ensemble_size))
    else:
        ensemble_init_func = import_callable(ensemble_init_func)
        models = ensemble_init_func(pipe, ensemble_size=ensemble_size)
    logger.info("Init ensemble: {} members".format(len(models)))
    if model_selection:
        model_selection = import_callable(model_selection)
    final_names = []
    dsk = make_samples_dask(X, y, sample_weight, pipe, args_list, sampler, data_source)
    models = tuple(zip(('tag_{}'.format(idx) for idx in range(len(models))), models))
    sample_keys = list(dsk)
    if models_share_sample:
        random.shuffle(sample_keys)
        gen_to_sample_key = {gen: s for gen, s in enumerate(sample_keys[:ngen])}
    sample_keys = tuple(sample_keys)
    for gen in range(ngen):
        if models_share_sample:
            sample_keys_passed = (gen_to_sample_key[gen % len(sample_keys)],)
        else:
            sample_keys_passed = sample_keys
        logger.info('Ensemble generation {} of {} - ({} estimators) '.format(gen + 1, ngen, len(models)))
        msg = (len(models), len(sample_keys_passed),
               partial_fit_batches, method,
               len(models) * len(sample_keys_passed) * partial_fit_batches,
               gen + 1,
               ngen)
        logger.info('Ensemble Generation {5} of {6}: ({0} members x {1} samples x {2} calls) = {4} {3} calls this gen'.format(*msg))
        dsk, model_keys, new_models_name = _one_generation_dask_graph(dsk,
                                                      models,
                                                      fit_score_kwargs,
                                                      sample_keys_passed,
                                                      partial_fit_batches,
                                                      gen,
                                                      method)
        if get_func is None:
            new_models = tuple(dask.get(dsk, new_models_name))
        else:
            new_models = tuple(get_func(dsk, new_models_name))
        models = tuple(zip(model_keys, new_models))
        logger.info('Trained {} estimators'.format(len(models)))
        if model_selection:
            models = _run_model_selection(models,
                                          model_selection,
                                          model_selection_kwargs or {},
                                          ngen,
                                          gen,
                                          scoring_kwargs)

        else:
            pass # Just training all ensemble members
                 # without replacing / re-ininializing / editing
                 # the model params
    if saved_ensemble_size:
        final_models = models[:saved_ensemble_size]
    else:
        final_models = models

    return final_models


