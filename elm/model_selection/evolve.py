import array
from collections import (defaultdict, OrderedDict,
                         namedtuple, Sequence)
import copy
from functools import partial
import numbers
import random
import logging
import warnings

from deap import base
from deap import creator
from deap import tools
import numpy as np

from elm.model_selection.util import (MODEL_FIELDS,
                                      ModelArgs,
                                      get_args_kwargs_defaults)
from elm.config import (import_callable,
                        ElmConfigError,
                        ConfigParser)

logger = logging.getLogger(__name__)

DEFAULT_PERCENTILES = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)

EVO_FIELDS = ['toolbox',
              'individual_to_new_config',
              'deap_params',
              'param_grid_name',
              'history_file',
              'score_weights',
              'early_stop']
EvoParams = namedtuple('EvoParams', EVO_FIELDS)

OK_PARAM_FIRST_KEYS = ['transform',
                       'train',
                       'feature_selection',
                       'model_scoring',
                       'data_sources',
                       'sklearn_preprocessing',
                       'sample_pipeline',]

DEFAULT_MAX_PARAM_RETRIES = 1000

LAST_TAG_IDX = 0

REQUIRED_CONTROL_KEYS_TYPES = {
    'select_method': str,
    'crossover_method': str,
    'mutate_method': str,
    'init_pop': str,
    'indpb': float,
    'mutpb': float,
    'cxpb':  float,
    'eta':   int,
    'ngen':  int,
    'mu':    int,
    'k':     int,
}
EARLY_STOP_KEYS = ['abs_change', 'percent_change', 'threshold']

def next_model_tag():
    '''Gives names like tag_0, tag_1, tag_2 sequentially'''
    global LAST_TAG_IDX
    model_name =  'tag_{}'.format(LAST_TAG_IDX)
    LAST_TAG_IDX += 1
    return model_name

def assign_names(pop, names=None):
    '''Assign names to the models in the population'''
    names = names or [next_model_tag() for _ in pop]
    for ind, name in zip(pop, names):
        ind.name = name


def _make_cfg_replace_keys(train_config, transform_config,
              train_step_name, transform_step_name,
              config):
    '''Return a callable which can parse param_grid params to tuple of keys
    into config

    For example the returned func can take a parameter named:
    "pca__n_components" and return:
    ("transform", "pca", "model_init_kwargs", "n_components")

    Parameters:
        train_config:         "train" section of config
        transform_config:     "transform" section of config
        train_step_name:      corresponding key in train
        transform_step_name:  corresponding key in transform
        config:               elm.config.ConfigParser instance

    Returns:
        make_cfg_replace_keys - callable returning tuple of keys to config
    '''
    if train_config:
        klass = import_callable(train_config['model_init_class'])
        _, default_kwargs, _ = get_args_kwargs_defaults(klass)
        train_or_transform = 'train'
    else:
        train_or_transform = 'transform'
        train_config = {}
        default_kwargs = {}
    if transform_config:
        klass = import_callable(transform_config['model_init_class'])
        _, default_transform_kwargs, _ = get_args_kwargs_defaults(klass)
    else:
        default_transform_kwargs = {}
        transform_config = {}

    def get_train_transform(k):
        if k[-1] in default_kwargs:
            if len(k) == 2:
                if k[0] != train_step_name:
                    raise ElmConfigError('Did not expect token {} in param_grid ({})'.format(k[0], k))
            return ('train', train_step_name, 'model_init_kwargs', k[-1])
        elif k[-1] in default_transform_kwargs:
            if len(k) == 2:
                if k[0] != transform_step_name:
                    raise ElmConfigError('Did not expect token {} in param_grid ({})'.format(k[0], k))
            return ('transform', transform_step_name, 'model_init_kwargs', k[-1])

    def make_cfg_replace_keys(k, v):
        if len(k) == 1:
            tt = get_train_transform(k)
            if tt:
                return (tt, v)
            has_under = '__' in k[0]
            if has_under:
                parts = k[0].split('__')
                if len(parts) != 2:
                    raise ElmConfigError('param_grid key: {} uses __ more than once'.format(k))
                tt = get_train_transform(tuple(parts))
                if tt:
                    return (tt, v)
                raise ElmConfigError('param_grid key: {} uses __ but left '
                              'side of __ is not in "train" or '
                              '"transform" keys of config (or '
                              'the left side of __ is not part of '
                              '{} step {} in config\'s '
                              'pipeline'.format(k,
                                                train_or_transform,
                                                train_step_name or transform_step_name))
            elif k[0] == 'sample_pipeline':
                return (('sample_pipeline_variable',), v)
            else:
                raise ElmConfigError('Expected param_grid key to be in "train" '
                                     'or "transform" model_init_kwargs, '
                                     'using "__" to split (train/transform name '
                                     'from a model_init_kwargs,'
                                     'or using nested dictionary. TODO further help.')
        else:
            d = config.config
            for key in k:
                if not key in d:
                    raise ElmConfigError('Given param_grid spec {}, expected {} in {}'.format(k, key, d))
                d = d[key]
            return (k, v)
    return make_cfg_replace_keys


def parse_param_grid_items(param_grid_name, train_config, transform_config,
                          make_cfg_replace_keys, param_grid):
    '''Parse each param in param_grid to tuple of keys into nested config

    Parameters:
        param_grid_name:      string name of a param_grid from config's param_grids
        train_config:         "train" section of config
        transform_config:     "transform" section of config
        make_cfg_replace_keys: callable returned by _make_cfg_replace_keys
        param_grid:           a param_grid dict from config's param_grids section
    Returns:
        param_grid_parsed: dict of tuple keys to param choices list
                           also has key "param_order"
    '''
    transform_config = transform_config or {}
    def switch_types(current_key, obj):
        if isinstance(obj, (list, tuple)):
            yield (current_key, obj)
        elif isinstance(obj, dict):
            for idx, (key, value) in enumerate(obj.items()):
                yield from switch_types(current_key + (key,), value)
                if idx > 0:
                    raise ElmConfigError('Expected nested dictionary with '
                                         'at most 1 key per dictionary at '
                                         'each nesting level.  Found: {}'.format(obj))
        else:
            raise ValueError('Did not expect {} (not dict, list or tuple) in {}'.format(obj, param_grid_name))
    param_grid_parsed = {}
    param_order = []
    for k, v in param_grid.items():
        if k == 'control':
            param_grid_parsed[k] = v
            continue
        unwound = tuple(switch_types((k,), v))[0]
        k2, v2 = make_cfg_replace_keys(*unwound)
        param_grid_parsed[k2] = v2
        param_order.append(k2)
    param_grid_parsed['param_order'] = param_order
    return param_grid_parsed


def get_param_grid(config, step1, step):
    '''Get the metadata for one config step that has a param_grid or return None'''
    param_grid_name = step.get('param_grid') or None
    if not param_grid_name:
        return None
    train_step_name = step.get('train')
    transform_step_name = step.get('transform')
    if train_step_name:
        train_config = config.train[train_step_name]
        transform_config = None
    elif transform_step_name:
        transform_config = config.transform[transform_step_name]
        train_config = None
    else:
        raise ElmConfigError('Expected param_grid to be used with a "train" '
                         'or "transform" step of a pipeline, but found param_grid '
                         'was used in step {}'.format(step))
    if 'sample_pipeline' in step1 and transform_step_name is None:
        sample_pipeline = step1['sample_pipeline']
        transform_steps = [_ for _ in sample_pipeline if 'transform' in _]
        transform_names = set(_.get('transform') for _ in transform_steps)
        if len(transform_names) > 1:
            raise ElmConfigError('Expected a single transform model but got {}'.format(transform_names))
        if transform_names:
            transform_step_name = tuple(transform_names)[0]
            transform_config = config.transform[transform_step_name]

    param_grid = config.param_grids[param_grid_name]

    if not isinstance(param_grid, dict):
        raise ElmConfigError('Expected param_grids: {} to be a dict '
                             'but found {}'.format(param_grid_name, param_grid))
    control = param_grid.get('control') or {}
    if not isinstance(control, dict):
        raise ElmConfigError("Expected 'control' as dict")
    early_stop = control.get('early_stop') or None
    if early_stop:
        if not isinstance(early_stop, dict) or not any(k in early_stop for k in ('abs_change', 'percent_change', 'threshold')):
            raise ElmConfigError('Expected "early_stop" to be a dict with a key in ("abs_change", "percent_change", "threshold")')
    if not isinstance(control, dict):
        raise ElmConfigError('Expected param_grids: {} - "control" to be a dict'.format(control))
    for required_key, typ in REQUIRED_CONTROL_KEYS_TYPES.items():
        item = control.get(required_key) or None
        if not isinstance(item, typ):
            raise ElmConfigError('Expected params_grids:{} '
                                 '"control" to have key {} with '
                                 'type {}'.format(param_grid_name, required_key, typ))
    make_cfg_replace_keys = _make_cfg_replace_keys(train_config,
                                       transform_config,
                                       train_step_name,
                                       transform_step_name,
                                       config)
    param_grid = parse_param_grid_items(param_grid_name,
                                        train_config,
                                        transform_config,
                                        make_cfg_replace_keys,
                                        param_grid)
    param_grid['control']['step'] = step
    assert all(isinstance(v, list) for k, v in param_grid.items() if k != 'control'), (repr(param_grid))
    if not 'control' in param_grid:
        raise ValueError('Expected a control dict in param_grid:{}'.format(param_grid_name))
    return {param_grid_name: _to_param_meta(param_grid)}


def _to_param_meta(param_grid):
    '''Acquire parameter metadata such as bounds that are useful for sampling'''
    low = []
    up = []
    is_int = []
    choices = []
    for key in param_grid['param_order']:
        values = param_grid[key]
        if key == 'control':
            continue
        if not isinstance(values, list) or not values:
            raise ValueError('param_grid: {} has a value that is not a '
                             '(list, tuple) with non-zero length: {}'.format(key, values))
        is_int.append(all(isinstance(v, numbers.Integral) for v in values))
        low.append(min(values))
        up.append(max(values))
        choices.append(values)
    param_meta = {
                  'is_int':      is_int,
                  'low':         low,
                  'up':          up,
                  'choices':     choices,
                  'control':     param_grid['control'],
                  'param_order': param_grid['param_order'],
                  }
    return param_meta


def out_of_bounds(params, choices):
    '''Check boudns on indices into choices lists'''
    for p, choice in zip(params, choices):
        if p < 0 or p >= len(choice):
            return True
    return False


def wrap_mutate(method, choices, max_param_retries, individual, **kwargs):
    '''Mutation for the method, choices and other config options

    Parameters:
        method:  string - imported from deap.tools such as mutUniformInt
        choices: list of lists choices for each parameter
        max_param_retries: how many times to retry when getting invalid params
        individual:        deap Individual parameter ste
        kwargs:            kwargs passed as args to method given

    Returns:
        tuple of one Individual parameter set

    '''
    kwargs = copy.deepcopy(kwargs)
    mut = getattr(tools, method, None)
    if not mut:
        raise ValueError('In wrap_mutate, method - {} is not in deap.tools'.format(method))
    required_args, _, _ = get_args_kwargs_defaults(mut)
    args = [individual]
    if len(required_args) > 1:
        for a in required_args[1:]:
            if a == 'low':
                args.append([0] * len(choices))
            elif a == 'up':
                args.append([len(choice) - 1 for choice in choices])
            else:
                args.append(kwargs[a])
    for retries in range(max_param_retries):
        params = mut(*args)
        if not out_of_bounds(params[0], choices):
            return params
    raise ValueError('wrap_mutate could not find a set of parameters that is within the given choices for the param_grid')


def init_Individual(icls, content):
    individual = icls(content)
    return individual


def init_Population(pcls, ind_init, initializer, mu):
    pop = []
    while len(pop) < mu:
        pop.append(ind_init(initializer()))
    return pcls(pop)


def crossover(toolbox, method, ind1, ind2, **kwargs):
    '''crossover two solutions using crossover 'method'
    Parameters:
        toolbox: deap toolbox
        ind1:    individual
        ind2:    individual
        method:  method name from deap.tools, e.g. cxTwoPoint
        kwargs:  passed as args where needed to method, e.g.:
            alpha: if using cxBlend, cxESBlend
            indpb: if using cxUniform or cxUniformPartialyMatched
            eta: if using cxSimulatedBinary or cxSimulatedBinaryBounded
            low: if using cxSimulatedBinaryBounded
            up:  if using cxSimulatedBinaryBounded
    '''
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
    cx = getattr(tools, method, None)
    if not cx:
        raise ValueError('{} method (crossover) is not in deap.tools'.format(method))
    required_args, _, _ = get_args_kwargs_defaults(cx)
    args = [child1, child2]
    if len(required_args) > 2:
        for a in required_args[2:]:
            if a not in kwargs:
                raise ValueError('Expected {} to be in control for '
                                 'param_grid with method {}'.format(a, method))
            args.append(kwargs[a])
    child1, child2 = cx(*args)
    del child1.fitness.values
    del child2.fitness.values
    return (child1, child2)


def wrap_select(method, individuals, k, **kwargs):
    '''wraps a selection method such as selNSGA2 from deap.tools

    Parameters:
        method:       method such as selNSGA2 or selBest
        individuals:  population of solutions
        k:            how many to select
        kwargs:       passed as args to method, e.g.:
            fitness_size:   with method selDoubleTournament
            parsimony_size: with method selDoubleTournament
            fitness_first:  with method selDoubleTournament
            tournsize:      with method selTournament

    '''
    sel = getattr(tools, method, None)
    if not sel:
        raise ValueError('Expected {} to be an attribute of deap.tools'.format(method))
    required_args, _, _ = get_args_kwargs_defaults(sel)
    args = [individuals, k]
    if len(required_args) > 2:
        for a in required_args[2:]:
            if not a in kwargs:
                raise ValueError('Expected control kwargs {} to have {} for method {}'.format(kwargs, a, method))
            args.append(kwargs[a])
    return sel(*args)


def _set_from_keys(config_dict, keys, value):
    '''Set a value in a new config given keys like
       ("train", "pca", "model_init_kwargs", "n_components")
    '''
    d = config_dict
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def individual_to_new_config(config, deap_params, ind):
    '''Take an Individual (ind), return new elm.config.ConfigParser instance

    Parameters:
        config: existing elm.config.ConfigParser instance
        deap_params:  deap_params field from EvoParams object
        ind: Individual (list of indices into deap_params in same
             order as param_order)
    Returns:
        new_config: elm.config.ConfigParser instance
    '''
    logger.debug('individual_to_new_config ind: {}'.format(ind))
    zipped = zip(deap_params['param_order'],
                 deap_params['choices'],
                 ind)
    new_config = copy.deepcopy(config.config)
    sample_pipeline = None
    for keys, choices, idx in zipped:
        if keys[0] == 'sample_pipeline_variable':
            sample_pipeline = choices[idx]
            continue
        _set_from_keys(new_config,
                       keys,
                       choices[idx])
    new_config = ConfigParser(config=new_config)
    return new_config


class ParamsSamplingError(ValueError):
    '''Raised when parameters are infeasible after MAX_PARAM_RETRIES'''
    pass


def avoid_repeated_params(max_param_retries):
    '''Keep a running set of hashes to avoid repeating parameter trials

    Parameters:
        max_param_reties: int - max number of retries to find
                          unused param set

    Returns:
        dec:              decorator of mutate, mate, or other functions
                          returning Individuals
    '''
    hashed_params = set()
    def dec(func):
        def wrapper(*args, **kwargs):
            nonlocal hashed_params
            for retries in range(max_param_retries):
                params = func(*args, **kwargs)
                if isinstance(params[0], Sequence):
                    pt = tuple(map(tuple, params))
                else:
                    pt = tuple(params)
                pt = hash(pt)
                if not pt in hashed_params:
                    hashed_params.add(pt)
                    return params
            raise ParamsSamplingError('Cannot find a param set that has not been tried')
        return wrapper
    return dec


def _configure_toolbox(toolbox, deap_params, config, **kwargs):
    '''Configure a deap toolbox for mate, mutate, select,
    Individual init, and population init methods'''
    max_param_retries = getattr(config, 'MAX_PARAM_RETRIES', DEFAULT_MAX_PARAM_RETRIES)
    dec = avoid_repeated_params(max_param_retries)
    toolbox.register('mate', dec(partial(crossover, toolbox,
                                 kwargs['crossover_method'],
                                 **kwargs)))
    toolbox.register('mutate', dec(partial(wrap_mutate,
                                   kwargs['mutate_method'],
                                   deap_params['choices'],
                                   max_param_retries,
                                   **kwargs)))
    toolbox.register('select', partial(wrap_select,
                                       kwargs['select_method'],
                                       **{k:v for k, v in kwargs.items()
                                          if not k == 'k'}))
    toolbox.register('individual_guess',
                     init_Individual,
                     creator.Individual)
    toolbox.register('population_guess',
                     init_Population,
                     list,
                     toolbox.individual_guess,
                     toolbox.individual,
                     kwargs['mu'])
    return toolbox


def _random_choice(choices):
    '''Random choice among indices'''
    return [random.choice(range(len(choice))) for choice in choices]


def _get_evolve_meta(config):
    '''Returns parsed param_grids info or None if not used'''
    param_grid_name_to_deap = {}
    step_name_to_param_grid_name = {}
    for idx, step1 in enumerate(config.pipeline):
        for step in step1['steps']:
            pg = get_param_grid(config, step1, step)
            if pg:
                param_grid_name_to_deap.update(pg)
                idx_name = (idx, step.get('train', step.get('transform')))
                step_name_to_param_grid_name[idx_name] = tuple(pg.keys())[0]
    if not param_grid_name_to_deap:
        return None
    return (step_name_to_param_grid_name, param_grid_name_to_deap)


def ea_setup(config):
    '''Return a dict of parsed EvoParams for each param_grid
    in config\'s param_grids section

    Parameters:
        config:  elm.config.ConfigParser instance
    Returns:
        evo_params_dict:  Dict of EvoParams instances for each param_grid
                          in config\'s param_grids section (if any)

    '''
    out = _get_evolve_meta(config)
    if not out:
        return {}
    step_name_to_param_grid_name, param_grid_name_to_deap = out
    evo_params_dict = {}
    for ((idx, step_name), param_grid_name) in step_name_to_param_grid_name.items():
        deap_params = param_grid_name_to_deap[param_grid_name]
        kwargs = copy.deepcopy(deap_params['control'])
        toolbox = base.Toolbox()
        tt = config.train.get(step_name, config.transform.get(step_name))
        score_weights = None
        if 'model_scoring' in tt:
            ms = config.model_scoring[tt['model_scoring']]
            if 'score_weights' in ms:
                score_weights = ms['score_weights']
        if score_weights is None:
            raise ElmConfigError('Cannot continue EA '
                                 'when score_weights is None (train section of config: '
                                 '{})'.format(step_name))
        creator.create('FitnessMulti', base.Fitness, weights=tuple(score_weights))
        creator.create('Individual', list, fitness=creator.FitnessMulti)
        toolbox.register('indices', _random_choice, deap_params['choices'])
        toolbox.register('individual', tools.initIterate, creator.Individual,
                         toolbox.indices)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        early_stop = deap_params['control'].get('early_stop') or None
        evo_params =  EvoParams(
               toolbox=_configure_toolbox(toolbox, deap_params, config, **kwargs),
               individual_to_new_config=partial(individual_to_new_config,
                                                config,
                                                deap_params),
               deap_params=deap_params,
               param_grid_name=param_grid_name,
               history_file='{}.csv'.format(param_grid_name),
               score_weights=score_weights,
               early_stop=early_stop,
        )
        evo_params_dict[idx] = evo_params
    return evo_params_dict


def evo_init_func(evo_params):
    '''From parsed EvoParams return the initial population'''
    toolbox = evo_params.toolbox
    pop = toolbox.population_guess()
    logger.info('Initialize population of {} solutions (param_grid: '
                '{})'.format(len(pop), evo_params.param_grid_name))
    return pop


def check_fitnesses(fitnesses, score_weights):
    '''Check fitnesses are Sequences of numbers as long as score_weights'''
    is_seq = isinstance(fitnesses, Sequence)
    if not is_seq or not all(len(fitness) == len(score_weights) and all(isinstance(f, numbers.Number) for f in fitness)
                         for fitness in fitnesses):
        raise ValueError('Invalid fitnesses sent to evo_general (not Sequence of numbers or not equal len to score_weights {}): {}'.format(fitnesses, score_weights))

def assign_check_fitness(invalid_ind, fitnesses, param_history, choices, score_weights):
    '''Assign fitness to each individual in invalid_ind, record parameter history

    Parameters:
        invalid_ind:   Individuals Sequence
        fitnesses:     Sequence of fitness Sequences
        param_history: List onto which param choices are appended
        choices:       List of actual param values corresponding to indices in Individuals
        score_weights: List of -1 or 1 ints indicating minimize/maximize
    Returns:
        None (assigns fitness to Individuals in place)
    '''
    check_fitnesses(fitnesses, score_weights)
    for ind, fit in zip(invalid_ind, fitnesses):
        fit = list(fit) if isinstance(fit, Sequence) else [fit]
        ind.fitness.values = fit
        ind_for_history = [choice[i] for i, choice in zip(ind, choices)]
        param_history.append(list(ind_for_history) + list(fit))


def _eval_agg_stop(agg, seq):
    '''Return an agg ("all", "any", or callable) on Sequence seq'''
    if callable(agg):
        return agg(seq)
    if agg == 'all':
        return np.all(seq)
    elif agg == 'any':
        return np.any(seq)
    else:
        raise ValueError('agg: {} argument not in '
                         '("all", "any", <callable>)'.format(agg))


def _check_ints(typ, value, positive_definite=True):
    '''Check all members of Sequence value are ints'''
    if not all(isinstance(val, int) and (val > 0 if positive_definite else True)
               for val in value):
        raise ElmConfigError('With early_stop: {} expected a Sequence of positive ints (indices) but found {}'.format(typ, value))


def _percent_change_stop(agg, score_weights, value, old, new):
    '''Evaluate an early_stop based on percent_change'''
    check_fitnesses([new], score_weights)
    pcent = [(o - n if sw == -1 else n - o) / o * 100.0 > v
             for o, n, sw, v in zip(old, new, score_weights, value)]
    return _eval_agg_stop(agg, pcent)


def _abs_change_stop(agg, score_weights, value, old, new):
    '''Evaluate an early_stop based on abs_change'''
    check_fitnesses([new], score_weights)
    chg = [(o - n if sw == -1 else n - o) > v
           for o, n, sw, v in zip(old, new, score_weights, value)]
    return _eval_agg_stop(agg, chg)


def _threshold_stop(agg, score_weights, value, old, new):
    '''Evaluate an early_stop based on threshold'''
    check_fitnesses([new], score_weights)
    meets = [(n < v if sw == -1 else n > v)
             for n, sw, v in zip(new, score_weights, value)]
    return agg(meets)


def _no_stop(agg, score_weights, value, old, new):
    '''Always returns False (placeholder when "early_stop" not in config)'''
    check_fitnesses([new], score_weights)
    return False


def eval_stop_wrapper(evo_params, original_fitness):
    '''Handle sections of config's {param_grids: {pg_name: control: {

    Examples of configs handled by this function:

         early_stop: {abs_change: [10], agg: all},
         early_stop: {percent_change: [10], agg: all}
         early_stop: {threshold: [10], agg: any}
    }
    Parameters
        evo_params: EvoParams namedtuple
    Returns
        decorator - evaluates stop on each EA iteration
    '''
    early_stop = evo_params.early_stop
    if not early_stop:
        # If not early_stop, this is a do-nothing function
        value = [-1,] * len(evo_params.score_weights)
        func = _no_stop
    elif 'percent_change' in early_stop:
        typ = 'percent_change'
        value = early_stop['percent_change']
        _check_ints(typ, value)
        func = _percent_change_stop
    elif 'threshold' in early_stop:
        typ = 'threshold'
        value = early_stop['threshold']
        _check_ints(typ, value, positive_definite=False)
        func = _threshold_stop
    elif 'abs_change' in early_stop:
        typ = 'abs_change'
        value = early_stop['abs_change']
        _check_ints(typ, value)
        func = _abs_change_stop
    else:
        raise ValueError('early_stop:{} does not have a '
                         'key in ("abs_change", "percent_change", "threshold")'.format(early_stop))
    if not isinstance(value, Sequence):
        raise ElmConfigError('Expected early_stop:{} to be a Sequence'.format(value))
    if not len(evo_params.score_weights) == len(value) == len(original_fitness):
        raise ElmConfigError('Expected score_weights {}, early_stop: {} ({}) '
                             'and fitness {} to all have same '
                             'len()'.format(evo_params.score_weights,
                                            typ, value, original_fitness))
    eval_stop = partial(func,
                        early_stop.get('agg', all),
                        evo_params.score_weights,
                        value,
                        original_fitness)
    return eval_stop


def ea_general(evo_params, cxpb, mutpb, ngen, k):
    '''This is a general EA based
    on an NSGA2 example from deap:

        https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    Parameters:
        toolbox: deap toolbox with select, mate, mutate methods
        pop:     population from toolbox.population_guess() or similar
        cxpb:    crossover prob (float 0 < cxpb < 1)
        mutpb:   mutation prob (float 0 < cxpb < 1)
        ngen:    number of generations (int)
                     (Note: the loop here starts at generation 1 not zero)
        mu:      population size
    '''
    toolbox = evo_params.toolbox
    param_history = []
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = invalid_ind = evo_init_func(evo_params)

    deap_params = evo_params.deap_params

    assign_names(pop)
    assert isinstance(pop, list)
    assert isinstance(pop[0][0], int)
    fitnesses = (yield (pop, invalid_ind, param_history))
    assign_check_fitness(invalid_ind, fitnesses,
                     param_history, deap_params['choices'],
                     evo_params.score_weights)
    # Forces assignment of crowding distance for
    # NSGA2 - no selection is done here
    pop = toolbox.select(pop, len(pop))
    # Find the best in original population for
    # comparison on stop conditions
    temp_pop = copy.deepcopy(pop)
    original_fitness = toolbox.select(temp_pop, 1)[0].fitness.values
    del temp_pop
    eval_stop = eval_stop_wrapper(evo_params, original_fitness)
    assert isinstance(invalid_ind, list)
    assert isinstance(invalid_ind[0][0], int)
    assert all(ind.fitness.valid for ind in pop)
    for gen in range(1, ngen):
        logger.info('Generation {} out of {} in evolutionary '
                    'algorithm'.format(gen + 1, ngen))
        offspring1 = tools.selTournamentDCD(pop, len(pop))

        offspring2 = [toolbox.clone(ind) for ind in offspring1]
        offspring3 = []
        for off1, off2 in zip(offspring1, offspring2):
            off2.name = off1.name
            offspring3.append(off2)
        offspring = offspring3
        try:
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= cxpb:
                    toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
        except ParamsSamplingError:
            logger.info('Evolutionary algorithm exited early (cannot find parameter set that has not been tried yet)')
            break
        # Evaluate the individuals with an invalid fitness

        # Expect the fitnesses to be sent here
        # with ea_gen.send(fitnesses)
        assert isinstance(invalid_ind, list)
        assert isinstance(invalid_ind[0][0], int)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        assign_names(invalid_ind)
        assert invalid_ind
        fitnesses = (yield (pop, invalid_ind, param_history))
        # Assign the fitnesses to the invalid_ind
        # evaluated
        assign_check_fitness(invalid_ind, fitnesses,
                         param_history, deap_params['choices'],
                         evo_params.score_weights)
        break_outer = False
        for fitness in fitnesses:
            if eval_stop(fitness):
                logger.info('Stopping: early_stop: {}'.format(evo_params.early_stop))
                break_outer = True
                break
        if break_outer:
            break
        # Select the next generation population
        pop = toolbox.select(pop + offspring, len(pop))
        #logger.info(logbook.stream)
    # Yield finally the record and logbook
    # The caller knows when not to .send again
    # based on the None in 2nd position below
    logger.info('Evolutionary algorithm finished')
    yield (pop, None, param_history)

