from __future__ import absolute_import, division, print_function

'''
----------------------------

``elm.model_selection.evolve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import array
from collections import (defaultdict, OrderedDict, Sequence)
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
from sklearn.model_selection import ParameterGrid

from xarray_filters.func_signatures import get_args_kwargs_defaults
from elm.config import (import_callable,
                        ElmConfigError,
                        ConfigParser)

logger = logging.getLogger(__name__)

DEFAULT_PERCENTILES = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)

EVO_FIELDS = ['toolbox',
              'deap_params',
              'param_grid_name',
              'history_file',
              'score_weights',
              'early_stop']


DEFAULT_MAX_PARAM_RETRIES = 1000

LAST_TAG_IDX = 0

DEFAULT_CONTROL = {
      'select_method': 'selNSGA2',
      'crossover_method': 'cxTwoPoint',
      'mutate_method': 'mutUniformInt',
      'init_pop': 'random',
      'indpb': 0.5,
      'mutpb': 0.9,
      'cxpb':  0.3,
      'eta':   20,
      'ngen':  2,
      'mu':    4,
      'k':     4,
      'early_stop': None
      # {'abs_change': [10], 'agg': all},
      # alternatively 'early_stop': {'percent_change': [10], 'agg': all}
      # alternatively 'early_stop': {'threshold': [10], 'agg': any}
    }

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

DEFAULT_EVO_PARAMS = dict(
    k=32,
    mu=32,
    ngen=3, cxpb=0.3, indpb=0.5, mutpb=0.9, eta=20,
    param_grid_name='param_grid',
    select_method='selNSGA2',
    crossover_method='cxTwoPoint',
    mutate_method='mutUniformInt',
    init_pop='random',
    early_stop=None,
    toolbox=None
)

def _call_rvs(choice):
    param = choice.rvs()
    if param.dtype.kind == 'f':
        param = float(param)
    else:
        param = int(param)
    return param


def _random_choice(choices):
    '''Random choice among indices'''
    params = []
    for choice in choices:
        if hasattr(choice, 'rvs'):
            param = _call_rvs(choice)
        else:
            param = int(np.random.choice(range(len(choice))))
        params.append(param)
    return params


def next_model_tag():
    '''Gives names like tag_0, tag_1, tag_2 sequentially'''
    global LAST_TAG_IDX
    model_name =  'evolve_{}'.format(LAST_TAG_IDX)
    LAST_TAG_IDX += 1
    return model_name

def assign_names(pop, names=None):
    '''Assign names to the models in the population'''
    names = names or [next_model_tag() for _ in pop]
    for ind, name in zip(pop, names):
        ind.name = name


def check_format_param_grid(param_grid, control, param_grid_name='param_grid_0'):
    '''Run validations on a param_grid'''
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
    return _to_param_meta(param_grid, control)


def _to_param_meta(param_grid, control):
    '''Acquire parameter metadata such as bounds that are useful for sampling'''
    choice_params = {k: v for k, v in param_grid.items()
                     if not hasattr(v, 'rvs')}
    distributions = {k: v for k, v in param_grid.items()
                     if k not in choice_params}
    pg_list = list(ParameterGrid(choice_params))
    choices, low, high, param_order, is_int = [], [], [], [], []
    is_continuous = lambda v: isinstance(v, numbers.Real)
    while len(pg_list):
        pg2 = pg_list.pop(0)
        for k, v in pg2.items():
            if k in param_order:
                idx = param_order.index(k)
            else:
                idx = len(param_order)
                param_order.append(k)
                low.append(v)
                high.append(v)
                choices.append([v])
                is_int.append(not is_continuous(v))
                continue
            if v not in choices[idx]:
                choices[idx].append(v)
            if is_continuous(v):
                is_int[idx] = False
                if v < low[idx]:
                    low[idx] = v
                if v > high[idx]:
                    high[idx] = v
            else:
                is_int[idx] = True
                low[idx] = high[idx] = v

    for k, v in distributions.items():
        choices.append(v)
        low.append(None)
        high.append(None)
        is_int.append(False)
        param_order.append(k)
    param_meta = dict(control=control, high=high, low=low,
                      choices=choices, is_int=is_int,
                      param_order=param_order)
    return param_meta


def out_of_bounds(params, choices):
    '''Check boudns on indices into choices lists'''
    for p, choice in zip(params, choices):
        if hasattr(choice, 'rvs'):
            continue
        if p < 0 or p >= len(choice):
            return True
    return False


def wrap_mutate(method, choices, max_param_retries, individual, **kwargs):
    '''Mutation for the method, choices and other config options

    Parameters:
        :method:  string - imported from deap.tools such as mutUniformInt
        :choices: list of lists choices for each parameter
        :max_param_retries: how many times to retry when getting invalid params
        :individual:        deap Individual parameter ste
        :kwargs:            kwargs passed as args to method given

    Returns:
        :tuple: of one Individual parameter set

    '''
    kwargs = copy.deepcopy(kwargs)
    if not callable(method):
        mut = getattr(tools, method, None)
        if not mut:
            raise ValueError('In wrap_mutate, method - {} is not in deap.tools'.format(method))
    else:
        mut = method
    required_args, _, _ = get_args_kwargs_defaults(mut)
    args = [individual]
    if len(required_args) > 1:
        for a in required_args[1:]:
            if a == 'low':
                args.append([0] * len(choices))
            elif a == 'up':
                args.append([(2 ** 32 if hasattr(choice, 'rvs') else len(choice) - 1)
                             for choice in choices])
            else:
                args.append(kwargs[a])
    for retries in range(max_param_retries):
        params = mut(*args)
        params2 = []
        for idx, (p, choice) in enumerate(zip(params[0], choices)):
            if hasattr(choice, 'rvs'):
                p = choice.rvs()
            if callable(choice):
                p = choice()
            params2.append(p)
        for idx in range(len(params2)):
            params[0][idx] = params2[idx]
        if not out_of_bounds(params[0], choices):
            return params
    raise ValueError('wrap_mutate could not find a set of parameters that is within the given choices for the param_grid')


def init_Individual(icls, content):
    '''Initialize a genetic algorithm individual with class icls and content (parameters)'''
    individual = icls(content)
    return individual


def init_Population(pcls, ind_init, initializer, mu):
    '''Initialize a population of size mu'''
    pop = []
    while len(pop) < mu:
        pop.append(ind_init(initializer()))
    return pcls(pop)


def crossover(toolbox, method, ind1, ind2, **kwargs):
    '''crossover two solutions using crossover 'method'

    Parameters:
        :toolbox: deap toolbox
        :ind1:    individual
        :ind2:    individual
        :method:  method name from deap.tools, e.g. cxTwoPoint
        :kwargs:  passed as args where needed to method, e.g.:

            * :alpha: if using cxBlend, cxESBlend
            * :indpb: if using cxUniform or cxUniformPartialyMatched
            * :eta: if using cxSimulatedBinary or cxSimulatedBinaryBounded
            * :low: if using cxSimulatedBinaryBounded
            * :up:  if using cxSimulatedBinaryBounded

    '''
    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]

    if not callable(method):
        cx = getattr(tools, method, None)
        if cx is None:
            raise ValueError('{} method (crossover) is not in deap.tools'.format(method))
    else:
        cx = method
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
        :method:       method such as selNSGA2 or selBest
        :individuals:  population of solutions
        :k:            how many to select
        :kwargs:       passed as args to method, e.g.:

            * :fitness_size:   with method selDoubleTournament
            * :parsimony_size: with method selDoubleTournament
            * :fitness_first:  with method selDoubleTournament
            * :tournsize:      with method selTournament

    '''

    if not callable(method):
        sel = getattr(tools, method, None)
        if sel is None:
            raise ValueError('Expected {} to be an attribute of deap.tools'.format(method))
    else:
        sel = method
    required_args, _, _ = get_args_kwargs_defaults(sel)
    args = [individuals, k]
    if len(required_args) > 2:
        for a in required_args[2:]:
            if not a in kwargs:
                raise ValueError('Expected control kwargs {} to have {} for method {}'.format(kwargs, a, method))
            args.append(kwargs[a])
    return sel(*args)


def _set_from_keys(config_dict, keys, value):
    '''Set a value in a new config given keys like /
       ("train", "pca", "model_init_kwargs", "n_components")
    '''
    d = config_dict
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def ind_to_new_params(deap_params, ind):
    '''Use deap_params' info to map ind's indices to parameter
    choices

    Parameters:
        :deap_params: deap parameters
        :ind:         Individual (list of indices into deap_params /
                      in same order as deap_params['param_order'])
    Returns:
        :new_params:  Unfitted copy of original elm.pipeline.Pipeline /
                      with parameter replacements
    '''
    zipped = zip(deap_params['param_order'],
                 deap_params['choices'],
                 ind)
    new_params = {}
    for key, choices, idx in zipped:
        if hasattr(choices, 'rvs'):
            new_params[key] = idx
        else:
            new_params[key] = choices[idx]
    return new_params


class ParamsSamplingError(ValueError):
    '''Raised when parameters are infeasible after MAX_PARAM_RETRIES'''
    pass


def avoid_repeated_params(max_param_retries):
    '''Keep a running set of hashes to avoid repeating parameter trials

    Parameters:
        :max_param_reties: int - max number of retries to find unused param set

    Returns:
        :dec: decorator of mutate, mate, or other functions returning Individuals
    '''
    hashed_params = set()
    def dec(hashed_params, func):
        def wrapper(hashed_params, *args, **kwargs):
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
        return partial(wrapper, hashed_params)
    return partial(dec, hashed_params)


def _configure_toolbox(deap_params, **kwargs):
    '''Configure a deap toolbox for mate, mutate, select, Individual init, and population init methods'''
    kwargs = kwargs.copy()
    control = deap_params['control']
    kwargs.update({k: v for k, v in control.items() if k not in kwargs})
    toolbox = kwargs.pop('toolbox')
    max_param_retries = kwargs.get('max_param_retries', DEFAULT_MAX_PARAM_RETRIES)
    dec = avoid_repeated_params(max_param_retries)
    toolbox.register('mate', dec(partial(crossover, toolbox,
                                 control['crossover_method'],
                                 **kwargs)))
    toolbox.register('mutate', dec(partial(wrap_mutate,
                                   control['mutate_method'],
                                   deap_params['choices'],
                                   max_param_retries,
                                   **kwargs)))
    toolbox.register('select', partial(wrap_select,
                                       control['select_method'],
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
                     control['mu'])
    return toolbox


def fit_ea(score_weights,
           control,
           param_grid,
           param_grid_name='param_grid',
           early_stop=None,
           toolbox=None):
    if score_weights is None:
        score_weights = (1,)
    control_defaults = {k: v for k, v in copy.deepcopy(DEFAULT_CONTROL).items()
                        if control.get(k, None) is None}
    control.update(control_defaults)
    deap_params = check_format_param_grid(param_grid, control)
    if toolbox is None:
        control['toolbox'] = toolbox = base.Toolbox()

    creator.create('FitnessMulti', base.Fitness, weights=tuple(score_weights))
    creator.create('Individual', list, fitness=creator.FitnessMulti)
    toolbox.register('indices', _random_choice, deap_params['choices'])
    toolbox.register('individual', tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    evo_params = dict(
        toolbox=_configure_toolbox(deap_params, **control),
        deap_params=deap_params,
        param_grid_name=param_grid_name,
        history_file='{}.csv'.format(param_grid_name),
        score_weights=score_weights,
        early_stop=early_stop,
    )
    ea_gen = ea_general(evo_params, **control)
    pop, _, _ = next(ea_gen)
    return pop, toolbox, ea_gen, evo_params


def evo_init_func(evo_params):
    '''From ea parameters return the initial population'''
    toolbox = evo_params['toolbox']
    pop = toolbox.population_guess()
    logger.info('Initialize population of {} solutions (param_grid: '
                '{})'.format(len(pop), evo_params['param_grid_name']))
    return pop


def check_fitnesses(fitnesses, score_weights):
    '''Check fitnesses are Sequences of numbers as long as score_weights'''
    is_seq = isinstance(fitnesses, Sequence)
    if not is_seq or not all(len(fitness) == len(score_weights) and all(isinstance(f, numbers.Number) for f in fitness)
                         for fitness in fitnesses):
        raise ValueError('Invalid fitnesses sent to evo_general (not Sequence of numbers or not equal len with score_weights {}): {}'.format(fitnesses, score_weights))


def assign_check_fitness(invalid_ind, fitnesses, param_history, choices, score_weights):
    '''Assign fitness to each individual in invalid_ind, record parameter history

    Parameters:
        :invalid_ind:   Individuals Sequence
        :fitnesses:     Sequence of fitness Sequences
        :param_history: List onto which param choices are appended
        :choices:       List of actual param values corresponding to indices in Individuals
        :score_weights: List of -1 or 1 ints indicating minimize/maximize

    Returns:
        :None: (assigns fitness to Individuals in place)
    '''
    check_fitnesses(fitnesses, score_weights)
    for ind, fit in zip(invalid_ind, fitnesses):
        fit = list(fit) if isinstance(fit, Sequence) else [fit]
        ind.fitness.values = fit
        ind_for_history = [(i if hasattr(choice, 'rvs') else choice[i])
                           for i, choice in zip(ind, choices)]
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


def _check_number(typ, value, positive_definite=True):
    '''Check all members of Sequence value are ints'''
    if not all(isinstance(val, numbers.Number) and (val > 0 if positive_definite else True)
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
    '''Handle sections of config's {param_grids: {pg_name: control: {}}

    Examples of configs handled by this function:

         * :early_stop: {abs_change: [10], agg: all},
         * :early_stop: {percent_change: [10], agg: all}
         * :early_stop: {threshold: [10], agg: any}

    Parameters:
        :evo_params: EA parameters

    Returns:
        :decorator: Evaluates stop on each EA iteration
    '''
    early_stop = evo_params['early_stop']
    if not early_stop:
        # If not early_stop, this is a do-nothing function
        value = [-1,] * len(evo_params['score_weights'])
        func = _no_stop
    elif 'percent_change' in early_stop:
        typ = 'percent_change'
        value = early_stop['percent_change']
        _check_number(typ, value)
        func = _percent_change_stop
    elif 'threshold' in early_stop:
        typ = 'threshold'
        value = early_stop['threshold']
        _check_number(typ, value, positive_definite=False)
        func = _threshold_stop
    elif 'abs_change' in early_stop:
        typ = 'abs_change'
        value = early_stop['abs_change']
        _check_number(typ, value)
        func = _abs_change_stop
    else:
        raise ValueError('early_stop:{} does not have a '
                         'key in ("abs_change", "percent_change", "threshold")'.format(early_stop))
    if not isinstance(value, Sequence):
        raise ElmConfigError('Expected early_stop:{} to be a Sequence'.format(value))
    if not len(evo_params['score_weights']) == len(value) == len(original_fitness):
        raise ElmConfigError('Expected score_weights {}, early_stop: {} ({}) '
                             'and fitness {} to all have same '
                             'len()'.format(evo_params['score_weights'],
                                            typ, value, original_fitness))
    if not early_stop:
        agg = all
    else:
        early_stop.get('agg', all)
    eval_stop = partial(func,
                        agg,
                        evo_params['score_weights'],
                        value,
                        original_fitness)
    return eval_stop


def ea_general(evo_params, cxpb, mutpb, ngen, k, **kw):
    '''This is a general EA based on an NSGA2 example from deap: /
        https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py

    Parameters:
        :evo_params: TODO docs (has toolbox and deap_params)
        :pop:     population from toolbox.population_guess() or similar
        :cxpb:    crossover prob (float 0 < cxpb < 1)
        :mutpb:   mutation prob (float 0 < cxpb < 1)
        :ngen:    number of generations (int) /
                     (Note: the loop here starts at generation 1 not zero)
        :k:       number to select
    '''
    toolbox = evo_params['toolbox']
    param_history = []
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = invalid_ind = evo_init_func(evo_params)

    deap_params = evo_params['deap_params']

    assign_names(pop)
    fitnesses = (yield (pop, invalid_ind, param_history))
    assign_check_fitness(invalid_ind, fitnesses,
                     param_history, deap_params['choices'],
                     evo_params['score_weights'])
    # Forces assignment of crowding distance for
    # NSGA2 - no selection is done here
    pop = toolbox.select(pop, len(pop))
    # Find the best in original population for
    # comparison on stop conditions
    temp_pop = copy.deepcopy(pop)
    original_fitness = toolbox.select(temp_pop, 1)[0].fitness.values
    del temp_pop
    eval_stop = eval_stop_wrapper(evo_params, original_fitness)
    for gen in range(1, ngen):
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
                if random.random() < mutpb:
                    toolbox.mutate(ind1)
                if random.random() < mutpb:
                    toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

        except ParamsSamplingError:
            logger.info('Evolutionary algorithm exited early (cannot find parameter set that has not been tried yet)')
            break
        # Evaluate the individuals with an invalid fitness

        # Expect the fitnesses to be sent here
        # with ea_gen.send(fitnesses)
        if not isinstance(invalid_ind, list) or not invalid_ind or not invalid_ind[0]:
            raise ValueError('Expected .send to be called with a list of tuples/lists of ints {}'.format(invalid_ind[0] if invalid_ind else invalid_ind))
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        assign_names(invalid_ind)
        fitnesses = (yield (pop, invalid_ind, param_history))
        # Assign the fitnesses to the invalid_ind
        # evaluated
        assign_check_fitness(invalid_ind, fitnesses,
                         param_history, deap_params['choices'],
                         evo_params['score_weights'])
        break_outer = False
        for fitness in fitnesses:
            if eval_stop(fitness):
                logger.info('Stopping: early_stop: {}'.format(evo_params['early_stop']))
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
