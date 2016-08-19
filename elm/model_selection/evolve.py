import array
from collections import defaultdict, OrderedDict, namedtuple
import copy
from functools import partial
import logging
import numbers
import random
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
              'param_grid_name',]
EvoParams = namedtuple('EvoParams', EVO_FIELDS)

OK_PARAM_FIRST_KEYS = ['transform',
                       'train',
                       'feature_selection',
                       'model_scoring',
                       'data_sources',
                       'sklearn_preprocessing',
                       'sample_pipeline',]

def _check_key(train_config, transform_config,
              train_step_name, transform_step_name,
              config):
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
        print(k, default_kwargs, default_transform_kwargs, len(k))
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

    def make_check_tuple_keys(k, v):
        print(make_check_tuple_keys, k, v)
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
    return make_check_tuple_keys


def parse_param_grid_items(param_grid_name, train_config, transform_config,
                          make_check_tuple_keys, param_grid):
    transform_config = transform_config or {}
    def switch_types(current_key, obj):
        print('c,o', current_key, obj)
        if isinstance(obj, (list, tuple)):
            print('c, o2', current_key, obj)
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
        k2, v2 = make_check_tuple_keys(*unwound)
        print('k2,v2', k2, v2)
        param_grid_parsed[k2] = v2
        param_order.append(k2)
    param_grid_parsed['param_order'] = param_order
    return param_grid_parsed


def get_param_grid(config, step):
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
        raise ValueError('Expected param_grid to be used with a "train" '
                         'or "transform" step of a pipeline, but found param_grid '
                         'was used in step {}'.format(step))
    if 'sample_pipeline' in step and transform_step_name is None:
        sample_pipeline = step['sample_pipeline']
        transform_steps = [_ for _ in sample_pipeline if 'transform' in _]
        transform_names = set(_.get('transform') for _ in transform_steps)
        assert len(transform_names) <= 1
        if transform_names:
            transform_step_name = tuple(transform_names)[0]
            transform_config = config.transform[transform_step_name]

    param_grid = config.param_grids[param_grid_name]
    make_check_tuple_keys = _check_key(train_config,
                                       transform_config,
                                       train_step_name,
                                       transform_step_name,
                                       config)
    param_grid = parse_param_grid_items(param_grid_name,
                                        train_config,
                                        transform_config,
                                        make_check_tuple_keys,
                                        param_grid)
    param_grid['control']['step'] = step
    assert all(isinstance(v, list) for k, v in param_grid.items() if k != 'control'), (repr(param_grid))
    if not 'control' in param_grid:
        raise ValueError('Expected a control dict in param_grid:{}'.format(param_grid_name))
    return {param_grid_name: _to_param_meta(param_grid)}


def _to_param_meta(param_grid):
    print(param_grid)
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


def wrap_mutate(method, individual, **kwargs):
    mut = getattr(tools, method, None)
    if not mut:
        raise ValueError('In wrap_mutate, method - {} is not in deap.tools'.format(method))
    required_args, _, _ = get_args_kwargs_defaults(mut)
    args = [individual]
    if len(required_args) > 1:
        for a in required_args[1:]:
            args.append(kwargs[a])
    params = mut(*args)
    return params,


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


def wrap_select(method, individuals, **kwargs):
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
    args = [individuals]
    if len(required_args) > 1:
        for a in required_args[1:]:
            if not a in kwargs:
                raise ValueError('Expected control kwargs {} to have {} for method {}'.format(kwargs, a, method))
            args.append(kwargs[a])
    return sel(*args)


def _set_from_keys(config_dict, keys, value):
    d = config_dict
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def individual_to_new_config(config, deap_params, ind):

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
    if sample_pipeline is not None:
        idx = [idx for idx, s in enumerate(new_config.pipeline)
               if s == deap_params['control']['step']][0]
        new_sp = new_config.sample_pipelines[sample_pipeline]
        new_config.pipeline[idx]['sample_pipeline'] = new_sp
    return new_config


def _configure_toolbox(toolbox, **kwargs):
    toolbox.register('mate', partial(crossover, toolbox,
                                     kwargs['crossover_method'],
                                     **kwargs))
    toolbox.register('mutate', partial(wrap_mutate,
                                       kwargs['mutate_method'],
                                       **kwargs))
    toolbox.register('select', partial(wrap_select,
                                       kwargs['select_method'],
                                       **kwargs))
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
    return [random.choice(range(len(choice))) for choice in choices]


def _get_evolve_meta(config):
    param_grid_name_to_deap = {}
    step_name_to_param_grid_name = {}
    for idx, step in enumerate(config.pipeline):
        pg = get_param_grid(config, step)
        if pg:
            param_grid_name_to_deap.update(pg)
            idx_name = (idx, step.get('train', step.get('transform')))
            step_name_to_param_grid_name[idx_name] = tuple(pg.keys())[0]
    if not param_grid_name_to_deap:
        return None
    return (step_name_to_param_grid_name, param_grid_name_to_deap)


def evolve_setup(config):
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
            raise ElmConfigError('Cannot continue evolutionary algorithm '
                                 'when score_weights is None (train section of config: '
                                 '{})'.format(step_name))
        creator.create('FitnessMulti', base.Fitness, weights=tuple(score_weights))
        creator.create('Individual', list, fitness=creator.FitnessMulti)
        toolbox.register('indices', _random_choice, deap_params['choices'])
        toolbox.register('individual', tools.initIterate, creator.Individual,
                         toolbox.indices)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        evo_params =  EvoParams(
               toolbox=_configure_toolbox(toolbox, **kwargs),
               individual_to_new_config=partial(individual_to_new_config,
                                                config,
                                                deap_params),
               deap_params=deap_params,
               param_grid_name=param_grid_name,
        )
        evo_params_dict[idx] = evo_params
    return evo_params_dict


def evo_init_func(evo_params):
    toolbox = evo_params.toolbox
    pop = toolbox.population_guess()
    logger.info('Initialize population of {} solutions (param_grid: '
                '{})'.format(len(pop), evo_params.param_grid_name))
    return pop


def evo_general(toolbox, pop, cxpb, mutpb, ngen):
    '''This is a general evolutionary algorithm based
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
    yield None # dummy to initialize
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('std', np.std, axis=0)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = ('gen', 'evals', 'std', 'min', 'avg', 'max')
    for gen in range(1, ngen): # starts at 1 because it assumes
                               # already evaluated once
        logger.info('Generation {} out of {} in evolutionary '
                    'algorithm'.format(gen + 1, ngen))
        offspring = toolbox.select(pop)
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness

        # Expect the fitnesses to be sent here
        # with ea_gen.send(fitnesses)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = (yield (pop, invalid_ind, None, None))

        logger.info('Fitnesses {}'.format(fitnesses))
        # Assign the fitnesses to the invalid_ind
        # evaluated
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        logger.info(logbook.stream)
    # Yield finally the record and logbook
    # The caller knows when not to .send again
    # based on the None in 2nd position below
    logger.info('Evolutionary algorithm finished')
    yield (pop, None, record, logbook)

