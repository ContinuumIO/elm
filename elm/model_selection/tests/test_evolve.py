from __future__ import absolute_import, division, print_function, unicode_literals

import copy
from io import StringIO
from itertools import product

import pytest
from sklearn.cluster import MiniBatchKMeans
import yaml

from elm.config import ConfigParser, ElmConfigError
from elm.config.tests.fixtures import *
from elm.model_selection.evolve import (get_param_grid,
                                        ind_to_new_pipe,
                                        _get_evolve_meta,
                                        EvoParams,
                                        ea_setup,
                                        evo_init_func,
                                        ea_general,
                                        assign_check_fitness)
from elm.model_selection.tests.evolve_example_config import CONFIG_STR



def _setup(config=None):
    '''Return the config above and the param_grid'''
    from elm.sample_util.sample_pipeline import make_pipeline_steps
    from elm.pipeline import Pipeline
    if not config:
        config = ConfigParser(config=yaml.load(CONFIG_STR))
    sample_steps = make_pipeline_steps(config, config.run[0]['pipeline'])
    estimator = [('kmeans', MiniBatchKMeans(**config.train['kmeans']['model_init_kwargs']))]
    pipe = Pipeline(sample_steps + estimator)
    idx_to_param_grid = ea_setup(config)
    return config, pipe, idx_to_param_grid


@pytest.mark.xfail # TODO - when elm-main is not deprecated, remove this decorator
def test_parse_param_grid():
    '''Tests that param_grid is parsed from config and
    metadata about parameters is extracted'''
    config, pipe, param_grid = _setup()

    k, v = tuple(param_grid.items())[0]
    assert k == 0
    for k2 in ('is_int', 'low', 'up', 'choices',):
        assert isinstance(v.deap_params.get(k2), list)
    assert all(isinstance(vi, list) for vi in v.deap_params['choices'])


@pytest.mark.xfail # TODO - when elm-main is not deprecated, remove this decorator
def test_ind_to_new_pipe():
    '''In the evolutionary algo, an individual consists
    of indices which can be used in the param_grid choices
    lists to find the corresponding parameter.  This test
    ensures that the indices can be used to make replacements
    in a config, so they actually have a model effect'''
    config, pipe, idx_to_param_grid = _setup()

    param_grid_item = idx_to_param_grid[0]
    ind = [0,] * len(param_grid_item.deap_params['choices'])
    pipe = ind_to_new_pipe(pipe, param_grid_item.deap_params, ind)
    p = pipe.get_params()
    assert p['kmeans__n_clusters'] == 3
    assert p['pca__n_components'] == 2

    ind = [1,] * len(param_grid_item.deap_params['choices'])
    pipe = ind_to_new_pipe(pipe, param_grid_item.deap_params, ind)
    p = pipe.get_params()
    assert p['kmeans__n_clusters'] == 4
    assert p['pca__n_components'] == 3


@pytest.mark.xfail # TODO - when elm-main is not deprecated, remove this decorator
def tst_evo_setup_evo_init_func(config=None):
    '''Tests that param_grid is parsed and the deap toolbox
    is created ok so that toolbox.population_guess can
    be called without error.'''
    (config, pipe, idx_to_param_grid) = _setup(config=config)
    assert isinstance(idx_to_param_grid, dict) and len(idx_to_param_grid) == 1
    evo_params = idx_to_param_grid[0]
    assert isinstance(evo_params, EvoParams)
    return config, evo_params


test_evo_setup_evo_init_func = tst_evo_setup_evo_init_func

# The following are test data to test_ea_general
min_fitnesses = [(50,), (0,)] + [(100,)] * 22
max_fitnesses = [(50,), (1000,)] + [(100,)] * 22
min_max_fitnesses = [(50, 100), (0, 1000)] + [(100, 50)] * 22
score_weights = [[-1,], [1,], [-1, 1]]
zipped_tst_args = zip((min_fitnesses, max_fitnesses, min_max_fitnesses),
                      score_weights)

@pytest.mark.xfail # TODO - when elm-main is not deprecated, remove this decorator
@pytest.mark.parametrize('fitnesses, score_weights', zipped_tst_args)
def test_ea_general(fitnesses, score_weights):
    '''This test ensures that ea_general, a general
    EA can minimize, maximize or
    handle multiple objectives.  In each of the fitnesses
    sequences passed in, the 2nd individual is known
    to be the most fit (given corresponding score_weights
    which determine whether min/maximizing the objective(s))
    '''
    config = yaml.load(CONFIG_STR)
    config['model_scoring']['testing_model_scoring']['score_weights'] = score_weights
    config['param_grids']['example_param_grid']['control']['early_stop'] = {'abs_change': [100,] * len(score_weights)}
    config, evo_params = tst_evo_setup_evo_init_func(config=ConfigParser(config=config))
    control = evo_params.deap_params['control']
    param_history = []
    ea_gen = ea_general(evo_params,
                        control['cxpb'],
                        control['mutpb'],
                        control['ngen'],
                        control['k'])

    pop, _, _ = next(ea_gen)
    for ind in pop:
        assert isinstance(ind, list)
    invalid_ind = pop
    assert len(pop) == control['mu']
    original_pop = copy.deepcopy(pop)
    best = fitnesses[1]  # in this synthetic data,
                            # the 2nd param set is always best
    assign_check_fitness(pop, fitnesses,
                     param_history, evo_params.deap_params['choices'],
                     evo_params.score_weights)
    while invalid_ind:
        (pop, invalid_ind, param_history) = ea_gen.send(fitnesses)
    matches_best = tuple(ind for ind in pop if ind.fitness.values == best)
    assert matches_best
    assert original_pop != pop


def set_key_tst_bad_config_once(key, bad):
    config2 = yaml.load(CONFIG_STR)
    d = config2
    for k in key[:-1]:
        d = d[k]
    d[key[-1]] = bad
    with pytest.raises(ElmConfigError):
        ConfigParser(config=config2)

# Below are parameters that are zipped
# together to form examples of bad
# configs. (key to set in config, value to set there)
control_key = ('param_grids', 'example_param_grid', 'control')
dict_keys = (
        control_key,
        control_key + ('early_stop',),
        control_key[:1],
        control_key[:2],
    )

bad_param = [
        ('pca_n_components'), # not double underline after pca
        'kmeans__not_in_init_kwargs', # not a valid init arg to kmeans
        ('model_init_kwargs', 'n_clusters'),
        9,
        ]
not_dicts = (9, 9.1, [2,3])
not_int = ({},[], 9.1, [1,3])
bad_control = [{}, [], 9, None, 'cant be string', [9]]
int_keys = tuple(control_key + (k,)
                 for k in ('mu', 'k', 'ngen'))
not_int = (9.2, None,  [2], [], {}, {7:2})
tst_params = list(zip(int_keys, not_int)) + list(zip(dict_keys, not_dicts)) + \
             list(zip((control_key,) * len(bad_control), bad_control))
tst_params = [t for t in tst_params
              if not ('early_stop' in t[0] and not t[1])]
@pytest.mark.xfail # TODO - when elm-main is not deprecated, remove this decorator
@pytest.mark.parametrize('key, value', tst_params)
def test_bad_param_grid_config(key, value):
    set_key_tst_bad_config_once(key, value)


