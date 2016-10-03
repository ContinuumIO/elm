import copy
from io import StringIO
from itertools import product

import pytest
import yaml

from elm.config import ConfigParser, ElmConfigError
from elm.model_selection.evolve import (get_param_grid,
                                        individual_to_new_config,
                                        _get_evolve_meta,
                                        EvoParams,
                                        ea_setup,
                                        evo_init_func,
                                        ea_general,
                                        assign_check_fitness)
from elm.model_selection.tests.evolve_example_config import CONFIG_STR

def _setup():
    '''Return the config above and the param_grid'''
    config = ConfigParser(config=yaml.load(CONFIG_STR))
    param_grid = get_param_grid(config, config.pipeline[0],config.pipeline[0]['steps'][0])
    return config, param_grid


def _setup_pg(config=None):
    '''Create the mapping of step name (e.g kmeans) to
    param_grid'''
    if config is None:
        config, param_grid = _setup()
    out = _get_evolve_meta(config)
    assert len(out) == 2
    step_name_to_param_grid_name, param_grid_name_to_deap = out
    return (config,
            step_name_to_param_grid_name,
            param_grid_name_to_deap)


def test_parse_param_grid():
    '''Tests that param_grid is parsed from config and
    metadata about parameters is extracted'''
    config, param_grid = _setup()
    k, v = tuple(param_grid.items())[0]
    assert k == 'example_param_grid'
    for k2 in ('is_int', 'low', 'up', 'choices',):
        assert isinstance(v[k2], list)
    assert all(isinstance(vi, list) for vi in v['choices'])


def test_individual_to_new_config():
    '''In the evolutionary algo, an individual consists
    of indices which can be used in the param_grid choices
    lists to find the corresponding parameter.  This test
    ensures that the indices can be used to make replacements
    in a config, so they actually have a model effect'''
    config, param_grid = _setup()
    minimal = config.sample_pipelines['minimal']
    top_n = config.sample_pipelines['top_n']
    param_grid_item = param_grid['example_param_grid']
    ind = [0,] * len(param_grid_item['choices'])
    new_config = individual_to_new_config(config, param_grid_item, ind)
    assert new_config.train['kmeans']['model_init_kwargs']['n_clusters'] == 3
    assert new_config.transform['pca']['model_init_kwargs']['n_components'] == 2
    assert new_config.pipeline[0]['sample_pipeline'] == minimal
    assert new_config.feature_selection['top_n']['kwargs']['percentile'] == 30
    ind = [1,] * len(param_grid_item['choices'])
    new_config = individual_to_new_config(config, param_grid_item, ind)
    assert new_config.train['kmeans']['model_init_kwargs']['n_clusters'] == 4
    assert new_config.transform['pca']['model_init_kwargs']['n_components'] == 3
    assert new_config.feature_selection['top_n']['kwargs']['percentile'] == 40


def test_get_evolve_meta():
    '''Tests param_grid metadata'''
    (config,
     step_name_to_param_grid_name,
     param_grid_name_to_deap) = _setup_pg()
    expected_key = ((0, 0), 'kmeans')
    assert expected_key in step_name_to_param_grid_name
    assert step_name_to_param_grid_name[expected_key] in param_grid_name_to_deap
    pg = param_grid_name_to_deap[step_name_to_param_grid_name[expected_key]]
    assert isinstance(pg, dict)
    assert 'control' in pg


def tst_evo_setup_evo_init_func(config=None):
    '''Tests that param_grid is parsed and the deap toolbox
    is created ok so that toolbox.population_guess can
    be called without error.'''
    (config,
     step_name_to_param_grid_name,
     param_grid_name_to_deap) = _setup_pg(config=config)
    eps = ea_setup(config)
    assert isinstance(eps, dict) and len(eps) == 1
    evo_params = eps[(0, 0)]
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
        9,
        [],
        ('not_a_key', 'kmeans', 'model_init_kwargs', 'n_clusters'),
        ('not_a_key', 'kmeans', 'model_init_kwargs', 'n_clusters'),
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
@pytest.mark.parametrize('key, value', tst_params)
def test_bad_param_grid_config(key, value):
    set_key_tst_bad_config_once(key, value)


