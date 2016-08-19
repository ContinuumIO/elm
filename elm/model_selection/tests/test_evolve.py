import copy

import pytest
import yaml

from elm.config import ConfigParser
from elm.model_selection.evolve import (get_param_grid,
                                        individual_to_new_config,
                                        _get_evolve_meta,
                                        EvoParams,
                                        evolve_setup,
                                        evo_init_func,
                                        evo_general)
from elm.model_selection.tests.evolve_example_config import CONFIG_STR

def _setup():
    '''Return the config above and the param_grid'''
    config = ConfigParser(config=yaml.load(CONFIG_STR))
    param_grid = get_param_grid(config, config.pipeline[0])
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
    assert k == 'pca_kmeans'
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
    param_grid_item = param_grid['pca_kmeans']
    ind = [0,] * len(param_grid_item['choices'])
    new_config = individual_to_new_config(config, param_grid_item, ind)
    assert new_config.train['kmeans']['model_init_kwargs']['n_clusters'] == 3
    assert new_config.transform['pca']['model_init_kwargs']['n_components'] == 2
    assert new_config.pipeline[0]['sample_pipeline'] == minimal
    assert new_config.feature_selection['top_n']['kwargs']['percentile'] == 30
    ind = [1,] * len(param_grid_item['choices'])
    new_config = individual_to_new_config(config, param_grid_item, ind)
    print(new_config.pipeline)
    assert new_config.train['kmeans']['model_init_kwargs']['n_clusters'] == 4
    assert new_config.transform['pca']['model_init_kwargs']['n_components'] == 3
    assert new_config.pipeline[0]['sample_pipeline'] == top_n
    assert new_config.feature_selection['top_n']['kwargs']['percentile'] == 40


def test__get_evolve_meta():
    '''Tests param_grid metadata'''
    (config,
     step_name_to_param_grid_name,
     param_grid_name_to_deap) = _setup_pg()
    expected_key = (0, 'kmeans')
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
    eps = evolve_setup(config)
    assert isinstance(eps, dict) and len(eps) == 1
    evo_params = eps[0]
    assert isinstance(evo_params, EvoParams)
    pop = evo_init_func(evo_params)
    assert len(pop) == 24
    for ind in pop:
        for item, choices in zip(ind, evo_params.deap_params['choices']):
            assert item < len(choices) and item >= 0
    assert len(set(map(tuple, pop))) > 1
    return config, evo_params, pop


test_evo_setup_evo_init_func = tst_evo_setup_evo_init_func

# The following are test data to test_evo_general
min_fitnesses = [(50,), (0,)] + [(100,)] * 22
max_fitnesses = [(50,), (1000,)] + [(100,)] * 22
min_max_fitnesses = [(50, 100), (0, 1000)] + [(100, 50)] * 22
score_weights = [[-1,], [1,], [-1, 1]]
zipped_tst_args = zip((min_fitnesses, max_fitnesses, min_max_fitnesses),
                      score_weights)

@pytest.mark.parametrize('fitnesses, score_weights', zipped_tst_args)
def test_evo_general(fitnesses, score_weights):
    '''This test ensures that evo_general, a general
    evolutionary algorithm can minimize, maximize or
    handle multiple objectives.  In each of the fitnesses
    sequences passed in, the 2nd individual is known
    to be the most fit (given corresponding score_weights
    which determine whether min/maximizing the objective(s))
    '''
    config = yaml.load(CONFIG_STR)
    config['model_scoring']['testing_model_scoring']['score_weights'] = score_weights
    config, evo_params, pop = tst_evo_setup_evo_init_func()
    control = evo_params.deap_params['control']
    ea_gen = evo_general(evo_params.toolbox,
                         pop,
                         control['cxpb'],
                         control['mutpb'],
                         control['ngen'])
    assert next(ea_gen) is None # dummy call to next
    invalid_ind = pop
    assert len(pop) == 24
    original_pop = copy.deepcopy(pop)
    best = original_pop[1]  # in this synthetic data,
                            # the 2nd param set is always best
    while invalid_ind:
        (pop, invalid_ind, record, logbook) = ea_gen.send(fitnesses)
    matches_best = tuple(ind for ind in pop if ind == best)
    assert matches_best
    assert original_pop != pop

