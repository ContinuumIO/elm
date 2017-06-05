from __future__ import absolute_import, division, print_function, unicode_literals

'''Module of helpers for building command line interfaces'''
from argparse import ArgumentParser

from elm.config.env import ENVIRONMENT_VARS_SPEC

def add_env_vars_override_options(parser):
    '''Add to parser overrides for env vars, e.g. DASK_SCHEDULER to --dask-scheduler'''
    group = parser.add_argument_group('Environment','Compute settings (see also help on environment variables)')
    lower_name = lambda n: '--' + n.lower().replace('_', '-')
    for v in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
        group.add_argument(lower_name(v['name']), help='See also env var {}'.format(v['name']))
    for v in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
        name = lower_name(v['name'])
        hlp = 'See also {}'.format(v['name'])
        if 'choices' in v:
            group.add_argument(name, help=hlp, choices=v['choices'])
        else:
            group.add_argument(name, help=hlp)


def add_config_file_argument(parser):
    '''Add parser arguments related to taking config file or directory'''
    group = parser.add_argument_group('Inputs', 'Input config file or directory')
    group = group.add_mutually_exclusive_group()
    group.add_argument('--config', type=str, help="Path to yaml config")
    group.add_argument('--config-dir', type=str, help='Path to a directory of yaml configs')


def add_ensemble_kwargs(parser):
    '''Add arguments to parser that override ensemble settings'''
    group = parser.add_argument_group('Control', 'Keyword arguments to elm.pipeline.ensemble')
    group.add_argument('--partial-fit-batches', type=int,
                        help='Partial fit batches (for estimator specified in config\'s "train"')
    group.add_argument('--init-ensemble-size', type=int,
                        help='Initial ensemble size (ignored if using "ensemble_init_func"')
    group.add_argument('--saved-ensemble-size', type=int,
                        help='How many of the "best" models to serialize')
    group.add_argument('--ngen', type=int,
                        help='Number of ensemble generations, defaulting to ngen from ensemble_kwargs in config')

def add_run_options(parser):
    '''Add the --train-only and --predict-only arguments to parser'''
    parser.add_argument_group('Run', 'Run options')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train-only', action='store_true',
                      help='Run only the training, not prediction, actions specified by config')

    group.add_argument('--predict-only', action='store_true',
                       help='Run only the prediction, not training, actions specified by config')

