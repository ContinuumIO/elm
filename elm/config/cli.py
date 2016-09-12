from argparse import ArgumentParser

from elm.config.defaults import DEFAULTS
from elm.config.env import ENVIRONMENT_VARS_SPEC

def add_cmd_line_options(parser):
    lower_name = lambda n: '--' + n.lower().replace('_', '-')
    for v in ENVIRONMENT_VARS_SPEC['int_fields_specs']:
        parser.add_argument(lower_name(v['name']), help='See also env var {}'.format(v['name']))
    for v in ENVIRONMENT_VARS_SPEC['str_fields_specs']:
        name = lower_name(v['name'])
        hlp = 'See also {}'.format(v['name'])
        if 'choices' in v:
            parser.add_argument(name, help=hlp, choices=v['choices'])
        else:
            parser.add_argument(name, help=hlp)


def add_years_data_days(parser):
    parser.add_argument('--years',
                        type=int,
                        nargs='+',
                        help='Integer years to include')
    parser.add_argument('--data_days',
                        type=int,
                        nargs='+',
                        help='Integer data day(s) to include')


def add_local_dataset_options(parser):
    parser.add_argument('--product_number',
                        type=int,
                        default=3001,
                        help="ladsweb integer directory number in allData/ (default: %(default)s)")
    parser.add_argument('--product_name',
                        type=str,
                        default='NPP_DSRF1KD_L2GD',
                        help='ladsweb dataset name within allData/--product_number (default: %(default)s)')
    add_years_data_days(parser)


def add_config_file_argument(parser=None):
    parser.add_argument('--config', type=str, help="Path to yaml config")
    parser.add_argument('--config-dir', type=str, help='Path to a directory of yaml configs')


def add_sample_ladsweb_options(parser):
    parser.add_argument('--product_numbers', type=str, nargs='+', help='Limit to these product_numbers or None for all product numbers')
    parser.add_argument('--product_names', type=str, nargs='+', help='Limit to these product names or None for all product names for each product number')
    parser.add_argument('-n', '--n-file-samples', default=1, type=int,help="How many files of each product")
