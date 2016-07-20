from argparse import ArgumentParser

from elm.config.defaults import DEFAULTS

d = [v for v in DEFAULTS['data_sources'].values() if 'product_name' in v][0]
DEFAULT_PRODUCT_NAME = d['product_name']
DEFAULT_PRODUCT_NUMBER = d['product_number']

def add_years_data_days(parser):
    parser.add_argument('--years',
                        type=int,
                        nargs='+',
                        help='Integer years to include')
    parser.add_argument('--data_days',
                        type=int,
                        nargs='+',
                        help='Integer data day(s) to include')
    return parser

def add_local_dataset_options(parser):
    parser.add_argument('--product_number',
                        type=int,
                        default=DEFAULT_PRODUCT_NUMBER,
                        help="ladsweb integer directory number in allData/ (default: %(default)s)")
    parser.add_argument('--product_name',
                        type=str,
                        default=DEFAULT_PRODUCT_NAME,
                        help='ladsweb dataset name within allData/--product_number (default: %(default)s)')
    return add_years_data_days(parser)

def add_config_file_argument(parser=None):
    parser.add_argument('--config', type=str, help="Path to yaml config")
    return parser

def add_sample_ladsweb_options(parser):
    parser.add_argument('--product_numbers', type=str, nargs='+', help='Limit to these product_numbers or None for all product numbers')
    parser.add_argument('--product_names', type=str, nargs='+', help='Limit to these product names or None for all product names for each product number')
    parser.add_argument('-n', '--n-file-samples', default=1, type=int,help="How many files of each product")
    return parser