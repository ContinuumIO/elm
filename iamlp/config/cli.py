from argparse import ArgumentParser


def add_local_dataset_options(parser):
    parser.add_argument('--product_number',
                        type=int,
                        default=3001,
                        help="ladsweb integer directory number in allData/ (default: %(default)s)")
    parser.add_argument('--product_name',
                        type=str,
                        default=DEFAULT_DATASET,
                        help='ladsweb dataset name within allData/--product_number (default: %(default)s)')
    parser.add_argument('--years',
                        type=int,
                        nargs='+',
                        help='Integer years to include')
    parser.add_argument('--data_days',
                        type=int,
                        nargs='+',
                        help='Integer data day(s) to include')
    return parser

def add_ensemble_partial_fit_args(parser=None):
    parser.add_argument('--config', type=str, help="Path to yaml config")
    return parser
