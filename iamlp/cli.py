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
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--n-clusters', type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument('--n-ensembles', type=int, default=DEFAULT_N_ENSEMBLES)
    parser.add_argument('--n-samples-each-fit', type=int,default=DEFAULT_N_SAMPLES_EACH_FIT)
    parser.add_argument('--files-per-sample', type=int, default=DEFAULT_FILES_PER_SAMPLE)
    parser.add_argument('--band-specs-json', type=str,help="Band specs from json file")
    return parser
