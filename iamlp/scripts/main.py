
from argparse import ArgumentParser
import datetime
from functools import partial
import os

from sklearn.cluster import MiniBatchKMeans
from dask.diagnostics import ProgressBar

from iamlp.config.cli import (add_local_dataset_options,
                              add_ensemble_partial_fit_args)
from iamlp.config import DEFAULTS, ConfigParser
from iamlp.writers.serialize import serialize
from iamlp.settings import executor_context


def cli(args=None, parse_this_str=None):
    if args:
        return args
    parser = ArgumentParser(description="Pipeline classifier / predictor using ensemble and partial_fit methods")
    parser = add_ensemble_partial_fit_args(parser)
    if parse_this_str:
        return parser.parse_args(parse_this_str)
    return parser.parse_args()


def main(args=None, parse_this_str=None):
    started = datetime.datetime.now()
    args = cli(args=args, parse_this_str=parse_this_str)
    config = ConfigParser(args.config)
    with executor_context() as (executor, get_func):
        with ProgressBar() as progress:
            if DASK_EXECUTOR == 'DISTRIBUTED':
                get = get_func
            else:
                get = None
                #print('keys', [tuple(m.dask.keys()) for m in models])
            ended = datetime.datetime.now()
            print('Ran from', started, 'to', ended, '({} seconds)'.format((ended - started).total_seconds()))
        return models


if __name__ == "__main__":
    models = main()
