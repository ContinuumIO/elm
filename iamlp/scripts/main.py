
from argparse import ArgumentParser
import datetime
import os

from dask.diagnostics import ProgressBar

from iamlp.config.cli import add_config_file_argument
from iamlp.config import DEFAULTS, ConfigParser, executor_context, delayed
from iamlp.writers.serialize import serialize
from iamlp.pipeline import pipeline


def cli(args=None, parse_this_str=None):
    if args:
        return args
    parser = ArgumentParser(description="Pipeline classifier / predictor using ensemble and partial_fit methods")
    parser = add_config_file_argument(parser)
    if parse_this_str:
        return parser.parse_args(parse_this_str)
    return parser.parse_args()


def main(args=None, parse_this_str=None):
    started = datetime.datetime.now()
    args = cli(args=args, parse_this_str=parse_this_str)
    config = ConfigParser(args.config)
    with executor_context() as (executor, get_func):
        with ProgressBar() as progress:
            pipeline(config, executor)
            ended = datetime.datetime.now()
            print('Ran from', started, 'to', ended, '({} seconds)'.format((ended - started).total_seconds()))
        return models


if __name__ == "__main__":
    models = main()
