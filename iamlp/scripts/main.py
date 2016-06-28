
from argparse import ArgumentParser
import datetime
import logging
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
    dask_executor = config.DASK_EXECUTOR
    dask_scheduler = config.DASK_SCHEDULER
    with executor_context(dask_executor, dask_scheduler) as executor:
        pipeline(config, executor)
        ended = datetime.datetime.now()
        logger.info('Ran from {} to {} ({} '
                    'seconds)'.format(started, ended,
                                      (ended - started).total_seconds()))
    return models


if __name__ == "__main__":
    models = main()
