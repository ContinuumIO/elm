
from argparse import ArgumentParser, Namespace
import contextlib
import copy
import datetime
from functools import partial
import logging
import os
import warnings

from dask.diagnostics import ProgressBar

from elm.config.cli import add_config_file_argument, add_cmd_line_options
from elm.config import (DEFAULTS, ConfigParser,
                        executor_context, ElmConfigError,
                        parse_env_vars)
from elm.pipeline import pipeline
from elm.pipeline.executor_util import wait_for_futures

logger = logging.getLogger(__name__)


def cli(args=None, sys_argv=None):
    if args:
        return args
    parser = ArgumentParser(description="Pipeline classifier / predictor using ensemble and partial_fit methods")
    add_config_file_argument(parser)
    add_cmd_line_options(parser)
    parser.add_argument('--echo-config', action='store_true',
                        help='Output running config as it is parsed')
    if sys_argv:
        return parser.parse_args(sys_argv)
    return parser.parse_args()


@contextlib.contextmanager
def try_finally_log_etime(started):
    err = None
    try:
        yield True
    except Exception as e:
        err = e
        raise
    finally:
        ended = datetime.datetime.now()
        logger.info('Ran from {} to {} ({} '
                    'seconds)'.format(started, ended,
                                      (ended - started).total_seconds()))
        logger.info('There were errors {}'.format(repr(err)) if err else 'ok')



def run_one_config(args=None, sys_argv=None,
                   return_0_if_ok=True, config_dict=None,
                   executor=None, started=None):

    started = started or datetime.datetime.now()
    args = args or cli(args=None, sys_argv=sys_argv)
    config_dict = config_dict or args.config
    with try_finally_log_etime(started) as _:
        config = ConfigParser(config_dict, cmd_args=args)
        if executor is None:
            if args.echo_config:
                logger.info(str(config))
            dask_executor = getattr(config, 'DASK_EXECUTOR', 'SERIAL')
            dask_scheduler = getattr(config, 'DASK_SCHEDULER', None)
            with warnings.catch_warnings():
                # scikit-learn has a number
                # of deprecation warnings for kmeans
                warnings.simplefilter("ignore")
                with executor_context(dask_executor, dask_scheduler) as executor:
                    return_values = pipeline(config, executor)
        else:
            return_values = pipeline(config, executor)

    if return_0_if_ok:
        return 0
    return return_values

def _run_one_config_of_many(fname, **kwargs):
    args = copy.deepcopy(kwargs['args'])
    args.config = fname
    args.config_dir = None
    kw = copy.deepcopy(kwargs)
    kw['args'] = args
    return run_one_config(**kw)


def run_many_configs(args=None, sys_argv=None, return_0_if_ok=None,
                     started=None):
    started = started or datetime.datetime.now()
    env_cmd_line = Namespace(**{k: v for k, v in d.items()
                                for d in (vars(args), parse_env_vars())})
    logger.info('With --config-dir, DASK_EXECUTOR and DASK_SCHEDULER in config files are ignored')
    dask_executor = getattr(env_cmd_line, 'DASK_EXECUTOR', 'SERIAL')
    dask_scheduler = getattr(env_cmd_line, 'DASK_SCHEDULER', None)
    ret_val = 1
    with try_finally_log_etime(started) as _:
        with warnings.catch_warnings():
            # scikit-learn has a number
            # of deprecation warnings for kmeans
            warnings.simplefilter("ignore")
            results = [1]
            with executor_context(dask_executor, dask_scheduler) as executor:
                kw = {'args': args, 'sys_argv': sys_argv,
                      'return_0_if_ok': True,
                      'executor': executor,}
                pipe = partial(_run_one_config_of_many, **kw)
                fnames = glob.glob(os.path.join(args.config_dir, '*.yaml'))
                futures = executor.map(pipe, fnames)
                results = wait_for_futures(futures, executor=None)
            ret_val = max(results)
    return ret_val


def main(args=None, sys_argv=None, return_0_if_ok=True):
    started = datetime.datetime.now()
    args = cli(args=args, sys_argv=sys_argv)
    if args.config_dir is not None and args.config is not None:
        raise ElmConfigError('Expected --config-dir or --config, not both args')
    elif args.config_dir:
        return run_many_configs(args, started=started)
    return run_one_config(args=args, sys_argv=sys_argv,
                          return_0_if_ok=return_0_if_ok,
                          started=started)

if __name__ == "__main__":
    models = main()
