from __future__ import absolute_import, division, print_function, unicode_literals


from argparse import ArgumentParser, Namespace
import contextlib
import copy
import datetime
from functools import partial
import glob
import logging
import os
import warnings

from dask.diagnostics import ProgressBar

from elm.config.cli import (add_config_file_argument,
                            add_run_options,
                            add_ensemble_kwargs,
                            add_env_vars_override_options,
                            )
from elm.config import (DEFAULTS, ConfigParser,
                        client_context, ElmConfigError,
                        parse_env_vars)
from elm.pipeline import parse_run_config

logger = logging.getLogger(__name__)


def cli(args=None, sys_argv=None):
    if args:
        return args
    parser = ArgumentParser(description="Pipeline classifier / predictor using ensemble and partial_fit methods")
    add_config_file_argument(parser)
    add_run_options(parser)
    add_ensemble_kwargs(parser)
    add_env_vars_override_options(parser)
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
        if err:
            logger.info('There were errors {}'.format(err))


def run_one_config(args=None, sys_argv=None,
                   return_0_if_ok=True, config_dict=None,
                   client=None, started=None):

    started = started or datetime.datetime.now()
    args = args or cli(args=None, sys_argv=sys_argv)
    config_dict = config_dict or args.config
    with try_finally_log_etime(started) as _:
        config = ConfigParser(config_dict, cmd_args=args)
        if client is None:
            if args.echo_config:
                logger.info(str(config))
            dask_client = getattr(config, 'DASK_CLIENT', 'SERIAL')
            dask_scheduler = getattr(config, 'DASK_SCHEDULER', None)
            with warnings.catch_warnings():
                # scikit-learn has a number
                # of deprecation warnings for kmeans
                warnings.simplefilter("ignore")
                with client_context(dask_client, dask_scheduler) as client:
                    return_values = parse_run_config(config, client)
        else:
            return_values = parse_run_config(config, client)

    if return_0_if_ok:
        return 0
    return return_values


def _run_one_config_of_many(fname, **kwargs):
    logger.info('Run config {}'.format(fname))
    args = copy.deepcopy(kwargs['args'])
    args.config = fname
    args.config_dir = None
    kw = copy.deepcopy(kwargs)
    kw['args'] = args
    return run_one_config(**kw)


def run_many_configs(args=None, sys_argv=None, return_0_if_ok=True,
                     started=None):
    started = started or datetime.datetime.now()
    env_cmd_line = Namespace(**{k: v  for d in (vars(args), parse_env_vars())
                                for k, v in d.items()
                                })
    logger.info('With --config-dir, DASK_CLIENT and DASK_SCHEDULER in config files are ignored')
    dask_client = getattr(env_cmd_line, 'DASK_CLIENT', 'SERIAL')
    dask_scheduler = getattr(env_cmd_line, 'DASK_SCHEDULER', None)
    ret_val = 1
    with try_finally_log_etime(started) as _:
        with warnings.catch_warnings():
            # scikit-learn has a number
            # of deprecation warnings for kmeans
            warnings.simplefilter("ignore")
            results = [1]
            with client_context(dask_client, dask_scheduler) as client:
                kw = {'args': args, 'sys_argv': sys_argv,
                      'return_0_if_ok': True,
                      'client': client,}
                pipe = partial(_run_one_config_of_many, **kw)
                fnames = glob.glob(os.path.join(args.config_dir, '*.yaml'))
                ret_val = max(map(pipe, fnames))
    return ret_val

class ElmMainDeprecation(ValueError):
    pass

def main(args=None, sys_argv=None, return_0_if_ok=True):
    raise ElmMainDeprecation('The console entry point elm-main for running '
                             'yaml configs is temporarily deprecated during '
                             'refactoring of elm')
    started = datetime.datetime.now()
    args = cli(args=args, sys_argv=sys_argv)
    if args.config_dir is not None and args.config is not None:
        raise ElmConfigError('Expected --config-dir or --config, not both args')
    elif args.config_dir:
        ret = run_many_configs(args, started=started, return_0_if_ok=return_0_if_ok)
        logger.info('Elapsed time {}'.format((datetime.datetime.now() - started).total_seconds()))
    ret = run_one_config(args=args, sys_argv=sys_argv,
                          return_0_if_ok=return_0_if_ok,
                          started=started)
    return ret

if __name__ == "__main__":
    models = main()
