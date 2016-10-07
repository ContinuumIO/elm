from argparse import Namespace, ArgumentParser
import contextlib
import glob
import logging
import os
import shutil
import subprocess as sp
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.stats import describe
import yaml

from elm.config import import_callable
from elm.pipeline.tests.util import tmp_dirs_context, test_one_config
from elm.scripts.main import main, cli
from elm.config.env import parse_env_vars



logger = logging.getLogger(__name__)

SKIP_TOKENS = ('ProjectedGradientNMF','BayesianRidge', 'BernoulliRBM',
               'Lasso','ElasticNetCV','FactorAnalysis',
               'Lars', 'LarsCV','LassoCV','LassoLarsIC')

LARGE_TEST_SETTINGS = {'ensembles': {'saved_ensemble_size': 2,
                                     'init_ensemble_size': 12,
                                     'ngen': 4,
                                     },
                       'param_grids': {'control': {'ngen': 4,
                                       'mu': 12,
                                       'k': 4}}}

DASK_CLIENTS = ['ALL', 'SERIAL', 'DISTRIBUTED', 'THREAD_POOL', ]

STATUS_COUNTER = {'ok': 0, 'fail': 0, 'xfail': 0}
ETIMES = {}

def add_large_test_settings(config):
    for k, v in LARGE_TEST_SETTINGS.items():
        if k in config:
            for k3, v3 in v.items():
                for k2 in config[k]:
                    if k == 'param_grids':
                        config[k][k2][k3].update(v3)
                    else:
                        config[k][k2][k3] = v3

    return config

@contextlib.contextmanager
def env_patch(**new_env):
    old_env = {k: v for k, v in os.environ.copy().items()
               if k in new_env}
    try:
        os.environ.update(new_env)
        yield os.environ
    finally:
        os.environ.update(old_env)


def print_status(s, context):
    global STATUS_COUNTER
    if isinstance(s, str) and 'XFAIL' in s:
        logger.info('TEST_XFAIL\t{}\t{}'.format(s, context))
        STATUS_COUNTER['xfail'] += 1
    if not s:
        STATUS_COUNTER['ok'] += 1
        logger.info('TEST_OK\t\t' + context)
    else:
        STATUS_COUNTER['fail'] += 1
        logger.info('TEST_FAIL\t{}\t{}'.format(s, context))
    if 'unit_tests' == context:
        STATUS_COUNTER[context] = 'ok' if not s else 'fail'

@contextlib.contextmanager
def modify_config_file(fname, env, large_test_mode):
    fname = os.path.abspath(fname)
    fname2 = 'no-file'
    with open(fname) as f:
        config = yaml.load(f.read())
    try:
        for k, v in env.items():
            config[k] = v
        if large_test_mode:
            config = add_large_test_settings(config)
        fname2 = fname + '_2.yaml'
        with open(fname2, 'w') as f:
            f.write(yaml.dump(config))
        yield fname2
    finally:
        if os.path.exists(fname2):
            os.remove(fname2)


def run_all_example_configs(env, path, large_test_mode, glob_pattern):
    global ETIMES
    test_configs = glob.glob(os.path.join(path, glob_pattern or '*.yaml'))
    logger.info('Found test configs: {}'.format(test_configs))
    for fname in test_configs:
        if any(s in fname for s in SKIP_TOKENS):
            print_status('XFAIL', fname)
            logger.info('Skip config {}'.format(fname))
            continue
        logger.info('Run config {}'.format(fname))
        with env_patch(**env) as new_env:
            with modify_config_file(fname, new_env, large_test_mode) as fname2:
                args = Namespace(config=fname2,
                                 config_dir=None,
                                 echo_config=False)
                started = time.time()
                try:
                    ret_val = main(args=args, sys_argv=None, return_0_if_ok=True)
                except Exception as e:
                    ret_val = repr(e)
                    print(e, traceback.format_exc())
                exe = new_env.get("DASK_EXECUTOR")
                if not fname in ETIMES:
                    ETIMES[fname] = {}
                if not exe in ETIMES[fname]:
                    ETIMES[fname][exe] = {}
                ETIMES[fname][exe] = time.time() - started if not ret_val else None
                print_status(ret_val, fname2)


def run_all_example_scripts(env, paths, glob_pattern):
    global ETIMES
    for path in paths:
        test_scripts = glob.glob(os.path.join(path, glob_pattern or '*.py'))
        logger.info('Found test scripts: {}'.format(test_scripts))
        for fname in test_scripts:
            logger.info('Run script {}'.format(fname))
            with env_patch(**env) as new_env:
                started = time.time()
                try:
                    ret_val = sp.check_output('python '+fname, shell=True, executable='/bin/bash')
                except Exception as e:
                    ret_val = repr(e)
                    print(e, traceback.format_exc())
                exe = new_env.get("DASK_EXECUTOR")
                if not fname in ETIMES:
                    ETIMES[fname] = {}
                if not exe in ETIMES[fname]:
                    ETIMES[fname][exe] = {}
                ETIMES[fname][exe] = time.time() - started if not ret_val else None
                print_status(ret_val, fname)


def proc_wrapper(proc):
    def write(proc):
        line = proc.stdout.read(1).decode()
        print(line, end='')
        return line
    while proc.poll() is None:
        if not write(proc):
            break
        time.sleep(0.02)
    while write(proc):
        pass
    return proc.wait()


def run_all_unit_tests(repo_dir, env, pytest_mark=None):
    with env_patch(**env) as new_env:
        proc_args = ['py.test']
        if pytest_mark:
            proc_args += ['-m', pytest_mark]
        proc_kwargs = dict(cwd=repo_dir, env=new_env,
                      stdout=sp.PIPE, stderr=sp.STDOUT)
        proc = sp.Popen(proc_args, **proc_kwargs)
        logger.info('{} with DASK_EXECUTOR={}'.format(proc_args, new_env['DASK_EXECUTOR']))
        ret_val = proc_wrapper(proc)
        print_status(ret_val, 'unit_tests')


def run_all_tests_remote_git_branch(args):
    branch = args.remote_git_branch
    tmp = os.path.join(args.repo_dir, 'tmp')
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    sp.Popen(['git', 'clone','http://github.com/ContinuumIO/elm'], cwd=tmp).wait()
    cwd = os.path.join(tmp, 'elm')
    sp.Popen(['git', 'fetch', '--all'], cwd=cwd).wait()
    sp.Popen(['git', 'checkout', branch], cwd=cwd).wait()
    sp.Popen(['source', 'activate', 'elm-env', '&&', 'python', 'setup.py', 'develop'],cwd=cwd).wait()
    args_lst = []
    for arg in sys.argv:
        if arg == args.repo_dir:
            arg = tmp
        args_lst.append(arg)
    proc = sp.Popen(args_lst, cwd=os.curdir,stdout=sp.PIPE, stderr=sp.STDOUT)
    return proc_wrapper(proc)


def build_cli_parser(include_positional=True):
    '''Return an argparse.ArgumentParser object which includes the arguments needed by this script.

    This function is also used by run_nightly.py.
    '''
    parser = ArgumentParser(description='Run longer-running tests of elm')
    if include_positional:
        parser.add_argument('repo_dir', help='Directory that is the top dir of cloned elm repo')
        parser.add_argument('elm_examples_path', help='Path to a directory which contains subdirectories "example_configs", "example_scripts", and "example_data" with yaml-configs, python-scripts, and example data, respectively')
    parser.add_argument('--pytest-mark', help='Mark to pass to py.test -m (marker of unit tests)')
    parser.add_argument('--dask-clients', choices=DASK_CLIENTS, nargs='+',
                        help='Dask client(s) to test: {}'.format(DASK_CLIENTS))
    parser.add_argument('--dask-scheduler', help='Dask scheduler URL')
    parser.add_argument('--skip-pytest', action='store_true', help='Do not run py.test (default is run py.test as well as configs)')
    parser.add_argument('--add-large-test-settings', action='store_true', help='Adjust configs for larger ensembles / param_grids')
    parser.add_argument('--glob-pattern', help='Glob within repo_dir')
    parser.add_argument('--remote-git-branch', help='Run on a remote git branch')
    return parser


def run_all_tests(args=None):
    global STATUS_COUNTER
    env = parse_env_vars()
    if args is None:
        parser = build_cli_parser()
        args = parser.parse_args()
    args.config_dir = None
    if not args.dask_scheduler:
        args.dask_scheduler = env.get('DASK_SCHEDULER', '10.0.0.10:8786')
    if not args.dask_clients or 'ALL' in args.dask_clients:
        args.dask_clients = [c for c in DASK_CLIENTS if c != 'ALL']
    logger.info('Running run_all_tests with args: {}'.format(args))
    assert os.path.exists(args.repo_dir)
    for client in args.dask_clients:
        new_env = {'DASK_SCHEDULER': args.dask_scheduler or '',
                   'DASK_EXECUTOR': client,
                   'ELM_EXAMPLE_DATA_PATH': os.path.join(args.elm_examples_path, 'example_data')}
        if not args.skip_pytest:
            run_all_unit_tests(args.repo_dir, new_env,
                               pytest_mark=args.pytest_mark)
        run_all_example_scripts(new_env, paths=[os.path.join(args.elm_examples_path, 'example_scripts'),
                                                os.path.join(args.elm_examples_path, 'example_configs')],
                                glob_pattern=args.glob_pattern)
        run_all_example_configs(new_env, path=os.path.join(args.elm_examples_path, 'example_configs'),
                                large_test_mode=args.add_large_test_settings,
                                glob_pattern=args.glob_pattern)
    failed_unit_tests = STATUS_COUNTER.get('unit_tests') != 'ok' and not args.skip_pytest
    if STATUS_COUNTER.get('fail') or failed_unit_tests:
        raise ValueError('Tests failed {}'.format(STATUS_COUNTER))
    print('ETIMES', ETIMES)
    speed_up_fracs = {k: [] for k in args.dask_clients if k != 'SERIAL'}
    for fname in ETIMES:
        if fname == 'unit_tests':
            continue
        if ETIMES[fname].get("SERIAL"):
            base = ETIMES[fname]['SERIAL']
            for k, v in ETIMES[fname].items():
                if k == 'SERIAL':
                    continue
                speed_up_fracs[k].append( (base - v) / base)
    speed_up_fracs_summary = {k: describe(np.array(v))
                              for k, v in speed_up_fracs.items()}
    print('speed_up_fracs {}'.format(speed_up_fracs))
    print('Speed up summary {}'.format(speed_up_fracs_summary))
    print('STATUS', STATUS_COUNTER)


