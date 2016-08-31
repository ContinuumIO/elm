from argparse import Namespace, ArgumentParser
import contextlib
import glob
import os
import subprocess as sp
import yaml
from elm.pipeline.tests.util import tmp_dirs_context, test_one_config
from elm.scripts.main import main, cli
from elm.config.env import parse_env_vars


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
    if not s:
        print('TEST_OK', '', context, sep='\t')
    else:
        print('TEST_FAIL', s, context, sep='\t')


def run_all_example_configs(env, path):
    assert os.path.exists(path), 'Define the environment variable ELM_CONFIGS_PATH (directory of yaml configs)'
    test_configs = glob.glob(os.path.join(path, '*.yaml'))
    for fname in test_configs:
        with env_patch(**env) as new_env:
            args = Namespace(config=os.path.abspath(fname),
                             config_dir=None,
                             echo_config=False)
            ret_val = main(args=args, sys_argv=None, return_0_if_ok=True)
            print_status(ret_val, fname)


def run_all_unit_tests(repo_dir, env, pytest_mark=None):
    with env_patch(**env) as new_env:
        proc_args = ['py.test']
        if pytest_mark:
            proc_args += ['-m', pytest_mark]
        proc = sp.Popen(proc_args, cwd=repo_dir, env=new_env,
                        stdout=sp.PIPE, stderr=sp.STDOUT)
        def write():
            line = proc.stdout.readline()
            if line:
                print(line, end='')
            return line
        while proc.poll() is None:
            if not write():
                break
        while write():
            pass
        print_status(proc.wait())


def run_all_tests():
    choices = ['ALL', 'SERIAL', 'DISTRIBUTED', 'THREAD_POOL', 'PROCESS_POOL']
    env = parse_env_vars()
    parser = ArgumentParser(description='Run longer-running tests of elm')
    parser.add_argument('repo_dir', help='Directory that is the top dir of cloned elm repo')
    parser.add_argument('elm_configs_path', help='Path')
    parser.add_argument('--pytest-mark', help='Mark to pass to py.test -m (marker of unit tests)')
    parser.add_argument('--dask-executors', choices=choices, nargs='+',
                        help='Dask executor(s) to test: %(choices)s')
    parser.add_argument('--dask-scheduler', help='Dask scheduler URL')
    parser.add_argument('--skip-pytest', action='store_true', help='Do not run py.test (default is run py.test as well as configs)')

    args = parser.parse_args()
    args.config_dir = None
    if not args.dask_scheduler:
        args.dask_scheduler = env.get('DASK_SCHEDULER', '10.0.0.10:8786')
    if not args.dask_executors or 'ALL' in args.dask_executors:
        args.dask_executors = [c for c in choices if c != 'ALL']
    print('Running run_all_tests with args: {}'.format(args))
    assert os.path.exists(args.repo_dir)
    for executor in args.dask_executors:
        new_env = {'DASK_SCHEDULER': args.dask_scheduler or '',
                   'DASK_EXECUTOR': executor}
        if not args.skip_pytest:
            run_all_unit_tests(args.repo_dir, new_env,
                               pytest_mark=args.pytest_mark)
        run_all_example_configs(new_env, path=args.elm_configs_path)
