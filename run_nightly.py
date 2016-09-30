#!/usr/bin/env python3

import os
import datetime
import argparse
import subprocess as sp
import tempfile
import sys
import shutil
import atexit
from elm.scripts import run_all_tests



def reconstruct_cmdline(test_cmd, parser, args):
    # Reconstruct command-line from parser and args
    actions = {action.dest: action for action in parser._actions}
    for arg, val in args._get_kwargs():
        if actions[arg].const:
            if val != actions[arg].default:
                test_cmd += ' {}'.format(actions[arg].option_strings[0])
        elif val is not None:
            if isinstance(val, (list, tuple)):
                val = ' '.join(val)
            elif not isinstance(val, (str,)):
                raise ValueError('Not sure how to handle argument value of type {}'.format(type(val)))
            test_cmd += ' {} {}'.format(actions[arg].option_strings[0], val)

    # Add positional arguments
    positional_args = (os.getcwd(), os.path.join(os.getcwd(), 'example_configs'))
    test_cmd += ' '+' '.join(positional_args)

    return test_cmd


def main():
    # Reuse the CLI parser from elm.scripts.run_all_tests
    parser = run_all_tests.build_cli_parser(include_positional=False)
    args = parser.parse_args()

    # Provide the arguments which the user omitted
    if args.dask_clients is None:
        args.dask_clients = ['SERIAL', 'DISTRIBUTED']
    if args.dask_scheduler is None:
        args.dask_scheduler = 'localhost:8786'
    if args.remote_git_branch is None:
        args.remote_git_branch = 'master'
    #args.repo_dir = os.getcwd()
    #args.elm_configs_path = os.path.join(args.repo_dir, 'example_configs')

    test_cmd = reconstruct_cmdline('elm-run-all-tests', parser, args)


    # Create a temporary directory for cloning/testing;
    # Remove this directory automatically if the script exits.
    tmp_dpath = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(tmp_dpath))

    test_env_name = os.path.basename(tmp_dpath)
    # The URL here needs to be git-based so that the SSH Deploy Key works
    sp.check_call('git clone git@github.com:ContinuumIO/elm {}'.format(tmp_dpath), shell=True, executable='/bin/bash')
    sp.check_call('git pull --all', shell=True, cwd=tmp_dpath, executable='/bin/bash')
    sp.check_call('git checkout {}'.format(args.remote_git_branch), shell=True, cwd=tmp_dpath, executable='/bin/bash')
    sp.check_call('conda env create -n {}'.format(test_env_name), shell=True, cwd=tmp_dpath, executable='/bin/bash')
    sp.check_call('source activate {}'.format(test_env_name), shell=True, cwd=tmp_dpath, executable='/bin/bash')
    sp.check_call('python setup.py develop', shell=True, cwd=tmp_dpath, executable='/bin/bash')

    print('\n####\nTEST COMMAND:\n\t{}\n####\n'.format(test_cmd))

    # Run the nightly tests
    start_dt = datetime.datetime.now()
    sp.check_call(test_cmd, shell=True, cwd=tmp_dpath, executable='/bin/bash')
    end_dt = datetime.datetime.now()
    print('\nSTART TIME: {}'.format(start_dt.isoformat()))
    print('\nEND TIME: {}'.format(end_dt.isoformat()))


if __name__ == '__main__':
    main()
