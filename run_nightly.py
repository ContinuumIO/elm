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




def reconstruct_cmdline(test_cmd, parser, args, elm_dir=os.getcwd(), examples_dir=os.path.join(os.getcwd())):
    # Reconstruct command-line from parser and args
    actions = {action.dest: action for action in parser._actions}
    for arg, val in args._get_kwargs():
        if actions[arg].const:
            if val != actions[arg].default:
                test_cmd += ' {}'.format(actions[arg].option_strings[0])
        elif val is not None:
            if isinstance(val, (list, tuple)):
                val = ' '.join(val)
            elif isinstance(val, (str,)):
                val = '"{}"'.format(val)
            else:
                raise ValueError('Not sure how to handle argument value of type {}'.format(type(val)))
            test_cmd += ' {} {}'.format(actions[arg].option_strings[0], val)

    # Add positional arguments
    positional_args = (elm_dir, examples_dir)
    test_cmd += ' '+' '.join(positional_args)

    return test_cmd



def setup_test_env(remote_git_branch, keep_tmp_dirs):
    """This function clones the necessary repositories into temporary directories,
    pulls / checks out branches, creates a conda environment for testing,
    activates the environment, and installs necessary libraries into it.

    A teardown function is registered for when this script exits so that the
    aforementioned temporary directories and conda environment get removed.

    Return the temp directory paths for the elm repo and elm-examples repo.
    """
    # Create temporary directories for cloning/testing
    tmp_elm_dpath = tempfile.mkdtemp(dir='/tmp')
    tmp_elm_examples_dpath = tempfile.mkdtemp(dir='/tmp')

    test_env_name = os.path.basename(tmp_elm_dpath)

    def teardown_test_env():
        """Remove temporary directories and test conda env.
        """
        if not keep_tmp_dirs:
            print('\nCleaning up temporary directories...')
            shutil.rmtree(tmp_elm_dpath, ignore_errors=True)
            shutil.rmtree(tmp_elm_examples_dpath, ignore_errors=True)

        print('Removing conda environment used for testing...')
        sp.call('conda env remove -y -q -n {}'.format(test_env_name), shell=True, executable='/bin/bash', stdout=sp.DEVNULL)
    atexit.register(teardown_test_env)


    # The URLs here need to be git-based for the SSH Deploy Keys to work
    sp.check_call('git clone git@github.com:ContinuumIO/elm {}'.format(tmp_elm_dpath), shell=True, executable='/bin/bash')
    # Below is a workaround for GitHub Deploy Keys disallowing access to more than one repo
    sp.check_call('git clone git@elm-examples.github.com:ContinuumIO/elm-examples {}'.format(tmp_elm_examples_dpath), shell=True, executable='/bin/bash')
    sp.check_call('git clone https://github.com:ContinuumIO/elm-data {}'.format(os.path.join(tmp_elm_examples_dpath, 'example_data')), shell=True, executable='/bin/bash')


    # Check out the "branch-under-test" from elm repository
    sp.check_call('git pull --all', shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')
    sp.check_call('git checkout {}'.format(remote_git_branch), shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')


    # Create a conda environment suitable for testing
    sp.check_call('conda env create -n {}'.format(test_env_name), shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')
    sp.check_call('source activate {}'.format(test_env_name), shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')
    sp.check_call('python setup.py develop', shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')

    return tmp_elm_dpath, tmp_elm_examples_dpath



def main():
    # Reuse the CLI parser from elm.scripts.run_all_tests
    parser = run_all_tests.build_cli_parser(include_positional=False)
    parser.add_argument('-t', '--test', action='store_true', help='Only print the tests command - do not run the nightly tests.')
    args = parser.parse_args()

    # Provide the arguments which the user omitted
    if args.dask_clients is None:
        args.dask_clients = ['SERIAL', 'DISTRIBUTED']
    if args.dask_scheduler is None:
        args.dask_scheduler = 'localhost:8786'
    if args.remote_git_branch is None:
        args.remote_git_branch = 'master'


    # Setup test environment. This includes cloning, environment
    # setup / activation, library installation, etc...
    tmp_elm_dpath, tmp_elm_examples_dpath = setup_test_env(args.remote_git_branch, args.test)
    tmp_elm_examples_dpath = os.path.join(tmp_elm_examples_dpath)


    test_cmd = reconstruct_cmdline('elm-run-all-tests', parser, args,
                                   elm_dir=tmp_elm_dpath,
                                   examples_dir=tmp_elm_examples_dpath)
    print('\n####\nTEST COMMAND:\n\t{}\n####\n'.format(test_cmd))

    if args.test:
        sys.exit(1)

    # Run the nightly tests
    start_dt = datetime.datetime.now()
    sp.check_call(test_cmd, shell=True, cwd=tmp_elm_dpath, executable='/bin/bash')
    end_dt = datetime.datetime.now()
    print('START TIME: {}'.format(start_dt.isoformat()))
    print('END TIME: {}'.format(end_dt.isoformat()))




if __name__ == '__main__':
    main()
