#!/usr/bin/env python3

import os
import datetime
import argparse
import sys
from elm.scripts import run_all_tests


# Reuse the CLI parser from elm.scripts.run_all_tests
parser = run_all_tests.build_cli_parser(include_required=False)
args = parser.parse_args()

# Provide the arguments which the user omitted
if args.dask_clients is None:
    args.dask_clients = ['SERIAL', 'DISTRIBUTED']
if args.dask_scheduler is None:
    args.dask_scheduler = 'localhost:8786'
if args.remote_git_branch is None:
    args.remote_git_branch = 'master'
args.repo_dir = os.getcwd()
args.elm_configs_path = os.path.join(args.repo_dir, 'example_configs')



print('PWD = {}'.format(os.getcwd()))
print('PATH = {}'.format(os.environ.get('PATH')))
print('dask_clients = {}'.format(args.dask_clients))
print('dask_scheduler = {}'.format(args.dask_scheduler))
print('remote_git_branch = {}'.format(args.remote_git_branch))
print('repo_dir = {}'.format(args.repo_dir))
print('elm_configs_path = {}'.format(args.elm_configs_path))



# Run the nightly tests
start_dt = datetime.datetime.now()

run_all_tests.run_all_tests(args)

end_dt = datetime.datetime.now()
print('\nSTART TIME: {}'.format(start_dt.isoformat()))
print('\nEND TIME: {}'.format(end_dt.isoformat()))



