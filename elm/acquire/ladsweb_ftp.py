from argparse import ArgumentParser
import calendar
import datetime
import ftplib
import json
import logging
import os
import pandas as pd
import shutil
import sys
import random
import time
import traceback

from elm.config import ConfigParser
from elm.config.cli import (add_local_dataset_options,
                              add_config_file_argument,
                              add_sample_ladsweb_options)
from elm.config.env import parse_env_vars

logger = logging.getLogger(__name__)

TOP_DIR = '/allData'
PRODUCT_FORMAT = TOP_DIR + '/{}/{}'

LADSWEB_FTP = "ladsweb.nascom.nasa.gov"

def cache_file_name(hashed_args_dir, *args):
    '''This does not save a data file but
    rather a small file indicating an operation has been done
     '''
    hash_dir = os.path.expanduser(hashed_args_dir)
    if not os.path.exists(hash_dir):
        os.mkdir(hash_dir)
    return os.path.join(hash_dir, '_'.join(map(str, args)))

def login():
    '''Login to ladsweb ftp, return ftp object'''
    logger.info("Logging into ftp...")
    ftp = ftplib.FTP(LADSWEB_FTP)
    ftp.login()
    logger.info('ok ftp login')
    return ftp

def product_meta_file(LADSWEB_LOCAL_CACHE, tag):
    d = os.path.join(LADSWEB_LOCAL_CACHE, 'meta')
    if not os.path.exists(d):
        os.mkdir(d)
    return os.path.join(d, 'product_meta_{}.json'.format(tag))

def _ftp_is_file(ftp, path):
    '''Check if an ftp path is a file (True) or dir (False)
    Note: non existent paths or no-permission paths also are False
    '''
    current = ftp.pwd()
    try:
        ftp.cwd(path)
        return False
    except:
        ftp.cwd(current)
        return True

def ftp_to_local_path(ftp_path, base_dir):
    '''Make a local path based on an ftp_path.

    This keeps the local data organized like the ftp
    system'''
    return ftp_path.replace(TOP_DIR, base_dir)


def ftp_walk(root, ftp,
             at_each_level,
             depth=0, max_depth=3, break_depth=2):
    '''Yield recursively ftp contents in root under constraints
    Params:
        root: starting dir
        ftp: ftp object
        break_depth: beyond this graph depth, break for loops
        at_each_level: after this many yields of a dirs contents, go
                       to next level if it exists
        depth: used internally
        max_depth: max depth of recursion

    '''
    ls = []
    keys = []
    if _ftp_is_file(ftp, root):
        yield root.split('/')
    elif depth <= max_depth:
        for idx1, item in enumerate(ftp_ls(ftp, dirname=root)):
            yield from ftp_walk(os.path.join(root, item), ftp,
                                  at_each_level,
                                  depth=depth + 1,
                                  max_depth=max_depth)
            if depth > break_depth:
                break
            if idx1 > at_each_level:
                break
    else:
        return

def ftp_ls(ftp, dirname=None):
    '''ls on ftp current working directory or in dirname if given
    Returns: list of contents (not joined to cwd)
    '''
    ls = []
    if dirname:
        current = ftp.pwd()
        ftp.cwd(dirname)
    ftp.retrlines('NLST', ls.append)
    if dirname:
        ftp.cwd(current)
    return ls

def get_sample_of_ladsweb_products(product_numbers=None, product_names=None,
                                   n_file_samples=1, ftp=None):
    '''
    Iterate over ladsweb products and get a file sample

    Params:
        product_numbers: list of string product numbers or None for all products
        product_names:   list of string product names or None for all product names in product_numbers
        n_file_samples:  how many file samples of each product number/name

    '''
    env = parse_env_vars()
    if not 'LADSWEB_LOCAL_CACHE' in env:
        raise ValueError('Define the LADSWEB_LOCAL_CACHE env var to a download location')
    LADSWEB_LOCAL_CACHE = env['LADSWEB_LOCAL_CACHE']
    if ftp is None:
        ftp = login()
    logger.info('Logged into ftp')
    def write_results(meta_file, results):
        d = os.path.dirname(meta_file)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(meta_file, 'w') as f:
            f.write(json.dumps(results))
    product_numbers = product_numbers or []
    meta_file = product_meta_file(LADSWEB_LOCAL_CACHE, 'unique_product_numbers')
    if not product_numbers:
        ftp.cwd(TOP_DIR)
        product_numbers = ftp_ls(ftp)
    write_results(meta_file, product_numbers)
    logger.info('There are {} product numbers'.format(len(product_numbers)))
    for pidx, p in enumerate(product_numbers):
        logger.info('Product number {} ({} out of {})'.format(p, pidx, len(product_numbers)))
        prod_dir = os.path.join(TOP_DIR, str(p))
        if not product_names:
            ftp.cwd(prod_dir)
            product_names = ftp_ls(ftp)
        for nidx, name in enumerate(product_names):
            product_dict = {}
            sample_files = 0
            meta_file = product_meta_file(LADSWEB_LOCAL_CACHE,
                                          '{}_{}'.format(p, name))
            logger.info('Product name {} ({} out of {})'.format(name, nidx, len(product_names)))
            prod_name_dir = os.path.join(prod_dir, name)
            counts_dict = {}
            cache_file = cache_file_name('product_meta', str(p), name)
            if os.path.exists(cache_file):
                continue
            for file_parts in ftp_walk(prod_name_dir, ftp, 1, 1,
                                       max_depth=4):
                logger.info('/'.join(file_parts))
                if sample_files < n_file_samples:
                    remote_f = '/'.join(file_parts)
                    local_f  = ftp_to_local_path(remote_f, LADSWEB_LOCAL_CACHE)
                    if not _try_download(local_f, remote_f, ftp,
                                        skip_on_fail=True,
                                        ):
                        counts_dict['failed_on_download'] = remote_f
                    sample_files += 1
                else:
                    # Note: Could not break here to count all files
                    break
            with open(cache_file, 'w') as f:
                f.write('download {}'.format(datetime.datetime.now().isoformat()))
            write_results(meta_file, counts_dict)

def get_sample_main(args=None, parse_this=None):
    '''This is a console entry point for getting a sample of each ladsweb data source'''
    if args is None:
        parser = ArgumentParser(description='Collect metadata on each product on ladsweb')
        parser = add_sample_ladsweb_options(parser)
        if parse_this:
            args = parser.parse_args(parse_this)
        else:
            args = parser.parse_args()
    logger.info('Running with args: {}'.format(args))
    get_sample_of_ladsweb_products(**vars(args))

def _try_download(local_f, remote_f, ftp, skip_on_fail=False):
    if os.path.exists(local_f):
        return True
    fhandle = None
    d = os.path.dirname(local_f)
    made_dir = False
    if not os.path.exists(d):
        os.makedirs(d)
        made_dir = True
    try:
        fhandle = open(local_f, 'wb')
        ftp.retrbinary('RETR ' + remote_f, fhandle.write)
        fhandle.close()
        return True
    except Exception as e:
        if fhandle and os.path.exists(local_f):
            os.remove(local_f)
        if made_dir:
            shutil.rmtree(d)
        if skip_on_fail:
            logger.info('Failed on retrbinary '
                        '{} {} {}'.format(remote_f,
                                          repr(e),
                                          traceback.format_exc()))
            return False
        raise

def download(yr, day, config,
             product_name,
             product_number,
             ftp=None,):

    LADSWEB_LOCAL_CACHE = config.LADSWEB_LOCAL_CACHE
    HASHED_ARGS_CACHE = config.HASHED_ARGS_CACHE
    product_number = str(product_number)
    day = '{:03d}'.format(day)
    top = PRODUCT_FORMAT.format(product_number, product_name)
    cache_file = cache_file_name(HASHED_ARGS_CACHE, product_number, product_name, yr, day)
    if os.path.exists(cache_file):
        return
    basedir = os.path.join(LADSWEB_LOCAL_CACHE, product_number, product_name, str(yr), day)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    ftp = ftp or login()
    ftp.cwd(os.path.join(top, str(yr), day))
    ls = ftp_ls(ftp)
    for f in ls:
        local_f = os.path.join(basedir, f)
        if os.path.exists(local_f):
            continue
        logger.info('Download', f, end=' ')
        _try_download(local_f, f, ftp)
        logger.info('ok')
        time.sleep(random.uniform(0.2, 0.6))
    with open(cache_file, 'w') as f:
        f.write('Downloaded {}'.format(datetime.datetime.now().isoformat()))

    return ftp

def main(args=None, parse_this_str=None, config=None):
    parser = ArgumentParser(description="Download util for {}".format(LADSWEB_FTP))
    parser = add_local_dataset_options(parser)
    if config is None:
        parser = add_config_file_argument(parser)
    if args is None:
        if not parse_this_str:
            args = parser.parse_args()
        else:
            args = parser.parse_args(parse_this_str)
    if config is None:
        config = ConfigParser(args.config)
    today = datetime.datetime.utcnow()
    current_julian_day = sum(calendar.monthrange(today.year, x)[0] for x in range(1, today.month))
    current_julian_day += today.day - 1
    ftp = None
    logger.info("Running with args {}".format(args))
    for yr in args.years:
        for day in args.data_days:
            ftp = download(yr, day, config,
                           args.product_name,
                           args.product_number,
                           ftp=ftp)

if __name__ ==  '__main__':
    main()