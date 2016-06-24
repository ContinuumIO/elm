from argparse import ArgumentParser
import calendar
import datetime
import ftplib
import json
import os
import pandas as pd
import shutil
import sys
import random
import time
import traceback

from iamlp.config import ConfigParser
from iamlp.config.cli import (add_local_dataset_options,
                              add_config_file_argument,
                              add_sample_ladsweb_options)
from iamlp.config.env import parse_env_vars

TOP_DIR = '/allData'
PRODUCT_FORMAT = TOP_DIR + '/{}/{}'

LADSWEB_FTP = "ladsweb.nascom.nasa.gov"

def cache_file_name(hashed_args_dir, *args):
    hash_dir = os.path.expanduser(hashed_args_dir)
    if not os.path.exists(hash_dir):
        os.mkdir(hash_dir)
    return os.path.join(hash_dir, '_'.join(map(str, args)))

def login():
    print("Logging into ftp...", end=" ", flush=True)
    ftp = ftplib.FTP(LADSWEB_FTP)
    ftp.login()
    print('ok')
    return ftp

def product_meta_file(LADSWEB_LOCAL_CACHE, product_number):
    d = os.path.join(LADSWEB_LOCAL_CACHE, 'meta')
    if not os.path.exists(d):
        os.mkdir(d)
    return os.path.join(d, 'product_meta_{}.json'.format(product_number))

def _ftp_is_file(ftp, path):
    current = ftp.pwd()
    try:
        ftp.cwd(path)
        return False
    except:
        ftp.cwd(current)
        return True

def ftp_to_local_path(ftp_path, base_dir):
    return ftp_path.replace(TOP_DIR, base_dir)


def ftp_walk(root, ftp, LADSWEB_LOCAL_CACHE, break_depth, at_each_level, depth=0, max_depth=3, has_yielded=False):
    ls = []
    keys = []
    if _ftp_is_file(ftp, root):
        yield root.split('/')
    elif depth <= max_depth:
        for idx1, item in enumerate(ftp_ls(ftp, dirname=root)):
            for idx2, item2 in enumerate(ftp_walk(os.path.join(root, item), ftp,
                                  LADSWEB_LOCAL_CACHE, break_depth, at_each_level,
                                  depth=depth + 1,
                                  max_depth=max_depth, has_yielded=has_yielded)):
                yield item2
                if depth > break_depth:
                    break
                if idx2 > at_each_level:
                    break
            if depth > break_depth:
                break
            if idx1 > at_each_level:
                break
    else:
        return

def ftp_ls(ftp, dirname=None):
    ls = []
    if dirname:
        current = ftp.pwd()
        ftp.cwd(dirname)
    ftp.retrlines('NLST', ls.append)
    if dirname:
        ftp.cwd(current)
    return ls

def _update_counts_dict(counts_dict, file_parts):
    file_parts = file_parts[3:]
    def inc(d, key):
        if not key in d:
            d[key] = 0
        d[key] += 1
    d = counts_dict
    if len(file_parts) < 2:
        return
    for f in file_parts[:-2]:
        if not f in d or isinstance(d[f], int):
            d[f] = {}
        d = d[f]
    f = file_parts[-2]
    if not f in d or not isinstance(d[f], int):
        d[f] = 0
    d[f] += 1

def get_sample_of_ladsweb_products(year=None, data_day=None,
                                   n_file_samples=1, ftp=None,
                                   ignore_downloaded=False):
    env = parse_env_vars()
    if not 'LADSWEB_LOCAL_CACHE' in env:
        raise ValueError('Define the LADSWEB_LOCAL_CACHE env var to a download location')
    LADSWEB_LOCAL_CACHE = env['LADSWEB_LOCAL_CACHE']
    if ftp is None:
        ftp = login()
    print('Logged into ftp')
    def write_results(meta_file, results):
        d = os.path.dirname(meta_file)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(meta_file, 'w') as f:
            f.write(json.dumps(results))
    product_numbers = []
    meta_file = product_meta_file(LADSWEB_LOCAL_CACHE, 'unique_product_numbers')
    ftp.cwd(TOP_DIR)
    product_numbers = ftp_ls(ftp)
    print('There are {} product numbers'.format(len(product_numbers)))
    write_results(meta_file, product_numbers)
    data_day_str = '{:03d}'.format(data_day) if data_day else None

    for pidx, p in enumerate(product_numbers):
        print('Product number {} ({} out of {})'.format(p, pidx, len(product_numbers)))
        product_dict = {}
        prod_dir = os.path.join(TOP_DIR, str(p))
        ftp.cwd(prod_dir)
        product_names = ftp_ls(ftp)
        product_dict['product_names'] = product_names
        for nidx, name in enumerate(product_names):
            sample_files = 0
            meta_file = product_meta_file(LADSWEB_LOCAL_CACHE,
                                          '{}_{}'.format(p, name))
            if os.path.exists(meta_file) and not ignore_downloaded:
                continue
            print('Product name {} ({} out of {})'.format(name, nidx, len(product_names)))
            prod_name_dir = os.path.join(prod_dir, name)
            counts_dict = {}
            cache_file = cache_file_name('product_meta', str(p), name)
            if not ignore_downloaded and os.path.exists(cache_file):
                continue
            for file_parts in ftp_walk(prod_name_dir, ftp, LADSWEB_LOCAL_CACHE, 1, 1,
                                       max_depth=4, has_yielded=False):
                _update_counts_dict(counts_dict, file_parts)
                print(file_parts)
                if sample_files < n_file_samples:
                    remote_f = '/'.join(file_parts)
                    local_f  = ftp_to_local_path(remote_f, LADSWEB_LOCAL_CACHE)
                    if not _try_download(local_f, remote_f, ftp,
                                        skip_on_fail=True,
                                        ignore_downloaded=ignore_downloaded):
                        counts_dict['failed_on_download'] = remote_f
                    sample_files += 1
                else:
                    # Note: Could not break here to count all files
                    break
            with open(cache_file, 'w') as f:
                f.write('download {}'.format(datetime.datetime.now().isoformat()))
            write_results(meta_file, counts_dict)

def get_sample_main(args=None):
    if args is None:
        parser = ArgumentParser(description='Collect metadata on each product on ladsweb')
        parser = add_sample_ladsweb_options(parser)
        args = parser.parse_args()
    print('Running with args: {}'.format(args))
    get_sample_of_ladsweb_products(year=args.year,
                                   data_day=args.data_day,
                                   n_file_samples=args.n_file_samples)

def _try_download(local_f, remote_f, ftp, skip_on_fail=False,
                  ignore_downloaded=False):
    if os.path.exists(local_f) and not ignore_downloaded:
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
            print('Failed on retrbinary', repr(e), traceback.format_exc(), file=sys.stderr)
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
        print('Download', f, end=' ')
        _try_download(local_f, f, ftp)
        print('ok')
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
    print("Running with args {}".format(args))
    for yr in args.years:
        for day in args.data_days:
            ftp = download(yr, day, config,
                           args.product_name,
                           args.product_number,
                           ftp=ftp)

if __name__ ==  '__main__':
    main()