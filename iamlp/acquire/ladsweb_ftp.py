from argparse import ArgumentParser
import calendar
import datetime
import ftplib
import json
import os
import pandas as pd
import sys
import random
import time
import traceback

from iamlp.config import ConfigParser
from iamlp.config.cli import (add_local_dataset_options,
                              add_config_file_argument,
                              add_sample_ladsweb_options)
TOP_DIR = '/allData/'
PRODUCT_FORMAT = TOP_DIR + '{}/{}'

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

def product_meta_file(examples_dir, product_number):
    return os.path.join(examples_dir,
                        'product_meta_{}.json'.format(product_number))

def _ftp_is_file(ftp, path):
    try:
        ftp.cwd(path)
        return False
    except:
        return True
def ftp_to_local_path(ftp_path, LADSWEB_LOCAL_CACHE):
    return os.path.join(LADSWEB_LOCAL_CACHE, ftp_path.replace(TOP_DIR, ''))

has_seen = set()
def ftp_walk(root, ftp, LADSWEB_LOCAL_CACHE, depth=0, max_depth=3):
    global has_seen
    global stop
    ls = []
    keys = []
    if _ftp_is_file(ftp, root):
        yield ('file', root, ftp_to_local_path(root, LADSWEB_LOCAL_CACHE))
    elif depth <= max_depth - 1:
        ftp.cwd(root)
        ftp.retrlines('NLST', ls.append)
        for item in ls:
            for item2 in ftp_walk(os.path.join(root, item), ftp, LADSWEB_LOCAL_CACHE,
                                depth=depth + 1, max_depth=max_depth):
                yield item2



def get_sample_of_ladsweb_products(year=2015, data_day=1,
                         n_file_samples=1, ftp=None):
    from iamlp.config.env import parse_env_vars
    env = parse_env_vars()
    if not 'LADSWEB_LOCAL_CACHE' in env:
        raise ValueError('Define the LADSWEB_LOCAL_CACHE env var to a download location')
    LADSWEB_LOCAL_CACHE = env['LADSWEB_LOCAL_CACHE']
    if ftp is None:
        ftp = login()
    print('Logged into ftp')
    examples_dir = os.path.join(LADSWEB_LOCAL_CACHE, 'examples')
    ls_recursive = []
    def write_results(meta_file, results):
        d = os.path.dirname(meta_file)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(meta_file, 'w') as f:
            f.write(json.dumps(results))
    col1, col2, col3 = [], [], []
    for item in ftp_walk(TOP_DIR, ftp, LADSWEB_LOCAL_CACHE, max_depth=5):
        is_file, remote_f, local_f = item
        col1.append(is_file)
        col2.append(remote_f)
        col3.append(local_f)
        ls_recursive.append(list(item))
    df = pd.DataFrame({'file_or_dir': col1,
                       'remote_file': col2,
                       'local_f': col3})
    df.to_csv(product_meta_file(examples_dir, 'ls_recursive'), index=False)

def get_sample_of_ladsweb_products(year=None, data_day=None,
                                   n_file_samples=1, ftp=None):
    from iamlp.config.env import parse_env_vars
    env = parse_env_vars()
    if not 'LADSWEB_LOCAL_CACHE' in env:
        raise ValueError('Define the LADSWEB_LOCAL_CACHE env var to a download location')
    LADSWEB_LOCAL_CACHE = env['LADSWEB_LOCAL_CACHE']
    if ftp is None:
        ftp = login()
    print('Logged into ftp')
    examples_dir = os.path.join(LADSWEB_LOCAL_CACHE, 'examples')
    def write_results(meta_file, results):
        d = os.path.dirname(meta_file)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(meta_file, 'w') as f:
            f.write(json.dumps(results))
    product_numbers = []
    meta_file = product_meta_file(examples_dir, 'unique_product_numbers')
    ftp.cwd(TOP_DIR)
    ftp.retrlines('NLST', product_numbers.append)
    print('There are {} product numbers'.format(len(product_numbers)))
    write_results(meta_file, product_numbers)
    data_day_str = '{:03d}'.format(data_day)

    for pidx, p in enumerate(product_numbers):
        meta_file = product_meta_file(examples_dir, str(p))
        if os.path.exists(meta_file):
            print('Skip product {} - already downloaded'.format(p))
            continue
        product_dict = {}
        prod_dir = os.path.join(TOP_DIR, str(p))
        ftp.cwd(prod_dir)
        product_names = []
        ftp.retrlines('NLST', product_names.append)
        product_dict['product_names'] = product_names
        for nidx, name in enumerate(product_names):
            prod_name_dir = os.path.join(prod_dir, name)
            years = []
            product_dict[name] = {}
            ftp.cwd(prod_name_dir)
            ftp.retrlines('NLST', years.append)
            product_dict[name]['years'] = years
            if str(year) in years or not year:
                if not year:
                    year = [year for year in years if '19' in year or '20' in year]
                    if not year:
                        continue
                    year = year[0]
                yr_dir = os.path.join(prod_name_dir, str(year))
                ftp.cwd(yr_dir)
                yr_dir_ls = []
                ftp.retrlines('NLST', yr_dir_ls.append)
                product_dict[name]['yr_dir_ls'] = yr_dir_ls
                is_data_days = True
                for item in yr_dir_ls:
                    try:
                        int_item = int(item)
                        if not (int_item >=1 and int_item <=365):
                            is_data_days = False
                            break
                    except Exception as e:
                        is_data_days = False
                        break

                if is_data_days and data_day_str in yr_dir_ls:
                    next_dir = os.path.join(yr_dir, data_day_str)
                elif yr_dir_ls:
                    next_dir = os.path.join(yr_dir, yr_dir_ls[0])
                else:
                    print('Nothing in ', yr_dir)
                    continue
                file_ls = []
                ftp.cwd(next_dir)
                ftp.retrlines('NLST', file_ls.append)
                product_dict[name]['example_ls'] = file_ls
                product_dict[name]['example_ls_year'] = year
                if is_data_days:
                    product_dict[name]['example_ls_data_day'] = data_day
                else:
                    product_dict[name]['ls_not_data_day'] = yr_dir_ls
                exts = (f.split('.') for f in file_ls)
                exts = {f[-1] for f in exts if len(f) > 1}
                product_dict[name]['examples'] = {}
                if exts:
                    for ext in exts:
                        example = [f for f in file_ls if f.endswith(ext)][0]
                        one_file = os.path.join(next_dir, example)
                        local_f = os.path.join(examples_dir, str(p), name, str(year),
                                               os.path.basename(next_dir), example)
                        product_dict[name]['examples'][ext] = example
                        if not _try_download(local_f, one_file, ftp, skip_on_fail=True):
                            print('Could not retrbinary on {} - maybe it is a dir'.format(one_file))
                            product_dict[name]['failed_on_retr'] = one_file
                            try:
                                ftp.cwd(one_file)
                                last_ls = []
                                ftp.retrlines('NLST', last_ls.append)
                                product_dict[name]['failed_on_retr_ls'] = last_ls
                            except Exception as e:
                                print('Failed on last_ls with ', repr(e))

                else:
                    product_dict[name]['examples']['no-extensions'] = file_ls
                    print('file_ls', file_ls)

                print('Product {} {} has {} file types'.format(p, name, exts))
            else:
                product_dict[name]['not_years'] = years
                print('not_years', years)
        pcent = pidx / len(product_numbers) * 100
        write_results(meta_file, product_dict)
        print('Done with {} out of {} product_numbers'.format(pidx, len(product_numbers)))
        print('{}% complete'.format(pcent))
def get_sample_main(args=None):
    if args is None:
        parser = ArgumentParser(description='Collect metadata on each product on ladsweb')
        parser = add_sample_ladsweb_options(parser)
        args = parser.parse_args()
    print('Running with args: {}'.format(args))
    get_sample_of_ladsweb_products(year=args.year,
                                   data_day=args.data_day,
                                   n_file_samples=args.n_file_samples)

def _try_download(local_f, remote_f, ftp, skip_on_fail=False):
    fhandle = None
    d = os.path.dirname(local_f)
    if not os.path.exists(d):
        os.makedirs(d)
    try:
        fhandle = open(local_f, 'wb')
        ftp.retrbinary('RETR ' + remote_f, fhandle.write)
        fhandle.close()
        return True
    except Exception as e:
        if fhandle and os.path.exists(local_f):
            os.remove(local_f)
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
    ls = []
    ftp.retrlines('NLST', ls.append)
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