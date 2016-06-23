from argparse import ArgumentParser
import calendar
import datetime
import ftplib
import os
import random
import time

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
    def write_results(meta_file, results):
        with open(meta_file, 'w') as f:
            f.write(yaml.dump(results))
    print('ls the product_numbers')
    product_numbers = []
    meta_file = product_meta_file(examples_dir, 'unique_product_numbers')
    ftp.cwd(TOP_DIR)
    ftp.retrlines('NLST', product_numbers.append)
    print('There are {} product numbers'.format(len(product_numbers)))
    write_results(meta_file, product_numbers)
    for idx, p in enumerate(product_numbers):
        meta_file = product_meta_file(examples_dir, str(p))
        if os.path.exists(meta_file):
            print('Skip product {} - already downloaded'.format(p))
            continue
        product_dict = {}
        prod_dir = os.path.join(TOP_DIR, str(p))
        number_to_name[str(p)] = []
        ftp.cwd(prod_dir)
        product_names = []
        ftp.retrlines('NLST', product_names)
        product_dict['product_names'] = product_names
        for idx, d in enumerate(product_names):
            prod_name_dir = os.path.join(prod_dir, name)
            years = []
            product_dict[name] = {}
            ftp.cwd(prod_name_dir)
            ftp.retrlines('NLST', years)
            product_dict[name]['years'] = years
            if str(year) in years:
                data_day_dir = os.path.join(prod_name_dir, str(year),
                                        '{:03d}'.format(data_day))

                file_ls = []
                ftp.cwd(data_day_dir)
                ftp.retrlines('NLST', file_ls)
                product_dict[name]['example_ls'] = file_ls
                product_dict[name]['example_ls_year'] = year
                product_dict[name]['example_ls_data_day'] = data_day
                unique_exts = {f.split('.')[-1] for f in file_ls}
                product_dict[[name]]['examples'] = {}
                for ext in unique_exts:
                    example = [f for f in file_ls if f.endswith(ext)][0]
                    one_file = os.path.join(data_day_dir, example)
                    local_f = os.path.join(examples_dir, prod_name_dir, example)
                    product_dict[[name]]['examples'][ext] = example
                    _try_download(local_f, one_file, ftp)
                print('Product {} {} has {} file type'.format(p,name))
        write_results(meta_file, product_dict)
        print('Done with {} out of {} product_numbers'.format(idx, len(product_numbers)))

def get_sample_main(args=None):
    if args is None:
        parser = ArgumentParser(description='Collect metadata on each product on ladsweb')
        parser = add_sample_ladsweb_options(parser)
        args = parser.parse_args()
    print('Running with args: {}'.format(args))
    get_sample_of_ladsweb_products(year=args.year,
                                   data_day=args.data_day,
                                   n_file_samples=args.n_file_samples)

def _try_download(local_f, remote_f, ftp):
    fhandle = None
    try:
        fhandle = open(local_f, 'wb')
        ftp.retrbinary('RETR ' + remote_f, fhandle.write)
        fhandle.close()
    except Exception as e:
        if fhandle and os.path.exists(local_f):
            os.remove(local_f)
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