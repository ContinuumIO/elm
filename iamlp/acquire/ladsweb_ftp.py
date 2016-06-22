from argparse import ArgumentParser
import calendar
import datetime
import ftplib
import os
import random
import time

from iamlp.config import ConfigParser
from iamlp.config.cli import (add_local_dataset_options, add_config_file_argument)
TOP_DIR = '/allData/{}/{}'

LADSWEB_FTP = "ladsweb.nascom.nasa.gov"

def cache_file_name(hashed_args_dir, *args):
    hash_dir = os.path.expanduser(hashed_args_dir)
    if not os.path.exists(hash_dir):
        os.mkdir(hash_dir)
    return os.path.join(hash_dir, '_'.join(map(str, args)))

def login():
    print("Login to ftp", end=" ")
    ftp = ftplib.FTP(LADSWEB_FTP)
    ftp.login()
    print('ok')
    return ftp

def download(yr, day, config,
             product_name,
             product_number,
             ftp=None,):

    LADSWEB_LOCAL_CACHE = config.LADSWEB_LOCAL_CACHE
    HASHED_ARGS_CACHE = config.HASHED_ARGS_CACHE
    product_number = str(product_number)
    day = '{:03d}'.format(day)
    top = TOP_DIR.format(product_number, product_name)
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
        fhandle = open(local_f, 'wb')
        ftp.retrbinary('RETR ' + f, fhandle.write)
        fhandle.close()
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
    args.years = args.years or [2016]
    args.data_days = args.data_days or list(range(1, 366))
    if args.data_days in ('all', ['all']):
        args.data_days = list(range(1, 366))
    print("Running with args {}".format(args))
    for yr in args.years:
        for day in args.data_days:
            ftp = download(yr, day, config,
                           args.product_name,
                           args.product_number,
                           ftp=ftp)

if __name__ ==  '__main__':
    main()