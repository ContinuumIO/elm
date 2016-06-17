import calendar
import datetime
import ftplib
import os
import random
import time

from iamlp.settings import DOWNLOAD_DIR
from iamlp.cli import add_local_dataset_options
TOP_DIR = '/allData/{}/{}'
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.downloaded_args')

LADSWEB_FTP = "ladsweb.nascom.nasa.gov"

DEFAULT_DATA_GROUP_NUM = 3001
DEFAULT_DATASET = 'NPP_DSRF1KD_L2GD'
def cache_file_name(*args):
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    return os.path.join(CACHE_DIR, '_'.join(map(str, args)))

def login():
    print("Login to ftp", end=" ")
    ftp = ftplib.FTP(LADSWEB_FTP)
    ftp.login()
    print('ok')
    return ftp

def download(yr, day, ftp=None,
             product_name=DEFAULT_DATASET,
             product_number=DEFAULT_DATA_GROUP_NUM):
    day = '{:03d}'.format(day)
    top = TOP_DIR.format(product_number, product_name)
    cache_file = cache_file_name(product_number, product_name, yr, day)
    if os.path.exists(cache_file):
        return
    basedir = os.path.join(DOWNLOAD_DIR, product_number, product_name, str(yr), day)
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

def main():

    args = add_local_dataset_options().parse_args()
    today = datetime.datetime.utcnow()
    current_julian_day = sum(calendar.monthrange(today.year, x)[0] for x in range(1, today.month))
    current_julian_day += today.day - 1
    ftp = None
    for yr in args.years:
        for day in args.data_days:
            ftp = download(yr, day,
                           ftp=ftp,
                           product_name=args.product_name,
                           product_number=args.product_number)

if __name__ ==  '__main__':
    main()