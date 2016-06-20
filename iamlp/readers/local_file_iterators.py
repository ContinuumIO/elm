import glob
import os

from iamlp.settings import DOWNLOAD_DIR, delayed

def VIIRS_L2_PATTERN(product_number, product_name, yr, data_day):
    return os.path.join(DOWNLOAD_DIR, str(product_number), product_name, str(yr),
                      '{:03d}'.format(data_day), '*.hdf')

def get_filenames_for_day(product_number, product_name, yr, data_day,
                          pattern=VIIRS_L2_PATTERN):
    for filename in glob.glob(pattern(product_name, yr, data_day)):
        yield filename

def get_all_filenames_for_product(product_number, product_name,
                                  pattern='*.hdf'):

    for yr_dir in glob.glob(os.path.join(DOWNLOAD_DIR, str(product_number), product_name, '*')):
        for day in glob.glob(os.path.join(yr_dir, '*')):
            for f in glob.glob(os.path.join(day, pattern)):
                yield f