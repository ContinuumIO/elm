import glob
import os


def VIIRS_L2_PATTERN(product_number, product_name, yr, data_day):
    return os.path.join(LADSWEB_LOCAL_CACHE, str(product_number), product_name, str(yr),
                      '{:03d}'.format(data_day), '*.hdf')

def get_all_filenames_for_product(data_source):
    product_name = data_source['product_name']
    product_number = data_source['product_number']
    pattern = data_source['file_pattern']
    LADSWEB_LOCAL_CACHE = data_source['LADSWEB_LOCAL_CACHE']
    product_name_dir = os.path.join(LADSWEB_LOCAL_CACHE, str(product_number), product_name)
    if data_source['years'] == ['all']:
        yr_dir_gen = (yr_dir for yr_dir in glob.glob(os.path.join(prod_name_dir, '*')))
    else:
        yr_dir_gen = (os.path.join(product_name_dir, str(yr)) for yr in data_source['years'])
    for yr_dir in yr_dir_gen:
        if data_source['data_days'] in (['all'], 'all'):
            day_dirs = glob.glob(os.path.join(yr_dir, '*'))
        else:
            day_dirs = (os.path.join(yr_dir, '{:03d}'.format(data_day)) for data_day in data_source['data_days'])
        for day in day_dirs:
            for f in glob.glob(os.path.join(day, pattern)):
                yield f


def iter_dirs_of_dirs(**kwargs):
    top_dir = kwargs['top_dir']
    for root, dirs, files in os.walk(top_dir):
        if any(f.lower().endswith('tif') or f.lower().endswith('tiff') for f in files):
            yield root