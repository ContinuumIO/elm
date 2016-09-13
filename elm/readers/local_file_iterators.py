import glob
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ['VIIRS_L2_PATTERN', 'get_all_filenames_for_product'
           'iter_files_recursively', 'iter_dirs_of_dirs']

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
    ext = kwargs.get('extension', '')
    logger.debug('Read files from {} ({})'.format(top_dir, '{} extension'.format(ext) if ext else 'no file glob param'))
    for root, dirs, files in os.walk(top_dir):
        if any(os.path.isfile(f) for f in files):
            if (ext and any(f.endswith(ext) for f in files)) or not ext:
                yield root


def iter_files_recursively(**kwargs):
    path = os.environ['ELM_EXAMPLE_DATA_PATH']
    ext = kwargs.get('extension', '')
    top_dir = kwargs['top_dir']
    logger.debug('Read files from {} ({})'.format(top_dir, '{} extension'.format(ext) if ext else 'no file glob param'))

    if not path or not os.path.exists(path):
        raise ValueError('Clone the ContinuumIO/elm-data repo and '
                         'define ELM_EXAMPLE_DATA_PATH env var')
    for root, dirs, files in os.walk(top_dir):
        if files:
            if ext:
                files = (f for f in files if f.endswith(ext))
            else:
                files = iter(files)
            yield from (os.path.join(root, f) for f in files)

