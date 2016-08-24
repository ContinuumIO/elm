
import glob
import os
from elm.config import parse_env_vars

ENV = parse_env_vars()
ELM_HAS_EXAMPLES = ENV['ELM_HAS_EXAMPLES']
if ELM_HAS_EXAMPLES:
    ELM_EXAMPLE_DATA_PATH = ENV['ELM_EXAMPLE_DATA_PATH']
    TIF_FILES = glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH,
                                       'tif',
                                       'L8',
                                       '015',
                                       '033',
                                       'LC80150332013207LGN00',
                                       '*.TIF'))
    HDF5_FILES = glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH,
                                        'hdf5',
                                        '2016',
                                        '01',
                                        '01',
                                        'imerg',
                                        '*.HDF5'))
    HDF4_FILES = glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH,
                                        'hdf4',
                                        '*.hdf'))
    NETCDF_FILES = glob.glob(os.path.join(ELM_EXAMPLE_DATA_PATH,
                                          'netcdf',
                                          '*.nc'))
else:
    ELM_EXAMPLE_DATA_PATH = None
    TIF_FILES = []
    HDF5_FILES = []
    HDF4_FILES = []
    NETCDF_FILES = []

def assertions_on_metadata(meta, is_band_specific=False):
    required_keys = ('meta',
                     'height',
                     'width')
    if not is_band_specific:
        required_keys += ('band_meta',)
    for key in required_keys:
        assert key in meta

def assertions_on_band_metadata(band_meta):
    required_keys = ('geo_transform', 'bounds')
    for key in required_keys:
        assert key in band_meta
