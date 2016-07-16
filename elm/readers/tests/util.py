
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
else:
    ELM_EXAMPLE_DATA_PATH = None
    TIF_FILES = None
    HDF5_FILES = None

def assertions_on_metadata(meta, is_band_specific=False):
    required_keys = ('GeoTransform',
                     'MetaData',
                     'Bounds',
                     'Height',
                     'Width')
    if not is_band_specific:
        required_keys += ('BandMetaData',)
    for key in required_keys:
        assert key in meta