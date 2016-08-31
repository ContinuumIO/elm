import pytest

from elm.sample_util.filename_selection import get_generated_args
from elm.example_data import EXAMPLE_FILES
from elm.readers.hdf4 import load_hdf4_array, load_hdf4_meta

EXAMPLE_BAND_SPECS = [['long_name', 'Band 1 ', 'band_1',],
                      ['long_name', 'Band 2',  'band_2']]

@pytest.mark.needs_examples
def test_get_generated_args():

    def filenames_gen(**kwargs):
        for f in EXAMPLE_FILES['hdf']:
            yield f
    files = get_generated_args(filenames_gen,
                               EXAMPLE_BAND_SPECS,
                               load_meta=load_hdf4_meta,
                               load_array=load_hdf4_array)
    assert files
