import pytest

from elm.data_selection.filename_selection import get_included_filenames
from elm.example_data import EXAMPLE_FILES

EXAMPLE_BAND_SPECS = [['long_name', 'Band 1 ', 'band_1',],
                      ['long_name', 'Band 2',  'band_2']]

@pytest.mark.needs_examples
def test_get_included_filenames():

    def filenames_gen():
        for f in EXAMPLE_FILES['hdf']:
            yield f
    for no_file_open in (True, False):
        files = get_included_filenames(filenames_gen,
                                       EXAMPLE_BAND_SPECS,
                                       no_file_open=no_file_open)
    assert sorted(files) == sorted(EXAMPLE_FILES['hdf'])
    nothing_matches = [['nothing_matches'] * 3 for _ in range(8)]
    with pytest.raises(ValueError):
        files = get_included_filenames(filenames_gen, nothing_matches, no_file_open=False)

