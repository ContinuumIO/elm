import glob
import os

import numpy as np
import pytest

from elm.readers.hdf4 import (load_hdf4_meta,
                              load_hdf4_array)
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH,
                                    HDF4_FILES,
                                    assertions_on_metadata,
                                    assertions_on_band_metadata)

HDF4_DIR = os.path.dirname(HDF4_FILES[0])

band_specs = [
    ['long_name', 'Band 1 ', 'band_1'],
    ['long_name', 'Band 2 ', 'band_2'],
    ['long_name', 'Band 3 ', 'band_3'],
    ['long_name', 'Band 4 ', 'band_4'],
    ['long_name', 'Band 5 ', 'band_5'],
    ['long_name', 'Band 7 ', 'band_7'],
    ['long_name', 'Band 8 ', 'band_8'],
    ['long_name', 'Band 10 ', 'band_10'],
    ['long_name', 'Band 11 ', 'band_11'],
]


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_meta():
    for hdf in HDF4_FILES:
        meta = load_hdf4_meta(hdf)
        assertions_on_metadata(meta, is_band_specific=False)
        assert 'GranuleBeginningDateTime' in meta['MetaData']
        for band_meta in meta['BandMetaData']:
            assert 'GranuleBeginningDateTime' in band_meta


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_read_array():
    for hdf in HDF4_FILES:
        meta = load_hdf4_meta(hdf)
        es = load_hdf4_array(hdf, meta, band_specs)
        for band in es.data_vars:
            sample = getattr(es, band)
            mean_y = np.mean(sample.y)
            mean_x = np.mean(sample.x)
            band_names = np.array([b[-1] for b in band_specs])
            assert sorted((mean_x,
                    sample.Bounds.left,
                    sample.Bounds.right))[1] == mean_x
            assert sorted((mean_y,
                    sample.Bounds.top,
                    sample.Bounds.bottom))[1] == mean_y
            assert sample.y.size == 1200
            assert sample.x.size == 1200
            assert len(es.data_vars) == len(band_specs)
            assert np.all(es.BandOrder == [x[-1] for x in band_specs])
            assertions_on_band_metadata(sample.attrs)
        es2 = load_hdf4_array(hdf, meta, band_specs=None)
        assert len(es2.data_vars) > len(es.data_vars)