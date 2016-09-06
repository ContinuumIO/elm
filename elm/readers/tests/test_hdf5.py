import glob
import gdal
import os

import numpy as np
import pytest

from elm.readers.hdf5 import (load_hdf5_meta,
                              load_array,
                              load_hdf5_array)
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH,
                                    HDF5_FILES,
                                    assertions_on_metadata,
                                    assertions_on_band_metadata)
from elm.readers.util import BandSpec

HDF5_DIR = os.path.dirname(HDF5_FILES[0])

band_specs = [BandSpec(search_key='FileName',
                       search_value='3B-HHR.MS.MRG.3IMERG.20160101-S220000-E222959.1320.V03D.HDF5',
                       name='mydata')]

@pytest.mark.parametrize('hdf', HDF5_FILES or [])
@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_meta(hdf):
    meta = load_hdf5_meta(hdf)
    assertions_on_metadata(meta, is_band_specific=False)
    assert 'GranuleBeginningDateTime' in meta['meta']
    for band_meta in meta['band_meta']:
        assert 'GranuleBeginningDateTime' in band_meta


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_load_array():
    f = HDF5_FILES[0]
    data_file = gdal.Open(f)
    sd = data_file.GetSubDatasets()[0][0]
    data = load_array(sd)
    assert data


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
@pytest.mark.parametrize('hdf', HDF5_FILES or [])
def test_read_array(hdf):
    meta = load_hdf5_meta(hdf)
    es = load_hdf5_array(hdf, meta, band_specs)
    for band in es.data_vars:
        sample = getattr(es, band)
        mean_y = np.mean(sample.y)
        mean_x = np.mean(sample.x)
        band_names = np.array([b[-1] for b in band_specs])
        assert sample.y.size == 1200
        assert sample.x.size == 1200
        assert len(es.data_vars) == len(band_specs)
        assert np.all(es.band_order == [x[-1] for x in band_specs])
        assertions_on_band_metadata(sample.attrs)
    es2 = load_hdf5_array(hdf, meta, band_specs=None)
    assert len(es2.data_vars) > len(es.data_vars)

