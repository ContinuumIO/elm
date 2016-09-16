import glob
import os

import attr
import numpy as np
import pytest

from elm.readers.hdf4 import (load_hdf4_meta,
                              load_hdf4_array)

from elm.readers.util import BandSpec

from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH,
                                    HDF4_FILES,
                                    assertions_on_metadata,
                                    assertions_on_band_metadata)

HDF4_DIR = os.path.dirname(HDF4_FILES[0])

band_specs = [
    BandSpec('long_name', 'Band 1 ', 'band_1'),
    BandSpec('long_name', 'Band 2 ', 'band_2'),
    BandSpec('long_name', 'Band 3 ', 'band_3'),
    BandSpec('long_name', 'Band 4 ', 'band_4'),
    BandSpec('long_name', 'Band 5 ', 'band_5'),
    BandSpec('long_name', 'Band 7 ', 'band_7'),
    BandSpec('long_name', 'Band 8 ', 'band_8'),
    BandSpec('long_name', 'Band 10 ', 'band_10'),
    BandSpec('long_name', 'Band 11 ', 'band_11'),
]

@pytest.mark.parametrize('hdf', HDF4_FILES or [])
@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_meta(hdf):
    meta = load_hdf4_meta(hdf)
    assertions_on_metadata(meta)
    assert 'GranuleBeginningDateTime' in meta['meta']
    for band_meta in meta['band_meta']:
        assert 'GranuleBeginningDateTime' in band_meta


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
@pytest.mark.parametrize('hdf', HDF4_FILES or [])
def test_read_array(hdf):

    meta = load_hdf4_meta(hdf)
    es = load_hdf4_array(hdf, meta, band_specs)
    for band in es.data_vars:
        sample = getattr(es, band)
        mean_y = np.mean(sample.y)
        mean_x = np.mean(sample.x)
        band_names = np.array([b.name for b in band_specs])
        assert sorted((mean_x,
                sample.canvas.bounds.left,
                sample.canvas.bounds.right))[1] == mean_x
        assert sorted((mean_y,
                sample.canvas.bounds.top,
                sample.canvas.bounds.bottom))[1] == mean_y
        assert sample.y.size == 1200
        assert sample.x.size == 1200
        assert len(es.data_vars) == len(band_specs)
        assert np.all(es.band_order == [x.name for x in band_specs])
        assertions_on_band_metadata(sample.attrs)
    es2 = load_hdf4_array(hdf, meta, band_specs=None)
    assert len(es2.data_vars) > len(es.data_vars)

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_reader_kwargs():
    band_specs_kwargs = []
    for b in band_specs:
        b = attr.asdict(b)
        b['xsize'], b['ysize'] = 200, 300
        band_specs_kwargs.append(BandSpec(**b))
    meta = load_hdf4_meta(HDF4_FILES[0])
    es = load_hdf4_array(HDF4_FILES[0], meta, band_specs_kwargs)
    for b in es.band_order:
        assert getattr(es, b).values.shape == (300, 200)

