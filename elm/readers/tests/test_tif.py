import glob
import os
import sys

import numpy as np
import pytest

from elm.readers.tif import (load_dir_of_tifs_meta,
                             load_dir_of_tifs_array,
                             load_tif_meta,
                             ls_tif_files)
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH,
                                    TIF_FILES,
                                    assertions_on_metadata,
                                    assertions_on_band_metadata)

TIF_DIR = os.path.dirname(TIF_FILES[0])
band_specs = [
    ['name', '_B1.TIF', 'band_1'],
    ['name', '_B2.TIF', 'band_2'],
    ['name', '_B3.TIF', 'band_3'],
    ['name', '_B4.TIF', 'band_4'],
    ['name', '_B5.TIF', 'band_5'],
    ['name', '_B6.TIF', 'band_6'],
    ['name', '_B7.TIF', 'band_7'],
    ['name', '_B9.TIF', 'band_9'],
    ['name', '_B10.TIF', 'band_10'],
    ['name', '_B11.TIF', 'band_11'],
]
@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_meta():
    for tif in TIF_FILES:
        raster, meta = load_tif_meta(tif)
        assert hasattr(raster, 'read')
        assert hasattr(raster, 'width')
        assertions_on_metadata(meta, is_band_specific=True)
        band_specs_with_band_8 = band_specs + [['name', '_B8.TIF', 'band_8']]
        meta = load_dir_of_tifs_meta(TIF_DIR, band_specs_with_band_8)
        for band_meta in meta['band_meta']:
            assertions_on_metadata(band_meta, is_band_specific=True)
        band_meta = meta['band_meta']
        heights_names = [(m['height'], m['name']) for m in band_meta]
        # band 8 is panchromatic with 15 m resolution
        # other bands have 30 m resolution.  They
        # have the same bounds, so band 8 has 4 times as many pixels
        heights_names.sort(key=lambda x:x[0])
        assert heights_names[-1][-1].endswith('_B8.TIF')


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_array():
    meta = load_dir_of_tifs_meta(TIF_DIR, band_specs)
    es = load_dir_of_tifs_array(TIF_DIR, meta, band_specs)
    for var in es.data_vars:
        sample = getattr(es, var)
        mean_y = np.mean(sample.y)
        mean_x = np.mean(sample.x)
        band_names = np.array([b[-1] for b in band_specs])
        assert sorted((mean_x,
                sample.bounds.left,
                sample.bounds.right))[1] == mean_x
        assert sorted((mean_y,
                sample.bounds.top,
                sample.bounds.bottom))[1] == mean_y
        assert np.all(band_names == es.band_order)
        assertions_on_band_metadata(sample.attrs)

