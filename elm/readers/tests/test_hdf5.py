import glob
import gdal
import os

import attr
import numpy as np
import pytest

from elm.readers.hdf5 import (load_hdf5_meta,
                              load_subdataset,
                              load_hdf5_array)

from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    ELM_EXAMPLE_DATA_PATH,
                                    HDF5_FILES,
                                    assertions_on_metadata,
                                    assertions_on_band_metadata)

from elm.readers.util import BandSpec

HDF5_DIR = os.path.dirname(HDF5_FILES[0])

def get_band_specs(filename):
    if os.path.basename(filename).startswith('3B-MO'):
        sub_dataset_names = ('/precipitation',)
    else:
        sub_dataset_names = ('HQobservationTime',
                             'HQprecipSource',
                             'HQprecipitation',
                             'IRkalmanFilterWeight',
                             'IRprecipitation',
                             'precipitationCal',
                             'precipitationUncal',
                             'probabilityLiquidPrecipitation',)
    band_specs = []
    for sub in sub_dataset_names:
        band_specs.append(BandSpec(search_key='sub_dataset_name',
                               search_value=sub + '$', # line ender regex
                               name=sub))
    return sub_dataset_names, band_specs


@pytest.mark.parametrize('hdf', HDF5_FILES or [])
@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_read_meta(hdf):
    meta = load_hdf5_meta(hdf)
    assertions_on_metadata(meta)


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_load_subdataset():
    f = HDF5_FILES[0]
    data_file = gdal.Open(f)
    data_array = load_subdataset(data_file.GetSubDatasets()[0][0])
    assert 'canvas' in data_array.attrs.keys()
    assert data_array.data is not None


@pytest.mark.skipif(not ELM_HAS_EXAMPLES, reason='elm-data repo has not been cloned')
@pytest.mark.parametrize('filename', HDF5_FILES)
def test_read_array(filename):
    sub_dataset_names, band_specs = get_band_specs(filename)
    meta = load_hdf5_meta(filename)
    es = load_hdf5_array(filename, meta, band_specs)
    assert len(es.data_vars) == len(sub_dataset_names)
    for band in es.data_vars:
        sample = getattr(es, band)
        assert sample.y.size == 1800
        assert sample.x.size == 3600
        assert len(es.data_vars) == len(band_specs)
        assertions_on_band_metadata(sample.attrs)


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
               reason='elm-data repo has not been cloned')
def test_reader_kwargs():
    pytest.xfail('Transpose issue in hdf5 files')
    sub_dataset_names, band_specs = get_band_specs(HDF5_FILES[0])
    band_specs_kwargs = []
    for b in band_specs:
        b = attr.asdict(b)
        b['buf_xsize'], b['buf_ysize'] = 200, 300
        band_specs_kwargs.append(BandSpec(**b))
    meta = load_hdf5_meta(HDF5_FILES[0])
    es = load_hdf5_array(HDF5_FILES[0], meta, band_specs_kwargs)
    for b in es.band_order:
        assert getattr(es, b).values.shape == (300, 200)

