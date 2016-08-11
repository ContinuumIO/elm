import os
import sys

import pytest
import numpy as np

from elm.readers.netcdf import load_netcdf_meta, load_netcdf_array
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    NETCDF_FILES,
                                    assertions_on_metadata)

NETCDF_DIR = os.path.dirname(NETCDF_FILES[0])

variables_dict = dict(HQobservationTime='HQobservationTime')
variables_list = ['HQobservationTime']

print(NETCDF_FILES, file=sys.stderr)


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                    reason='elm-data repo has not been cloned')
def test_read_meta():
    for nc_file in NETCDF_FILES:
        meta = load_netcdf_meta(nc_file)
        assertions_on_metadata(meta, is_band_specific=True)


def _validate_array_test_result(ds):
    sample = ds.HQobservationTime
    mean_y = np.mean(sample.y)
    mean_x = np.mean(sample.x)

    assert sorted((mean_x,
            ds.Bounds.left,
            ds.Bounds.right))[1] == mean_x
    assert sorted((mean_y,
            ds.Bounds.top,
            ds.Bounds.bottom))[1] == mean_y
    assert ds.y.size == 1800
    assert ds.x.size == 3600

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_read_using_dict_of_variables():
    for nc in NETCDF_FILES:
        meta = load_netcdf_meta(nc)
        ds = load_netcdf_array(nc, meta, variables_dict)
        _validate_array_test_result(ds)

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_read_using_list_of_variables():
    for nc in NETCDF_FILES:
        meta = load_netcdf_meta(nc)
        ds = load_netcdf_array(nc, meta, variables_list)
        _validate_array_test_result(ds)

