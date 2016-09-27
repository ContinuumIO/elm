import os
import sys

import pytest
import numpy as np

from elm.readers.netcdf import load_netcdf_meta, load_netcdf_array
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    NETCDF_FILES,
                                    assertions_on_metadata)
from elm.readers.util import BandSpec
NETCDF_DIR = os.path.dirname(NETCDF_FILES[0])

variables_dict = dict(HQobservationTime='HQobservationTime')
variables_list = ['HQobservationTime']


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                    reason='elm-data repo has not been cloned')
def test_read_meta():
    for nc_file in NETCDF_FILES:
        meta = load_netcdf_meta(nc_file)
        assertions_on_metadata(meta)


def _validate_array_test_result(ds):
    sample = ds.HQobservationTime
    mean_y = np.mean(sample.y)
    mean_x = np.mean(sample.x)

    assert sorted((mean_x,
            ds.canvas.bounds.left,
            ds.canvas.bounds.right))[1] == mean_x
    assert sorted((mean_y,
            ds.canvas.bounds.top,
            ds.canvas.bounds.bottom))[1] == mean_y
    assert ds.y.size == 1800
    assert ds.x.size == 3600

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_read_using_dict_of_variables():
    for nc_file in NETCDF_FILES:
        meta = load_netcdf_meta(nc_file)
        ds = load_netcdf_array(nc_file, meta, variables_dict)
        _validate_array_test_result(ds)


@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                   reason='elm-data repo has not been cloned')
def test_read_using_list_of_variables():
    for nc_file in NETCDF_FILES:
        meta = load_netcdf_meta(nc_file)
        ds = load_netcdf_array(nc_file, meta, variables_list)
        _validate_array_test_result(ds)
        variables_list2 = [BandSpec('', '', v) for v in variables_list]
        ds = load_netcdf_array(nc_file, meta, variables_list2)
        _validate_array_test_result(ds)


