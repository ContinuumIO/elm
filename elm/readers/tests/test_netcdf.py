import os
import sys

import pytest

from elm.readers.netcdf import load_netcdf_meta
from elm.readers.tests.util import (ELM_HAS_EXAMPLES,
                                    NETCDF_FILES,
                                    assertions_on_metadata)

NETCDF_DIR = os.path.dirname(NETCDF_FILES[0])
print(NETCDF_FILES, file=sys.stderr)

@pytest.mark.skipif(not ELM_HAS_EXAMPLES,
                    reason='elm-data repo has not been cloned')
def test_read_meta():
    for nc_file in NETCDF_FILES:
        meta = load_netcdf_meta(nc_file)
        assertions_on_metadata(meta, is_band_specific=True)
