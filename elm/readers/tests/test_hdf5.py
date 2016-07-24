import os

import pytest
import numpy as np

from elm.readers import hdf5_tools

test_file = os.path.join(os.path.dirname(__file__), "test1.hdf5")

def test_hdf5_dataset_loads():
    """Load a few datasets from our test hdf5 file and check that our results are correct
    Also try loading datasets that don't exist and check that we get the expected error.
    """
    d0 = hdf5_tools.load_hdf5_dataset(test_file, "dataset0")
    assert np.array_equal(d0, np.arange(10).reshape(2,5))

    #load a dataset from within a group
    d1 = hdf5_tools.load_hdf5_dataset(test_file, "group1/dataset1")
    assert np.array_equal(d1, np.zeros(10))

    d2 = hdf5_tools.load_hdf5_dataset(test_file, "group2/nested2/dataset2")
    assert np.array_equal(d2, np.ones((10,2)))
    
def test_hdf5_info():
    """Check that hdf5_info() returns all of the groups, datasets, and metadata about the test file
    """
    info = hdf5_tools.hdf5_info(test_file)

    assert info == {'/': {'CLASS': b'GROUP',
                                 'PYTABLES_FORMAT_VERSION': b'2.1',
                                 'TITLE': b'',
                                 'VERSION': b'1.0'},
                     'dataset0': {},
                     'group1': {'CLASS': b'GROUP', 'TITLE': b'', 'VERSION': b'1.0', 'attr1': b'attr1 value'}, #This group contains a user-set attribute "attr1"
                     'group1/dataset1': {},
                     'group1/nested1': {'CLASS': b'GROUP', 'TITLE': b'', 'VERSION': b'1.0'},
                     'group2': {'CLASS': b'GROUP', 'TITLE': b'', 'VERSION': b'1.0'},
                     'group2/nested2': {'CLASS': b'GROUP', 'TITLE': b'', 'VERSION': b'1.0'},
                     'group2/nested2/dataset2': {},
                     'group2/nested2/nestednested2': {'CLASS': b'GROUP', 'TITLE': b'', 'VERSION': b'1.0'}}
    

def test_hdf5_attrs():
    """Check that hdf5_attrs() gives the user attributes of the test file"""
    attrs = hdf5_tools.hdf5_attrs(test_file)
    assert attrs == {'CLASS': b'GROUP', 'PYTABLES_FORMAT_VERSION': b'2.1', 'TITLE': b'', 'VERSION': b'1.0'}
    
