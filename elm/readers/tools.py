
from .hdf4_tools import (load_hdf4)
from .hdf5_tools import (load_hdf5_dataset, hdf5_attrs, hdf5_info)
from .netcdf_tools import (load_netcdf_variable, netcdf_attrs, netcdf_variables)

import magic

def load_data(fpath, dataset=None):
    file_type = magic.from_file(fpath)
    print file_type
    if file_type == "HDF5":
        return load_hdf5_dataset(fpath, dataset)
    elif file_type == "HDF4":
        return load_hdf4(fpath, dataset)
    elif file_type.startswith("NETCDF"):
        return load_netcdf_variable(fpath, dataset)

def file_attrs(fpath):
    file_type = magic.from_file(fpath)
    print file_type
    if file_type == "HDF5":
        return hdf5_attrs(fpath)
    elif file_type.startswith("NETCDF"):
        return netcdf_variables(fpath, dataset)
    
