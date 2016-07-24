import h5py

def load_hdf5_dataset(fpath, dataset):
    """Load a specific dataset from an hdf5 file
       This can be a simple name like "Grid",
         or a path-like name like "Grid/precipitation"
    """
    with h5py.File(fpath, 'r') as f:
        if dataset not in f:
            raise ValueError("dataset {} not found in {}. Available datasets: {}".format(dataset, f, f.keys()))
        return f[dataset][:]


def hdf5_attrs(fpath, attr=None):
    """Load and return the given attribute for
         the given file as a dict.
       If attr is None, return all attrs
    """
    with h5py.File(fpath, 'r') as f:
        if attr is None:
            return dict(f.attrs)
        elif attr in f.attrs:
            return {attr: f.attrs[attr]}
        else:
            raise ValueError("attr {} not found in {} attributes".format(attr))

def hdf5_info(fpath):
    """Return a dict of all groups and datasets with corresponding metadata
       The key will be path of the group or dataset and the value will be the metadata
    """
    info = {}
    with h5py.File(fpath, 'r') as f:
        info['/'] = dict(f.attrs)
        groups = []
        f.visit(groups.append)
        for path in groups:
            info[path] = dict(f[path].attrs)

    return info
