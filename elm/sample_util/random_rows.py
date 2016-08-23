import copy
import numpy as np
from elm.sample_util.elm_store import data_arrays_as_columns, ElmStore

@data_arrays_as_columns
def random_rows(es, n_rows):
    '''Drop '''
    attrs = copy.deepcopy(es.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(es.flat.values.shape[0])
    np.random.shuffle(inds)
    return ElmStore({'flat': es.flat[inds[:n_rows], :]}, attrs=attrs)
