import copy
import numpy as np
from elm.pipeline.sample_pipeline import flatten_data_arrays
from elm.sample_util.elm_store import data_arrays_as_columns, ElmStore

@flatten_data_arrays
def random_rows(es, n_rows):
    '''Drop '''
    attrs = copy.deepcopy(es.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(es.flat.values.shape[0])
    np.random.shuffle(inds)
    return ElmStore({'sample': es.flat[inds[:n_rows], :]}, attrs=attrs)
