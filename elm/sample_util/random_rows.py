import copy
import numpy as np
from elm.sample_util.elm_store import ElmStore
from elm.pipeline.sample_pipeline import flatten_cube
from elm.sample_util.util import bands_as_columns

@bands_as_columns
def random_rows(es, n_rows):
    '''Drop '''
    attrs = copy.deepcopy(es.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(es.sample.values.shape[0])
    np.random.shuffle(inds)
    return ElmStore({'sample': es.sample[inds[:n_rows], :]}, attrs=attrs)
