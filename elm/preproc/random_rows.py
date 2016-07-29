import copy
import numpy as np
from elm.preproc.elm_store import ElmStore

def random_rows(es, n_rows):
    '''Drop '''
    attrs = copy.deepcopy(es.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(es.sample.values.shape[0])
    np.random.shuffle(inds)
    return ElmStore({'sample': es.sample[inds[:n_rows], :]}, attrs=attrs)
