import copy
import numpy as np
from elm.readers import ElmStore

def random_rows(es, n_rows, **kwargs):
    '''Drop '''
    attrs = copy.deepcopy(es.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(es.flat.values.shape[0])
    np.random.shuffle(inds)
    es = ElmStore({'flat': es.flat[inds[:n_rows], :]}, attrs=attrs)
    assert es.flat.canvas
    return es