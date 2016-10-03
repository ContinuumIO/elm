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
    sample_y = kwargs.get('sample_y')
    sample_weight = kwargs.get('sample_weight')
    if sample_y is not None:
        sample_y = sample_y[inds[:n_rows]]
    if sample_weight is not None:
        sample_weight = sample_weight[inds[:n_rows]]
    return es, sample_y, sample_weight