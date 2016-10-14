import copy

import numpy as np

from elm.readers import ElmStore

def random_rows(X, y=None, sample_weight=None, **kwargs):
    '''Drop '''
    attrs = copy.deepcopy(X.attrs)
    attrs['random_rows'] = n_rows
    inds = np.arange(X.flat.values.shape[0])
    np.random.shuffle(inds)
    X = ElmStore({'flat': X.flat[inds[:n_rows], :]}, attrs=attrs)
    if y is not None:
        y = y[inds[:n_rows]]
    if sample_weight is not None:
        sample_weight = sample_weight[inds[:n_rows]]
    return X, y, sample_weight