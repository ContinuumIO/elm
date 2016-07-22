import numpy as np

def random_rows(es, n_rows):
    '''Drop '''
    sample = es.sample.where(es.sample.notnull()).dropna(dim='space')
    if n_rows > sample.space.size
        return es
    inds = np.arange(sample.values.shape[0])
    np.random.shuffle(inds)
    es['sample'] = sample.isel(space=inds[:n_rows])
    return es
