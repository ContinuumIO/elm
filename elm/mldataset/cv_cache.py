from sklearn.model_selection import KFold
from dask_searchcv.methods import CVCache
from xarray_filters.pipeline import Step

class CVCacheSampleId(CVCache):
    def __init__(self, sampler, splits, pairwise=False, cache=True):
        self.sampler = sampler
        super(CVCacheSampleId, self).__init__(splits, pairwise=pairwise,
                                              cache=cache)
        print('cvcache', vars(self))

    def _post_splits(self, X, y, n, is_x=True, is_train=False):
        print('sampler called on ', X, y, is_x, is_train)
        if y is not None:
            raise ValueError('Expected y to be None (returned by Sampler() instance or similar.')
        print('sampler called on ', X)
        return self.sampler.fit_transform(X)

