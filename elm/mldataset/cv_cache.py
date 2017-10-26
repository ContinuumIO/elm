from sklearn.model_selection import KFold
from dask_searchcv.methods import CVCache
from xarray_filters.pipeline import Step

class CVCacheSampleId(CVCache):
    def __init__(self, sampler, splits, pairwise=False, cache=True):
        self.sampler = sampler
        super(CVCacheSampleId, self).__init__(splits, pairwise=pairwise,
                                              cache=cache)

    def _post_splits(self, X, y=None, n=None, is_x=True, is_train=False):
        if y is not None:
            raise ValueError('Expected y to be None (returned by Sampler() instance or similar.')
        return self.sampler.fit_transform(X)

