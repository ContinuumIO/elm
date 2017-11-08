from dask_searchcv.methods import CVCache
from sklearn.base import BaseEstimator
import numpy as np

class CVCacheSampler(BaseEstimator, CVCache):
    def __init__(self, sampler, splits=None, pairwise=None, cache=None):
        self.sampler = sampler
        super(CVCacheSampler, self).__init__(splits, pairwise=pairwise,
                                              cache=cache)

    def _post_splits(self, X, y=None, n=None, is_x=True, is_train=False):
        if self.splits is None:
            raise ValueError('Expected .splits to before _post_splits')
        if y is not None:
            raise ValueError('y should be None (found {})'.format(type(y)))
        func = getattr(self.sampler, 'fit_transform', None)
        if func is None:
            func = getattr(self.sampler, 'transform', self.sampler)
        return func(X, y=y, is_x=is_x, is_train=is_train)

    def _extract(self, X, y, n, is_x=True, is_train=True):
        if self.cache is not None and (n, is_x, is_train) in self.cache:
            return self.cache[n, is_x, is_train]

        inds = self.splits[n][0] if is_train else self.splits[n][1]

        post_splits = getattr(self, '_post_splits', None)
        if post_splits:
            if self.cache in (None, False):
                raise ValueError('Must set cache_cv=True with _post_splits')
            result = post_splits(np.array(X)[inds])
            self.cache[n, True, is_train] = result
        else:
            result = safe_indexing(X if is_x else y, inds)
            self.cache[n, is_x, is_train] = result
        return result


    def _extract_pairwise(self, X, y, n, is_train=True):
        if self.cache is not None and (n, True, is_train) in self.cache:
            return self.cache[n, True, is_train]

        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                            "to be passed as arrays or sparse matrices.")
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        train, test = self.splits[n]
        post_splits = getattr(self, '_post_splits', None)
        result = X[np.ix_(train if is_train else test, train)]
        if post_splits:
            result = post_splits(result)
            if _is_xy_tuple(result):
                if self.cache is not None:
                    (self.cache[n, True, is_train],
                     self.cache[n, False, is_train]) = result
            elif self.cache is not None:
                self.cache[n, True, is_train] = result
        elif self.cache is not None:
                self.cache[n, True, is_train] = result
        return result
