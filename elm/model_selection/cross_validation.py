from dask_searchcv.methods import CVCache
import numpy as np

class CVCacheSampler(CVCache):
    def __init__(self, sampler, splits=None, pairwise=None, cache=True):
        self.sampler = sampler
        assert cache is True
        CVCache.__init__(self, splits, pairwise=pairwise, cache=True)

    def _call_sampler(self, X, y=None, n=None, is_x=True, is_train=False):
        if self.splits is None:
            raise ValueError('Expected .splits to before _call_sampler')
        if y is not None:
            raise ValueError('y should be None (found {})'.format(type(y)))
        func = getattr(self.sampler, 'fit_transform', None)
        if func is None:
            func = getattr(self.sampler, 'transform', self.sampler)
        if not callable(func):
            raise ValueError('Expected "sampler" to be callable or have fit_transform/transform methods')
        out = func(X, y=y, is_x=is_x, is_train=is_train)
        return out

    def _extract(self, X, y, n, is_x=True, is_train=True):
        if self.cache is not None and (n, is_x, is_train) in self.cache:
            return self.cache[n, is_x, is_train]

        inds = self.splits[n][0] if is_train else self.splits[n][1]

        if self.cache in (None, False):
            raise ValueError('Must set cache_cv=True with _call_sampler')
        result = self._call_sampler(np.array(X)[inds])
        if isinstance(result, tuple) and len(result) == 2:
            (self.cache[n, True, is_train],
             self.cache[n, False, is_train]) = result
        else:
            self.cache[n, True, is_train] = result
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
        result = X[np.ix_(train if is_train else test, train)]
        result = self._call_sampler(result)
        if _is_xy_tuple(result):
            if self.cache is not None:
                (self.cache[n, True, is_train],
                 self.cache[n, False, is_train]) = result
        elif self.cache is not None:
            self.cache[n, True, is_train] = result
        return result

    def extract(self, X, y, n, is_x=True, is_train=True):
        if is_x:
            if self.pairwise:
                return self._extract_pairwise(X, y, n, is_train=is_train)
        return self._extract(X, y, n, is_x=is_x, is_train=is_train)


def cv_split_sampler(sampler, cv, X, y, groups, is_pairwise, cache):
    return CVCacheSampler(sampler=sampler,
                          splits=list(cv.split(X, y, groups)),
                          pairwise=is_pairwise,
                          cache=cache)
