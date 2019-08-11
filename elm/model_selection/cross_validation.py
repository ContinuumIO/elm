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
        inds = self.splits[n][0] if is_train else self.splits[n][1]

        result = self._call_sampler(np.array(X)[inds])
        return result


    def extract(self, X, y, n, is_x=True, is_train=True):
        if not is_x:
            return None
        return self._extract(X, y, n, is_x=is_x, is_train=is_train)


def cv_split_sampler(sampler, cv, X, y, groups, is_pairwise, cache):
    return CVCacheSampler(sampler=sampler,
                          splits=list(cv.split(X, y, groups)),
                          pairwise=is_pairwise,
                          cache=cache)
