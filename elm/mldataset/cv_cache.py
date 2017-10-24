from sklearn.model_selection import KFold
from dask_searchcv.methods import CVCache
from xarray_filters.pipeline import Step

class CVCacheSampleId(CVCache):
    def __init__(self, sampler, splits, pairwise=False, cache=True):
        self.sampler = sampler
        super(CVCacheSampleId, self).__init__(splits, pairwise=pairwise,
                                              cache=cache)

    def _post_splits(self, X, y, n, is_x=True, is_train=False):
        if y is not None:
            raise ValueError('Expected y to be None (returned by Sampler() instance or similar.')
        return self.sampler(X)



'''
class CVWrap(Generic):
    cv = None
    sampler = None

    def transform(self, *a, **kw):
         for test, train in self.cv.split(*a, **kw)
        return tuple((self.sampler(train), self.sampler(test)))



sample_args_list = tuple(zip(*np.meshgrid(np.linspace(0, 1, 100),
                                            np.linspace(0, 2, 50))))
cv = sk_KFold()
tuple(cv.split(sample_args_list))



TEST - TODO like the following
def sampler(filenames):
    print(filenames)
cv = CVCacheSampleId([['file_1', 'file_2'],
                      ['file_3', 'file_4']],
                     sampler=sampler)
cv.extract('ignore', 'ignore', 0)


def cv_split(cv, X, y, groups, is_pairwise, cache):
    return CVCache(list(cv.split(X, y, groups)), is_pairwise, cache)
list(cv.split(X, y, groups))
X_train = cv.extract(X, y, n, True, True)
y_train = cv.extract(X, y, n, False, True)
X_test = cv.extract(X, y, n, True, False)
y_test = cv.extract(X, y, n, False, False)
    def __reduce__(self):
        return (CVCache, (self.splits, self.pairwise, self.cache is not None))
    def num_test_samples(self):
        return np.array([i.sum() if i.dtype == bool else len(i)
                         for i in pluck(1, self.splits)])
    def extract(self, X, y, n, is_x=True, is_train=True):
        if is_x:
            if self.pairwise:
                return self._extract_pairwise(X, y, n, is_train=is_train)
            return self._extract(X, y, n, is_x=True, is_train=is_train)
        if y is None:
            return None
        return self._extract(X, y, n, is_x=False, is_train=is_train)
'''