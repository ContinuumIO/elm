from dask_searchcv.methods import CVCache
class CVCacheSampleId(CVCache):
    def __init__(self, splits, sampler, pairwise=False, cache=True):

        self.splits = splits
        self.pairwise = pairwise
        self.cache = {} if cache else None
        self.sampler = sampler

    def extract_param(self, key, x, n):
        print('extract_param', key, x, n)
        if self.cache is not None and (n, key) in self.cache:
            return self.cache[n, key]
        out = self.sampler(self.splits[n][0])
        if self.cache is not None:
            self.cache[n, key] = out
        return out

    def extract(self, X, y, n, is_x=True, is_train=True):
        print('extract', X, y, n, is_x, is_train)
        if self.cache is not None and (n, is_x, is_train) in self.cache:
            return self.cache[n, is_x, is_train]

        inds = self.splits[n][0] if is_train else self.splits[n][1]
        result = self.sampler(inds)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            X, y = result
        else:
            X = result
            y = None
        if self.cache is not None:
            self.cache[n, True, is_train] = X
            self.cache[n, False, is_train] = y
        return X if is_x else y

    def split(self, X, y, groups):
        return self.splits
        print('groups', groups, X, y)
        if y is None:
            y = (None,) * len(X)
        #for train, test in X:
            #yield train, test



def cv_split(sampler, cv, X, y, groups, is_pairwise, cache):
    print('samp', sampler, cv, X, y, groups, is_pairwise, cache)
    return CVCacheSampleId(list(cv.split(X, y, groups)),
                           sampler, is_pairwise, cache)

'''

TEST - TODO like the following
def sampler(filenames):
    print(filenames)

cv = CVCacheSampleId([['file_1', 'file_2'],
                      ['file_3', 'file_4']],
                     sampler=sampler)
cv.extract('ignore', 'ignore', 0)


'''
'''
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