from sklearn.model_selection import KFold
from dask_searchcv.methods import CVCache
from xarray_filters.pipeline import Step

from sklearn.model_selection import GroupKFold as _GroupKFold
from sklearn.model_selection import GroupShuffleSplit as _GroupShuffleSplit
from sklearn.model_selection import KFold as _KFold
from sklearn.model_selection import LeaveOneGroupOut as _LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut as _LeavePGroupsOut
from sklearn.model_selection import LeaveOneOut as _LeaveOneOut
from sklearn.model_selection import LeavePOut as _LeavePOut
from sklearn.model_selection import PredefinedSplƒit as _PredefinedSplƒit
from sklearn.model_selection import RepeatedKFold as _RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold as _RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit as _ShuffleSplit
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit as _StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit




class CVCacheSampleId(CVCache):
    def __init__(self, sampler, splits, pairwise=False, cache=True):
        self.sampler = sampler
        super(CVCacheSampleId, self).__init__(splits, pairwise=pairwise,
                                              cache=cache)

    def _post_splits(self, X, y=None, n=None, is_x=True, is_train=False):
        if y is not None:
            raise ValueError('Expected y to be None (returned by Sampler() instance or similar.')
        return self.sampler.fit_transform(X)


def make_dec(cls):
    def split_wrap(func):
        def new_func(self, *a, **kw):
            for test, train in super(cls, self).split(*a, **kw):
                for a, b in zip(test, train):
                    yield a, b
        return new_func
    return split_wrap

class RepeatedKFold:

