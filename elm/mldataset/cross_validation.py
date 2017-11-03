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
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import RepeatedKFold as _RepeatedKFold
from sklearn.model_selection import ShuffleSplit as _ShuffleSplit
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit as _StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit
# TODO Add support for sklearn.model_selection.RepeatedStratifiedKFold

CV_CLASSES = [
    'GroupKFold',
    'GroupShuffleSplit',
    'KFold',
    'LeaveOneGroupOut',
    'LeavePGroupsOut',
    'LeaveOneOut',
    'LeavePOut',
    'PredefinedSplit',
    'RepeatedKFold',
    'ShuffleSplit',
    'StratifiedKFold',
    'StratifiedShuffleSplit',
    'TimeSeriesSplit',
]

__all__ = CV_CLASSES + ['MLDatasetMixin', 'CV_CLASSES']


class MLDatasetMixin:
    #def split(self, *args, **kw):
     #   for test, train in super().split(*args, **kw):
      #      for a, b in zip(test, train):
       #         yield a, b
    pass

class GroupKFold(MLDatasetMixin, _GroupKFold):
    pass


class GroupShuffleSplit(MLDatasetMixin, _GroupShuffleSplit):
    pass


class KFold(MLDatasetMixin, _KFold):
    pass


class LeaveOneGroupOut(MLDatasetMixin, _LeaveOneGroupOut):
    pass


class LeavePGroupsOut(MLDatasetMixin, _LeavePGroupsOut):
    pass


class LeaveOneOut(MLDatasetMixin, _LeaveOneOut):
    pass


class LeavePOut(MLDatasetMixin, _LeavePOut):
    pass


class PredefinedSplit(MLDatasetMixin, _PredefinedSplit):
    pass


class RepeatedKFold(MLDatasetMixin, _RepeatedKFold):
    pass


class ShuffleSplit(MLDatasetMixin, _ShuffleSplit):
    pass


class StratifiedKFold(MLDatasetMixin, _StratifiedKFold):
    pass


class StratifiedShuffleSplit(MLDatasetMixin, _StratifiedShuffleSplit):
    pass


class TimeSeriesSplit(MLDatasetMixin, _TimeSeriesSplit):
    pass


