'''
elm.pipeline.steps.linear_model

Wraps sklearn.model_selection for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.model_selection import BaseCrossValidator as _BaseCrossValidator
from sklearn.model_selection import GroupKFold as _GroupKFold
from sklearn.model_selection import GroupShuffleSplit as _GroupShuffleSplit
from sklearn.model_selection import KFold as _KFold
from sklearn.model_selection import LeaveOneGroupOut as _LeaveOneGroupOut
from sklearn.model_selection import LeaveOneOut as _LeaveOneOut
from sklearn.model_selection import LeavePGroupsOut as _LeavePGroupsOut
from sklearn.model_selection import LeavePOut as _LeavePOut
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import RepeatedKFold as _RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold as _RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit as _ShuffleSplit
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit as _StratifiedShuffleSplit
from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit



class BaseCrossValidator(SklearnMixin, _BaseCrossValidator):
    _cls = _BaseCrossValidator
    __init__ = _BaseCrossValidator.__init__



class GroupKFold(SklearnMixin, _GroupKFold):
    _cls = _GroupKFold
    __init__ = _GroupKFold.__init__



class GroupShuffleSplit(SklearnMixin, _GroupShuffleSplit):
    _cls = _GroupShuffleSplit
    __init__ = _GroupShuffleSplit.__init__



class KFold(SklearnMixin, _KFold):
    _cls = _KFold
    __init__ = _KFold.__init__



class LeaveOneGroupOut(SklearnMixin, _LeaveOneGroupOut):
    _cls = _LeaveOneGroupOut
    __init__ = _LeaveOneGroupOut.__init__



class LeaveOneOut(SklearnMixin, _LeaveOneOut):
    _cls = _LeaveOneOut
    __init__ = _LeaveOneOut.__init__



class LeavePGroupsOut(SklearnMixin, _LeavePGroupsOut):
    _cls = _LeavePGroupsOut
    __init__ = _LeavePGroupsOut.__init__



class LeavePOut(SklearnMixin, _LeavePOut):
    _cls = _LeavePOut
    __init__ = _LeavePOut.__init__



class PredefinedSplit(SklearnMixin, _PredefinedSplit):
    _cls = _PredefinedSplit
    __init__ = _PredefinedSplit.__init__



class RepeatedKFold(SklearnMixin, _RepeatedKFold):
    _cls = _RepeatedKFold
    __init__ = _RepeatedKFold.__init__



class RepeatedStratifiedKFold(SklearnMixin, _RepeatedStratifiedKFold):
    _cls = _RepeatedStratifiedKFold
    __init__ = _RepeatedStratifiedKFold.__init__



class ShuffleSplit(SklearnMixin, _ShuffleSplit):
    _cls = _ShuffleSplit
    __init__ = _ShuffleSplit.__init__



class StratifiedKFold(SklearnMixin, _StratifiedKFold):
    _cls = _StratifiedKFold
    __init__ = _StratifiedKFold.__init__



class StratifiedShuffleSplit(SklearnMixin, _StratifiedShuffleSplit):
    _cls = _StratifiedShuffleSplit
    __init__ = _StratifiedShuffleSplit.__init__



class TimeSeriesSplit(SklearnMixin, _TimeSeriesSplit):
    _cls = _TimeSeriesSplit
    __init__ = _TimeSeriesSplit.__init__

