'''
elm.pipeline.steps.feature_selection

Wraps sklearn.feature_selection for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.feature_selection import GenericUnivariateSelect as _GenericUnivariateSelect
from sklearn.feature_selection import RFE as _RFE
from sklearn.feature_selection import RFECV as _RFECV
from sklearn.feature_selection import SelectFdr as _SelectFdr
from sklearn.feature_selection import SelectFpr as _SelectFpr
from sklearn.feature_selection import SelectFromModel as _SelectFromModel
from sklearn.feature_selection import SelectFwe as _SelectFwe
from sklearn.feature_selection import SelectKBest as _SelectKBest
from sklearn.feature_selection import SelectPercentile as _SelectPercentile
from sklearn.feature_selection import VarianceThreshold as _VarianceThreshold



class GenericUnivariateSelect(SklearnMixin, _GenericUnivariateSelect):
    _cls = _GenericUnivariateSelect
    __init__ = _GenericUnivariateSelect.__init__



class RFE(SklearnMixin, _RFE):
    _cls = _RFE
    __init__ = _RFE.__init__



class RFECV(SklearnMixin, _RFECV):
    _cls = _RFECV
    __init__ = _RFECV.__init__



class SelectFdr(SklearnMixin, _SelectFdr):
    _cls = _SelectFdr
    __init__ = _SelectFdr.__init__



class SelectFpr(SklearnMixin, _SelectFpr):
    _cls = _SelectFpr
    __init__ = _SelectFpr.__init__



class SelectFromModel(SklearnMixin, _SelectFromModel):
    _cls = _SelectFromModel
    __init__ = _SelectFromModel.__init__



class SelectFwe(SklearnMixin, _SelectFwe):
    _cls = _SelectFwe
    __init__ = _SelectFwe.__init__



class SelectKBest(SklearnMixin, _SelectKBest):
    _cls = _SelectKBest
    __init__ = _SelectKBest.__init__



class SelectPercentile(SklearnMixin, _SelectPercentile):
    _cls = _SelectPercentile
    __init__ = _SelectPercentile.__init__



class VarianceThreshold(SklearnMixin, _VarianceThreshold):
    _cls = _VarianceThreshold
    __init__ = _VarianceThreshold.__init__

