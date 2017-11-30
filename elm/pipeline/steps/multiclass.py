'''
elm.pipeline.steps.multiclass

Wraps sklearn.multiclass for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.multiclass import OneVsOneClassifier as _OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier as _OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier as _OutputCodeClassifier



class OneVsOneClassifier(SklearnMixin, _OneVsOneClassifier):
    _cls = _OneVsOneClassifier
    __init__ = _OneVsOneClassifier.__init__



class OneVsRestClassifier(SklearnMixin, _OneVsRestClassifier):
    _cls = _OneVsRestClassifier
    __init__ = _OneVsRestClassifier.__init__



class OutputCodeClassifier(SklearnMixin, _OutputCodeClassifier):
    _cls = _OutputCodeClassifier
    __init__ = _OutputCodeClassifier.__init__

