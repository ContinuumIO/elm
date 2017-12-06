'''
elm.pipeline.steps.multioutput

Wraps sklearn.multioutput for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.multioutput import ClassifierChain as _ClassifierChain
from sklearn.multioutput import MultiOutputClassifier as _MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor



class ClassifierChain(SklearnMixin, _ClassifierChain):
    _cls = _ClassifierChain
    __init__ = _ClassifierChain.__init__



class MultiOutputClassifier(SklearnMixin, _MultiOutputClassifier):
    _cls = _MultiOutputClassifier
    __init__ = _MultiOutputClassifier.__init__



class MultiOutputRegressor(SklearnMixin, _MultiOutputRegressor):
    _cls = _MultiOutputRegressor
    __init__ = _MultiOutputRegressor.__init__

