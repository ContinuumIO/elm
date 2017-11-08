'''
elm.pipeline.steps.linear_model

Wraps sklearn.dummy for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.dummy import DummyClassifier as _DummyClassifier
from sklearn.dummy import DummyRegressor as _DummyRegressor



class DummyClassifier(SklearnMixin, _DummyClassifier):
    _cls = _DummyClassifier
    __init__ = _DummyClassifier.__init__



class DummyRegressor(SklearnMixin, _DummyRegressor):
    _cls = _DummyRegressor
    __init__ = _DummyRegressor.__init__

