'''
elm.pipeline.steps.linear_model

Wraps sklearn.tree for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as _DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier as _ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor as _ExtraTreeRegressor



class DecisionTreeClassifier(SklearnMixin, _DecisionTreeClassifier):
    _cls = _DecisionTreeClassifier
    __init__ = _DecisionTreeClassifier.__init__



class DecisionTreeRegressor(SklearnMixin, _DecisionTreeRegressor):
    _cls = _DecisionTreeRegressor
    __init__ = _DecisionTreeRegressor.__init__



class ExtraTreeClassifier(SklearnMixin, _ExtraTreeClassifier):
    _cls = _ExtraTreeClassifier
    __init__ = _ExtraTreeClassifier.__init__



class ExtraTreeRegressor(SklearnMixin, _ExtraTreeRegressor):
    _cls = _ExtraTreeRegressor
    __init__ = _ExtraTreeRegressor.__init__

