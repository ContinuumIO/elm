'''
elm.pipeline.steps.linear_model

Wraps sklearn.neighbors for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.neighbors import BallTree as _BallTree
from sklearn.neighbors import DistanceMetric as _DistanceMetric
from sklearn.neighbors import KDTree as _KDTree
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor as _KNeighborsRegressor
from sklearn.neighbors import KernelDensity as _KernelDensity
from sklearn.neighbors import LSHForest as _LSHForest
from sklearn.neighbors import LocalOutlierFactor as _LocalOutlierFactor
from sklearn.neighbors import NearestCentroid as _NearestCentroid
from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
from sklearn.neighbors import RadiusNeighborsClassifier as _RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor as _RadiusNeighborsRegressor



class BallTree(SklearnMixin, _BallTree):
    _cls = _BallTree
    __init__ = _BallTree.__init__



class DistanceMetric(SklearnMixin, _DistanceMetric):
    _cls = _DistanceMetric
    __init__ = _DistanceMetric.__init__



class KDTree(SklearnMixin, _KDTree):
    _cls = _KDTree
    __init__ = _KDTree.__init__



class KNeighborsClassifier(SklearnMixin, _KNeighborsClassifier):
    _cls = _KNeighborsClassifier
    __init__ = _KNeighborsClassifier.__init__



class KNeighborsRegressor(SklearnMixin, _KNeighborsRegressor):
    _cls = _KNeighborsRegressor
    __init__ = _KNeighborsRegressor.__init__



class KernelDensity(SklearnMixin, _KernelDensity):
    _cls = _KernelDensity
    __init__ = _KernelDensity.__init__



class LSHForest(SklearnMixin, _LSHForest):
    _cls = _LSHForest
    __init__ = _LSHForest.__init__



class LocalOutlierFactor(SklearnMixin, _LocalOutlierFactor):
    _cls = _LocalOutlierFactor
    __init__ = _LocalOutlierFactor.__init__



class NearestCentroid(SklearnMixin, _NearestCentroid):
    _cls = _NearestCentroid
    __init__ = _NearestCentroid.__init__



class NearestNeighbors(SklearnMixin, _NearestNeighbors):
    _cls = _NearestNeighbors
    __init__ = _NearestNeighbors.__init__



class RadiusNeighborsClassifier(SklearnMixin, _RadiusNeighborsClassifier):
    _cls = _RadiusNeighborsClassifier
    __init__ = _RadiusNeighborsClassifier.__init__



class RadiusNeighborsRegressor(SklearnMixin, _RadiusNeighborsRegressor):
    _cls = _RadiusNeighborsRegressor
    __init__ = _RadiusNeighborsRegressor.__init__

