'''
elm.pipeline.steps.linear_model

Wraps sklearn.random_projection for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.random_projection import BaseRandomProjection as _BaseRandomProjection
from sklearn.random_projection import GaussianRandomProjection as _GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection as _SparseRandomProjection



class BaseRandomProjection(SklearnMixin, _BaseRandomProjection):
    _cls = _BaseRandomProjection
    __init__ = _BaseRandomProjection.__init__



class GaussianRandomProjection(SklearnMixin, _GaussianRandomProjection):
    _cls = _GaussianRandomProjection
    __init__ = _GaussianRandomProjection.__init__



class SparseRandomProjection(SklearnMixin, _SparseRandomProjection):
    _cls = _SparseRandomProjection
    __init__ = _SparseRandomProjection.__init__

