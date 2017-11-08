'''
elm.pipeline.steps.linear_model

Wraps sklearn.covariance for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.covariance import EllipticEnvelope as _EllipticEnvelope
from sklearn.covariance import EmpiricalCovariance as _EmpiricalCovariance
from sklearn.covariance import GraphLasso as _GraphLasso
from sklearn.covariance import GraphLassoCV as _GraphLassoCV
from sklearn.covariance import LedoitWolf as _LedoitWolf
from sklearn.covariance import MinCovDet as _MinCovDet
from sklearn.covariance import OAS as _OAS
from sklearn.covariance import ShrunkCovariance as _ShrunkCovariance



class EllipticEnvelope(SklearnMixin, _EllipticEnvelope):
    _cls = _EllipticEnvelope
    __init__ = _EllipticEnvelope.__init__



class EmpiricalCovariance(SklearnMixin, _EmpiricalCovariance):
    _cls = _EmpiricalCovariance
    __init__ = _EmpiricalCovariance.__init__



class GraphLasso(SklearnMixin, _GraphLasso):
    _cls = _GraphLasso
    __init__ = _GraphLasso.__init__



class GraphLassoCV(SklearnMixin, _GraphLassoCV):
    _cls = _GraphLassoCV
    __init__ = _GraphLassoCV.__init__



class LedoitWolf(SklearnMixin, _LedoitWolf):
    _cls = _LedoitWolf
    __init__ = _LedoitWolf.__init__



class MinCovDet(SklearnMixin, _MinCovDet):
    _cls = _MinCovDet
    __init__ = _MinCovDet.__init__



class OAS(SklearnMixin, _OAS):
    _cls = _OAS
    __init__ = _OAS.__init__



class ShrunkCovariance(SklearnMixin, _ShrunkCovariance):
    _cls = _ShrunkCovariance
    __init__ = _ShrunkCovariance.__init__

