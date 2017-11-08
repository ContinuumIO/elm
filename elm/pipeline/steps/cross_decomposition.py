'''
elm.pipeline.steps.linear_model

Wraps sklearn.cross_decomposition for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.cross_decomposition import CCA as _CCA
from sklearn.cross_decomposition import PLSCanonical as _PLSCanonical
from sklearn.cross_decomposition import PLSRegression as _PLSRegression
from sklearn.cross_decomposition import PLSSVD as _PLSSVD



class CCA(SklearnMixin, _CCA):
    _cls = _CCA
    __init__ = _CCA.__init__



class PLSCanonical(SklearnMixin, _PLSCanonical):
    _cls = _PLSCanonical
    __init__ = _PLSCanonical.__init__



class PLSRegression(SklearnMixin, _PLSRegression):
    _cls = _PLSRegression
    __init__ = _PLSRegression.__init__



class PLSSVD(SklearnMixin, _PLSSVD):
    _cls = _PLSSVD
    __init__ = _PLSSVD.__init__

