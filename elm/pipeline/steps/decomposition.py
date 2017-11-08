'''
elm.pipeline.steps.decomposition

Wraps sklearn.decomposition for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.decomposition import DictionaryLearning as _DictionaryLearning
from sklearn.decomposition import FactorAnalysis as _FactorAnalysis
from sklearn.decomposition import FastICA as _FastICA
from sklearn.decomposition import IncrementalPCA as _IncrementalPCA
from sklearn.decomposition import KernelPCA as _KernelPCA
from sklearn.decomposition import LatentDirichletAllocation as _LatentDirichletAllocation
from sklearn.decomposition import MiniBatchDictionaryLearning as _MiniBatchDictionaryLearning
from sklearn.decomposition import MiniBatchSparsePCA as _MiniBatchSparsePCA
from sklearn.decomposition import NMF as _NMF
from sklearn.decomposition import PCA as _PCA
from sklearn.decomposition import SparseCoder as _SparseCoder
from sklearn.decomposition import SparsePCA as _SparsePCA
from sklearn.decomposition import TruncatedSVD as _TruncatedSVD



class DictionaryLearning(SklearnMixin, _DictionaryLearning):
    _cls = _DictionaryLearning
    __init__ = _DictionaryLearning.__init__



class FactorAnalysis(SklearnMixin, _FactorAnalysis):
    _cls = _FactorAnalysis
    __init__ = _FactorAnalysis.__init__



class FastICA(SklearnMixin, _FastICA):
    _cls = _FastICA
    __init__ = _FastICA.__init__



class IncrementalPCA(SklearnMixin, _IncrementalPCA):
    _cls = _IncrementalPCA
    __init__ = _IncrementalPCA.__init__



class KernelPCA(SklearnMixin, _KernelPCA):
    _cls = _KernelPCA
    __init__ = _KernelPCA.__init__



class LatentDirichletAllocation(SklearnMixin, _LatentDirichletAllocation):
    _cls = _LatentDirichletAllocation
    __init__ = _LatentDirichletAllocation.__init__



class MiniBatchDictionaryLearning(SklearnMixin, _MiniBatchDictionaryLearning):
    _cls = _MiniBatchDictionaryLearning
    __init__ = _MiniBatchDictionaryLearning.__init__



class MiniBatchSparsePCA(SklearnMixin, _MiniBatchSparsePCA):
    _cls = _MiniBatchSparsePCA
    __init__ = _MiniBatchSparsePCA.__init__



class NMF(SklearnMixin, _NMF):
    _cls = _NMF
    __init__ = _NMF.__init__



class PCA(SklearnMixin, _PCA):
    _cls = _PCA
    __init__ = _PCA.__init__



class SparseCoder(SklearnMixin, _SparseCoder):
    _cls = _SparseCoder
    __init__ = _SparseCoder.__init__



class SparsePCA(SklearnMixin, _SparsePCA):
    _cls = _SparsePCA
    __init__ = _SparsePCA.__init__



class TruncatedSVD(SklearnMixin, _TruncatedSVD):
    _cls = _TruncatedSVD
    __init__ = _TruncatedSVD.__init__

