'''
elm.pipeline.steps.linear_model

Wraps sklearn.manifold for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.manifold import Isomap as _Isomap
from sklearn.manifold import LocallyLinearEmbedding as _LocallyLinearEmbedding
from sklearn.manifold import MDS as _MDS
from sklearn.manifold import SpectralEmbedding as _SpectralEmbedding
from sklearn.manifold import TSNE as _TSNE



class Isomap(SklearnMixin, _Isomap):
    _cls = _Isomap
    __init__ = _Isomap.__init__



class LocallyLinearEmbedding(SklearnMixin, _LocallyLinearEmbedding):
    _cls = _LocallyLinearEmbedding
    __init__ = _LocallyLinearEmbedding.__init__



class MDS(SklearnMixin, _MDS):
    _cls = _MDS
    __init__ = _MDS.__init__



class SpectralEmbedding(SklearnMixin, _SpectralEmbedding):
    _cls = _SpectralEmbedding
    __init__ = _SpectralEmbedding.__init__



class TSNE(SklearnMixin, _TSNE):
    _cls = _TSNE
    __init__ = _TSNE.__init__

