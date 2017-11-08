'''
elm.pipeline.steps.linear_model

Wraps sklearn.cluster for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.cluster import AffinityPropagation as _AffinityPropagation
from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
from sklearn.cluster import Birch as _Birch
from sklearn.cluster import DBSCAN as _DBSCAN
from sklearn.cluster import FeatureAgglomeration as _FeatureAgglomeration
from sklearn.cluster import KMeans as _KMeans
from sklearn.cluster import MeanShift as _MeanShift
from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans
from sklearn.cluster import SpectralBiclustering as _SpectralBiclustering
from sklearn.cluster import SpectralClustering as _SpectralClustering
from sklearn.cluster import SpectralCoclustering as _SpectralCoclustering



class AffinityPropagation(SklearnMixin, _AffinityPropagation):
    _cls = _AffinityPropagation
    __init__ = _AffinityPropagation.__init__



class AgglomerativeClustering(SklearnMixin, _AgglomerativeClustering):
    _cls = _AgglomerativeClustering
    __init__ = _AgglomerativeClustering.__init__



class Birch(SklearnMixin, _Birch):
    _cls = _Birch
    __init__ = _Birch.__init__



class DBSCAN(SklearnMixin, _DBSCAN):
    _cls = _DBSCAN
    __init__ = _DBSCAN.__init__



class FeatureAgglomeration(SklearnMixin, _FeatureAgglomeration):
    _cls = _FeatureAgglomeration
    __init__ = _FeatureAgglomeration.__init__



class KMeans(SklearnMixin, _KMeans):
    _cls = _KMeans
    __init__ = _KMeans.__init__



class MeanShift(SklearnMixin, _MeanShift):
    _cls = _MeanShift
    __init__ = _MeanShift.__init__



class MiniBatchKMeans(SklearnMixin, _MiniBatchKMeans):
    _cls = _MiniBatchKMeans
    __init__ = _MiniBatchKMeans.__init__



class SpectralBiclustering(SklearnMixin, _SpectralBiclustering):
    _cls = _SpectralBiclustering
    __init__ = _SpectralBiclustering.__init__



class SpectralClustering(SklearnMixin, _SpectralClustering):
    _cls = _SpectralClustering
    __init__ = _SpectralClustering.__init__



class SpectralCoclustering(SklearnMixin, _SpectralCoclustering):
    _cls = _SpectralCoclustering
    __init__ = _SpectralCoclustering.__init__

