'''
elm.pipeline.steps.linear_model

Wraps sklearn.cluster.bicluster for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster.bicluster
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.cluster.bicluster import BaseSpectral as _BaseSpectral



class BaseSpectral(SklearnMixin, _BaseSpectral):
    _cls = _BaseSpectral
    __init__ = _BaseSpectral.__init__

