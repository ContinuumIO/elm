'''
elm.pipeline.steps.linear_model

Wraps sklearn.pipeline for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.pipeline import FeatureUnion as _FeatureUnion



class FeatureUnion(SklearnMixin, _FeatureUnion):
    _cls = _FeatureUnion
    __init__ = _FeatureUnion.__init__

