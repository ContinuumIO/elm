'''
elm.pipeline.steps.feature_extraction

Wraps sklearn.feature_extraction for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.feature_extraction import DictVectorizer as _DictVectorizer
from sklearn.feature_extraction import FeatureHasher as _FeatureHasher



class DictVectorizer(SklearnMixin, _DictVectorizer):
    _cls = _DictVectorizer
    __init__ = _DictVectorizer.__init__



class FeatureHasher(SklearnMixin, _FeatureHasher):
    _cls = _FeatureHasher
    __init__ = _FeatureHasher.__init__

