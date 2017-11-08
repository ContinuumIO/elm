'''
elm.pipeline.steps.isotonic

Wraps sklearn.isotonic for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.isotonic import IsotonicRegression as _IsotonicRegression



class IsotonicRegression(SklearnMixin, _IsotonicRegression):
    _cls = _IsotonicRegression
    __init__ = _IsotonicRegression.__init__

