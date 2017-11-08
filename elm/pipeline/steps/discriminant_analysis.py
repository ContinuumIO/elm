'''
elm.pipeline.steps.linear_model

Wraps sklearn.discriminant_analysis for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _QuadraticDiscriminantAnalysis



class LinearDiscriminantAnalysis(SklearnMixin, _LinearDiscriminantAnalysis):
    _cls = _LinearDiscriminantAnalysis
    __init__ = _LinearDiscriminantAnalysis.__init__



class QuadraticDiscriminantAnalysis(SklearnMixin, _QuadraticDiscriminantAnalysis):
    _cls = _QuadraticDiscriminantAnalysis
    __init__ = _QuadraticDiscriminantAnalysis.__init__

