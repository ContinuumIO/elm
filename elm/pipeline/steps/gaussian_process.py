'''
elm.pipeline.steps.gaussian_process

Wraps sklearn.gaussian_process for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.gaussian_process import GaussianProcess as _GaussianProcess
from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor as _GaussianProcessRegressor



class GaussianProcess(SklearnMixin, _GaussianProcess):
    _cls = _GaussianProcess
    __init__ = _GaussianProcess.__init__



class GaussianProcessClassifier(SklearnMixin, _GaussianProcessClassifier):
    _cls = _GaussianProcessClassifier
    __init__ = _GaussianProcessClassifier.__init__



class GaussianProcessRegressor(SklearnMixin, _GaussianProcessRegressor):
    _cls = _GaussianProcessRegressor
    __init__ = _GaussianProcessRegressor.__init__

