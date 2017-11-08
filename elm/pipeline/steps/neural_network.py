'''
elm.pipeline.steps.linear_model

Wraps sklearn.neural_network for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.neural_network import BernoulliRBM as _BernoulliRBM
from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sklearn.neural_network import MLPRegressor as _MLPRegressor



class BernoulliRBM(SklearnMixin, _BernoulliRBM):
    _cls = _BernoulliRBM
    __init__ = _BernoulliRBM.__init__



class MLPClassifier(SklearnMixin, _MLPClassifier):
    _cls = _MLPClassifier
    __init__ = _MLPClassifier.__init__



class MLPRegressor(SklearnMixin, _MLPRegressor):
    _cls = _MLPRegressor
    __init__ = _MLPRegressor.__init__

