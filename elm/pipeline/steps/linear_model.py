'''
elm.pipeline.steps.linear_model

Wraps sklearn.linear_model for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.linear_model import ARDRegression as _ARDRegression
from sklearn.linear_model import BayesianRidge as _BayesianRidge
from sklearn.linear_model import ElasticNet as _ElasticNet
from sklearn.linear_model import ElasticNetCV as _ElasticNetCV
from sklearn.linear_model import Hinge as _Hinge
from sklearn.linear_model import Huber as _Huber
from sklearn.linear_model import HuberRegressor as _HuberRegressor
from sklearn.linear_model import Lars as _Lars
from sklearn.linear_model import LarsCV as _LarsCV
from sklearn.linear_model import Lasso as _Lasso
from sklearn.linear_model import LassoCV as _LassoCV
from sklearn.linear_model import LassoLars as _LassoLars
from sklearn.linear_model import LassoLarsCV as _LassoLarsCV
from sklearn.linear_model import LassoLarsIC as _LassoLarsIC
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import Log as _Log
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.linear_model import LogisticRegressionCV as _LogisticRegressionCV
from sklearn.linear_model import ModifiedHuber as _ModifiedHuber
from sklearn.linear_model import MultiTaskElasticNet as _MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV as _MultiTaskElasticNetCV
from sklearn.linear_model import MultiTaskLasso as _MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV as _MultiTaskLassoCV
from sklearn.linear_model import OrthogonalMatchingPursuit as _OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV as _OrthogonalMatchingPursuitCV
from sklearn.linear_model import PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor as _PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron as _Perceptron
from sklearn.linear_model import RANSACRegressor as _RANSACRegressor
from sklearn.linear_model import RandomizedLasso as _RandomizedLasso
from sklearn.linear_model import RandomizedLogisticRegression as _RandomizedLogisticRegression
from sklearn.linear_model import Ridge as _Ridge
from sklearn.linear_model import RidgeCV as _RidgeCV
from sklearn.linear_model import RidgeClassifier as _RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV as _RidgeClassifierCV
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.linear_model import SGDRegressor as _SGDRegressor
from sklearn.linear_model import SquaredLoss as _SquaredLoss
from sklearn.linear_model import TheilSenRegressor as _TheilSenRegressor



class ARDRegression(SklearnMixin, _ARDRegression):
    _cls = _ARDRegression
    __init__ = _ARDRegression.__init__



class BayesianRidge(SklearnMixin, _BayesianRidge):
    _cls = _BayesianRidge
    __init__ = _BayesianRidge.__init__



class ElasticNet(SklearnMixin, _ElasticNet):
    _cls = _ElasticNet
    __init__ = _ElasticNet.__init__



class ElasticNetCV(SklearnMixin, _ElasticNetCV):
    _cls = _ElasticNetCV
    __init__ = _ElasticNetCV.__init__



class Hinge(SklearnMixin, _Hinge):
    _cls = _Hinge
    __init__ = _Hinge.__init__



class Huber(SklearnMixin, _Huber):
    _cls = _Huber
    __init__ = _Huber.__init__



class HuberRegressor(SklearnMixin, _HuberRegressor):
    _cls = _HuberRegressor
    __init__ = _HuberRegressor.__init__



class Lars(SklearnMixin, _Lars):
    _cls = _Lars
    __init__ = _Lars.__init__



class LarsCV(SklearnMixin, _LarsCV):
    _cls = _LarsCV
    __init__ = _LarsCV.__init__



class Lasso(SklearnMixin, _Lasso):
    _cls = _Lasso
    __init__ = _Lasso.__init__



class LassoCV(SklearnMixin, _LassoCV):
    _cls = _LassoCV
    __init__ = _LassoCV.__init__



class LassoLars(SklearnMixin, _LassoLars):
    _cls = _LassoLars
    __init__ = _LassoLars.__init__



class LassoLarsCV(SklearnMixin, _LassoLarsCV):
    _cls = _LassoLarsCV
    __init__ = _LassoLarsCV.__init__



class LassoLarsIC(SklearnMixin, _LassoLarsIC):
    _cls = _LassoLarsIC
    __init__ = _LassoLarsIC.__init__



class LinearRegression(SklearnMixin, _LinearRegression):
    _cls = _LinearRegression
    __init__ = _LinearRegression.__init__



class Log(SklearnMixin, _Log):
    _cls = _Log
    __init__ = _Log.__init__



class LogisticRegression(SklearnMixin, _LogisticRegression):
    _cls = _LogisticRegression
    __init__ = _LogisticRegression.__init__



class LogisticRegressionCV(SklearnMixin, _LogisticRegressionCV):
    _cls = _LogisticRegressionCV
    __init__ = _LogisticRegressionCV.__init__



class ModifiedHuber(SklearnMixin, _ModifiedHuber):
    _cls = _ModifiedHuber
    __init__ = _ModifiedHuber.__init__



class MultiTaskElasticNet(SklearnMixin, _MultiTaskElasticNet):
    _cls = _MultiTaskElasticNet
    __init__ = _MultiTaskElasticNet.__init__



class MultiTaskElasticNetCV(SklearnMixin, _MultiTaskElasticNetCV):
    _cls = _MultiTaskElasticNetCV
    __init__ = _MultiTaskElasticNetCV.__init__



class MultiTaskLasso(SklearnMixin, _MultiTaskLasso):
    _cls = _MultiTaskLasso
    __init__ = _MultiTaskLasso.__init__



class MultiTaskLassoCV(SklearnMixin, _MultiTaskLassoCV):
    _cls = _MultiTaskLassoCV
    __init__ = _MultiTaskLassoCV.__init__



class OrthogonalMatchingPursuit(SklearnMixin, _OrthogonalMatchingPursuit):
    _cls = _OrthogonalMatchingPursuit
    __init__ = _OrthogonalMatchingPursuit.__init__



class OrthogonalMatchingPursuitCV(SklearnMixin, _OrthogonalMatchingPursuitCV):
    _cls = _OrthogonalMatchingPursuitCV
    __init__ = _OrthogonalMatchingPursuitCV.__init__



class PassiveAggressiveClassifier(SklearnMixin, _PassiveAggressiveClassifier):
    _cls = _PassiveAggressiveClassifier
    __init__ = _PassiveAggressiveClassifier.__init__



class PassiveAggressiveRegressor(SklearnMixin, _PassiveAggressiveRegressor):
    _cls = _PassiveAggressiveRegressor
    __init__ = _PassiveAggressiveRegressor.__init__



class Perceptron(SklearnMixin, _Perceptron):
    _cls = _Perceptron
    __init__ = _Perceptron.__init__



class RANSACRegressor(SklearnMixin, _RANSACRegressor):
    _cls = _RANSACRegressor
    __init__ = _RANSACRegressor.__init__



class RandomizedLasso(SklearnMixin, _RandomizedLasso):
    _cls = _RandomizedLasso
    __init__ = _RandomizedLasso.__init__



class RandomizedLogisticRegression(SklearnMixin, _RandomizedLogisticRegression):
    _cls = _RandomizedLogisticRegression
    __init__ = _RandomizedLogisticRegression.__init__



class Ridge(SklearnMixin, _Ridge):
    _cls = _Ridge
    __init__ = _Ridge.__init__



class RidgeCV(SklearnMixin, _RidgeCV):
    _cls = _RidgeCV
    __init__ = _RidgeCV.__init__



class RidgeClassifier(SklearnMixin, _RidgeClassifier):
    _cls = _RidgeClassifier
    __init__ = _RidgeClassifier.__init__



class RidgeClassifierCV(SklearnMixin, _RidgeClassifierCV):
    _cls = _RidgeClassifierCV
    __init__ = _RidgeClassifierCV.__init__



class SGDClassifier(SklearnMixin, _SGDClassifier):
    _cls = _SGDClassifier
    __init__ = _SGDClassifier.__init__



class SGDRegressor(SklearnMixin, _SGDRegressor):
    _cls = _SGDRegressor
    __init__ = _SGDRegressor.__init__



class SquaredLoss(SklearnMixin, _SquaredLoss):
    _cls = _SquaredLoss
    __init__ = _SquaredLoss.__init__



class TheilSenRegressor(SklearnMixin, _TheilSenRegressor):
    _cls = _TheilSenRegressor
    __init__ = _TheilSenRegressor.__init__

