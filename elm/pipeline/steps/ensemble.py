'''
elm.pipeline.steps.linear_model

Wraps sklearn.ensemble for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier
from sklearn.ensemble import BaggingRegressor as _BaggingRegressor
from sklearn.ensemble import BaseEnsemble as _BaseEnsemble
from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as _GradientBoostingRegressor
from sklearn.ensemble import IsolationForest as _IsolationForest
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding as _RandomTreesEmbedding
from sklearn.ensemble import VotingClassifier as _VotingClassifier



class AdaBoostClassifier(SklearnMixin, _AdaBoostClassifier):
    _cls = _AdaBoostClassifier
    __init__ = _AdaBoostClassifier.__init__



class AdaBoostRegressor(SklearnMixin, _AdaBoostRegressor):
    _cls = _AdaBoostRegressor
    __init__ = _AdaBoostRegressor.__init__



class BaggingClassifier(SklearnMixin, _BaggingClassifier):
    _cls = _BaggingClassifier
    __init__ = _BaggingClassifier.__init__



class BaggingRegressor(SklearnMixin, _BaggingRegressor):
    _cls = _BaggingRegressor
    __init__ = _BaggingRegressor.__init__



class BaseEnsemble(SklearnMixin, _BaseEnsemble):
    _cls = _BaseEnsemble
    __init__ = _BaseEnsemble.__init__



class ExtraTreesClassifier(SklearnMixin, _ExtraTreesClassifier):
    _cls = _ExtraTreesClassifier
    __init__ = _ExtraTreesClassifier.__init__



class ExtraTreesRegressor(SklearnMixin, _ExtraTreesRegressor):
    _cls = _ExtraTreesRegressor
    __init__ = _ExtraTreesRegressor.__init__



class GradientBoostingClassifier(SklearnMixin, _GradientBoostingClassifier):
    _cls = _GradientBoostingClassifier
    __init__ = _GradientBoostingClassifier.__init__



class GradientBoostingRegressor(SklearnMixin, _GradientBoostingRegressor):
    _cls = _GradientBoostingRegressor
    __init__ = _GradientBoostingRegressor.__init__



class IsolationForest(SklearnMixin, _IsolationForest):
    _cls = _IsolationForest
    __init__ = _IsolationForest.__init__



class RandomForestClassifier(SklearnMixin, _RandomForestClassifier):
    _cls = _RandomForestClassifier
    __init__ = _RandomForestClassifier.__init__



class RandomForestRegressor(SklearnMixin, _RandomForestRegressor):
    _cls = _RandomForestRegressor
    __init__ = _RandomForestRegressor.__init__



class RandomTreesEmbedding(SklearnMixin, _RandomTreesEmbedding):
    _cls = _RandomTreesEmbedding
    __init__ = _RandomTreesEmbedding.__init__



class VotingClassifier(SklearnMixin, _VotingClassifier):
    _cls = _VotingClassifier
    __init__ = _VotingClassifier.__init__

