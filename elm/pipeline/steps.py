from importlib import import_module
import sklearn
from sklearn.base import BaseEstimator

from elm.model_selection.sklearn_mldataset import SklearnMixin

MODULES = ['calibration', 'cluster', 'cluster.bicluster',
           'covariance', 'cross_decomposition',
           'decomposition', 'discriminant_analysis',
           'dummy', 'ensemble',
           'feature_extraction', 'feature_selection',
           'gaussian_process', 'isotonic',
           'kernel_approximation', 'kernel_ridge',
           'linear_model', 'manifold', 'model_selection',
           'mixture', 'model_selection',
           'multiclass', 'multioutput',
           'naive_bayes', 'neighbors',
           'neural_network', 'pipeline',
           'preprocessing', 'random_projection',
           'semi_supervised', 'svm', 'tree']

SKIP = ('SearchCV', 'ParameterGrid', 'ParameterSampler',
        'BaseEstimator', 'KERNEL_PARAMS', 'Pipeline')

def get_module_classes(m):
    module =  import_module('sklearn.{}'.format(m))
    attrs = tuple(_ for _ in dir(module)
                  if not _.startswith('_')
                  and _[0].isupper()
                  and not any(s in _ for s in SKIP))
    return {attr: getattr(module, attr) for attr in attrs}


DELEGATES = ('_final_estimator', 'best_estimator_')

def patch_cls(cls):

    class Wrapped(SklearnMixin, cls):
        _cls = cls
        __init__ = cls.__init__
        __name__ = cls.__name__
        '''if hasattr(cls, '_final_estimator'):
            _final_estimator = cls._final_estimator
        if hasattr(cls, 'best_estimator_'):
            cls.best_estimator_ = cls.best_estimator_'''
    return Wrapped





_all = []
for m in MODULES:
    for cls in get_module_classes(m).values():
        w = patch_cls(cls)
        if any(s in cls.__name__ for s in SKIP):
            continue
        globals()[cls.__name__] = w
        _all.append(cls.__name__)

__all__ = [ 'patch_cls'] + _all