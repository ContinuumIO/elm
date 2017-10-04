from __future__ import absolute_import, division, print_function, unicode_literals
from argparse import Namespace
from importlib import import_module
import sklearn
from sklearn.base import BaseEstimator

from elm.mldataset.wrap_sklearn import SklearnMixin

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
        'BaseEstimator', 'KERNEL_PARAMS', 'Pipeline',
        'Parallel', 'RegressorMixin', 'ClassifierMixin', 'ABCMeta',
        'TransformerMixin', 'VBGMM', 'RandomizedPCA', 'GMM',
        'MultiOutputEstimator')

def get_module_classes(m):
    module =  import_module('sklearn.{}'.format(m))
    attrs = tuple(_ for _ in dir(module)
                  if not _.startswith('_')
                  and _[0].isupper()
                  and not any(s in _ for s in SKIP))
    return {attr: getattr(module, attr) for attr in attrs}


def patch_cls(cls):

    class Wrapped(SklearnMixin, cls):
        _cls = cls
        __init__ = cls.__init__
        _cls_name = cls.__name__
    name = 'Elm{}'.format(cls.__name__)
    globals()[name] = Wrapped
    return globals()[name]


_all = []
_seen = set()
ALL_STEPS = {}
for m in MODULES:
    this_module = dict()
    for cls in get_module_classes(m).values():
        if cls.__name__ in _seen:
            continue
        _seen.add(cls.__name__)
        w = patch_cls(cls)
        if any(s in cls.__name__ for s in SKIP):
            continue
        this_module[cls.__name__] = w
        ALL_STEPS[(m, cls.__name__)] = w
    this_module = Namespace(**this_module)
    if m == 'cluster.bicluster':
        bicluster = this_module # special case (dotted name)
        continue
    globals()[m] = this_module
    _all.append(m)
    for name, estimator in vars(this_module).items():
        ALL_STEPS[(m, name)] = estimator

vars(cluster)['bicluster'] = bicluster
__all__ = [ 'patch_cls'] + _all
del _all
del m
del this_module
del w
del _seen