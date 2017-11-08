'''
elm.pipeline.steps.kernel_approximation

Wraps sklearn.kernel_approximation for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.kernel_approximation import AdditiveChi2Sampler as _AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem as _Nystroem
from sklearn.kernel_approximation import RBFSampler as _RBFSampler
from sklearn.kernel_approximation import SkewedChi2Sampler as _SkewedChi2Sampler



class AdditiveChi2Sampler(SklearnMixin, _AdditiveChi2Sampler):
    _cls = _AdditiveChi2Sampler
    __init__ = _AdditiveChi2Sampler.__init__



class Nystroem(SklearnMixin, _Nystroem):
    _cls = _Nystroem
    __init__ = _Nystroem.__init__



class RBFSampler(SklearnMixin, _RBFSampler):
    _cls = _RBFSampler
    __init__ = _RBFSampler.__init__



class SkewedChi2Sampler(SklearnMixin, _SkewedChi2Sampler):
    _cls = _SkewedChi2Sampler
    __init__ = _SkewedChi2Sampler.__init__

