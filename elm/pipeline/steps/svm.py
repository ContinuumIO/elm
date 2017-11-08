'''
elm.pipeline.steps.svm

Wraps sklearn.svm for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.svm import LinearSVC as _LinearSVC
from sklearn.svm import LinearSVR as _LinearSVR
from sklearn.svm import NuSVC as _NuSVC
from sklearn.svm import NuSVR as _NuSVR
from sklearn.svm import OneClassSVM as _OneClassSVM
from sklearn.svm import SVC as _SVC
from sklearn.svm import SVR as _SVR



class LinearSVC(SklearnMixin, _LinearSVC):
    _cls = _LinearSVC
    __init__ = _LinearSVC.__init__



class LinearSVR(SklearnMixin, _LinearSVR):
    _cls = _LinearSVR
    __init__ = _LinearSVR.__init__



class NuSVC(SklearnMixin, _NuSVC):
    _cls = _NuSVC
    __init__ = _NuSVC.__init__



class NuSVR(SklearnMixin, _NuSVR):
    _cls = _NuSVR
    __init__ = _NuSVR.__init__



class OneClassSVM(SklearnMixin, _OneClassSVM):
    _cls = _OneClassSVM
    __init__ = _OneClassSVM.__init__



class SVC(SklearnMixin, _SVC):
    _cls = _SVC
    __init__ = _SVC.__init__



class SVR(SklearnMixin, _SVR):
    _cls = _SVR
    __init__ = _SVR.__init__

