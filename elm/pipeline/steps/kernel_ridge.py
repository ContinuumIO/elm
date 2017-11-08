'''
elm.pipeline.steps.linear_model

Wraps sklearn.kernel_ridge for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.kernel_ridge import KernelRidge as _KernelRidge



class KernelRidge(SklearnMixin, _KernelRidge):
    _cls = _KernelRidge
    __init__ = _KernelRidge.__init__

