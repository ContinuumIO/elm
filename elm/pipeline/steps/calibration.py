'''
elm.pipeline.steps.linear_model

Wraps sklearn.calibration for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.calibration import CalibratedClassifierCV as _CalibratedClassifierCV



class CalibratedClassifierCV(SklearnMixin, _CalibratedClassifierCV):
    _cls = _CalibratedClassifierCV
    __init__ = _CalibratedClassifierCV.__init__

