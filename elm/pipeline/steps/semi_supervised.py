'''
elm.pipeline.steps.linear_model

Wraps sklearn.semi_supervised for usage with xarray.Dataset / xarray_filters.MLDataset

See:
 * http://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised
'''

from elm.mldataset.wrap_sklearn import SklearnMixin
from sklearn.semi_supervised import LabelPropagation as _LabelPropagation
from sklearn.semi_supervised import LabelSpreading as _LabelSpreading



class LabelPropagation(SklearnMixin, _LabelPropagation):
    _cls = _LabelPropagation
    __init__ = _LabelPropagation.__init__



class LabelSpreading(SklearnMixin, _LabelSpreading):
    _cls = _LabelSpreading
    __init__ = _LabelSpreading.__init__

